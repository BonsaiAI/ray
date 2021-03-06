#!/usr/bin/env python

import argparse
import csv
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import List, Any, Dict, Optional

import h5py
import numpy as np
from pathlib import Path
import shelve

import ray
from ray.rllib.agents.trainer import COMMON_CONFIG
from ray.rllib.evaluation import SampleBatchBuilder
from ray.rllib.offline import JsonWriter, IOContext
from ray.tune import Callback
from ray.tune.checkpoint_manager import Checkpoint
from ray.tune.config_parser import make_parser
from ray.tune.stopper import TimeoutStopper
from ray.tune.trial import Trial
from ray.rllib import train, rollout, SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch

try:
    class_name = get_ipython().__class__.__name__
    IS_NOTEBOOK = True if "Terminal" not in class_name else False
except NameError:
    IS_NOTEBOOK = False

# Try to import both backends for flag checking/warnings.
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

log = logging.getLogger(__name__)


EPILOG = """
Train a reinforcement learning agent, create a checkpoint of the final policy that
reach the evaluation threshold desired and rollout the policy to populate the desire
datasets.

Five datasets are generated by default:
- Expert: the policy to rollout is trained up to the expert level provided.
- Medium: the policy to rollout is trained up to 50% of the expert level provided.
- Random: the policy to rollout is the initial random policy obtained after the first
          iteration of training.
- Expert-Medium: the dataset contains half of the episodes in the Expert dataset and
                 half of the episodes in the Medium dataset.
- Medium-Random: the dataset contains half of the episodes in the Medium dataset and
                 half of the episodes in the Random dataset.

Note that same parameters of the train and rollout command are allowed but some of
them could be overridden. The following parameters are automatically modified:
* --config: the following values are overridden:
            - evaluation_interval = 1.
            - evaluation_reward_threshold = policy level expected
* --stop: evaluation.episode_reward_mean is checked and nothing else.
* --use-shelve: always set to true.
* checkpoint: checkpoint for rollout is automatically set.
* --out: output directory for rollouts is set to the same checkpoint directory.

Also consider that YML support multiple experiments defined and run concurrently but
that is not supported by this command.

Training example via RLlib CLI:
    rllib dataset 250 --run PPO --env MoabMoveToCenterSim-v0 --steps 1000000

Moab example via RLlib CLI:
    rllib dataset 250 -f tuned_examples/moab-ppo.yaml --steps 1000000

Moab example via executable:
    ./dataset.py 250 -f tuned_examples/moab-ppo.yaml

Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate datasets for Offline RL experiments.",
        epilog=EPILOG)

    parser.add_argument(
        "expert_level",
        type=float,
        help="Expected average reward of an expert level policy. "
             "This will be used to set episode_reward_mean stop criteria.")
    parser.add_argument(
        "--no-medium",
        action="store_true",
        help="Whether to not generate the Medium dataset.")
    parser.add_argument(
        "--no-random",
        action="store_true",
        help="Whether to not generate the Random dataset.")
    parser.add_argument(
        "--no-expert-medium",
        action="store_true",
        help="Whether to not generate the Expert-Medium dataset.")
    parser.add_argument(
        "--no-medium-random",
        action="store_true",
        help="Whether to not generate the Medium-Random dataset.")
    parser.add_argument(
        "--train-timeout",
        default=3600,
        type=int,
        help="Train timeout in seconds.")

    # See the base parsers definition in:
    # - ray/tune/config_parser.py
    # - ray/rllib/train.py
    # - ray/rllib/rollout.py
    parser = train.create_parser(parser_creator=parser_creator,
                                 pre_created_parser=parser)
    parser = rollout.create_parser(parser_creator=parser_creator,
                                   pre_created_parser=parser)

    return parser


class _ExperimentCallback(TimeoutStopper, Callback):
    def __init__(self, policy_name, timeout, evaluation_reward_level):
        super().__init__(timeout)
        self.policy_name = policy_name
        self.evaluation_reward_level = evaluation_reward_level
        self.last_iteration = None
        self.last_trial = None
        self.last_checkpoint = None

    def __call__(self, trial_id, result):
        if ("evaluation" in result and
            result["evaluation"]["episode_reward_mean"] >= self.evaluation_reward_level):
            return True
        return False

    def on_checkpoint(self, iteration: int, trials: List["Trial"],
                      trial: "Trial", checkpoint: Checkpoint, **info):
        self.last_iteration = iteration
        self.last_trial = trial
        self.last_checkpoint = checkpoint


class _ProgressReport:
    def __init__(self):
        self._messages = []
        self._errors = []
        self.directory = "/tmp"
        self.run = "alg"
        self.experiment = "env"
        self.train_batch_size = COMMON_CONFIG["train_batch_size"]
        self.num_workers = COMMON_CONFIG["num_workers"]
        self.output_max_file_size = COMMON_CONFIG["output_max_file_size"]
        self.output_compress_columns = COMMON_CONFIG["output_compress_columns"]

    def report(self, message):
        print(message)
        self._messages.append(message)

    def report_error(self, message: str, ex: BaseException = None):
        print("Error: " + message)
        log.error(message, exc_info=ex)
        self._errors.append((message, ex))

    def save(self):
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{self.run}_{self.experiment}_{now_str}.txt"
        file_path = Path(self.directory).absolute().joinpath(file_name)
        all_content = self._messages + ["Error List:"] + self._errors
        with open(file_path, "w") as file:
            file.write("\n".join(all_content))


OMISSION_COLUMN_NAME = "exclude"

class _DataSync:
    RLLIB_HEADER = [SampleBatch.OBS, SampleBatch.ACTIONS, SampleBatch.NEXT_OBS,
                    SampleBatch.REWARDS, SampleBatch.DONES]
    D4RL_HEADER = ["observations", "actions", OMISSION_COLUMN_NAME, "rewards",
                   "terminals"]
    D4RL_OMISSION_COLUMN_INDEXES = [i for i, c in enumerate(RLLIB_HEADER)
                                    if c == OMISSION_COLUMN_NAME]
    REWARD_COLUMN_NAME = "reward"
    REQUIRED_HEADER = ["state", "action", OMISSION_COLUMN_NAME, REWARD_COLUMN_NAME,
                       "done"]
    REWARD_INDEX = REQUIRED_HEADER.index(REWARD_COLUMN_NAME)
    OMISSION_COLUMN_INDEXES = [i for i, c in enumerate(REQUIRED_HEADER)
                               if c == OMISSION_COLUMN_NAME]

    def __init__(self, dataset_id: str, rollout_dir: Path, reporter: _ProgressReport):
        self._dataset_id = dataset_id
        self._rollout_dir = rollout_dir
        self._reporter = reporter
        self._file_name = str(self._rollout_dir.joinpath(
            f"{self._dataset_id}-rollout.{self._file_extension()}"))

    @property
    def file_name(self):
        return self._file_name

    @classmethod
    def _file_extension(cls):
        raise NotImplementedError

    def process_episode(self, episode_index: int, episode: List[List[Any]]):
        raise NotImplementedError

    def process_iteration(self, iteration_index: int, iteration: List[Any]):
        raise NotImplementedError

    def close(self, total_episodes_count: int):
        raise NotImplementedError


class _CSVSync(_DataSync):
    def __init__(self, dataset_id: str, rollout_dir: Path, reporter: _ProgressReport):
        super().__init__(dataset_id, rollout_dir, reporter)
        self._csv_file = open(self._file_name, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        self._expanded_header = []

    @classmethod
    def _file_extension(cls):
        return "csv"

    def process_episode(self, episode_index: int, episode: Dict[str, List[Any]]):
        if not self._expanded_header:
            first_iteration = episode[0][:len(self.REQUIRED_HEADER)]
            for i, meta_column, first_cell in zip(range(len(self.REQUIRED_HEADER)),
                                               self.REQUIRED_HEADER, first_iteration):
                if i not in self.OMISSION_COLUMN_INDEXES:
                    if isinstance(first_cell, (list, np.ndarray)):
                        for i in range(len(first_cell)):
                            column = f"{meta_column}.{i}"
                            self._expanded_header.append(column)
                    else:
                        self._expanded_header.append(meta_column)
            self._csv_writer.writerow(self._expanded_header)

    def process_iteration(self, iteration_index: int, iteration: List[Any]):
        expanded_iteration_row = []
        for i, field in enumerate(iteration[:len(self.REQUIRED_HEADER)]):
            if i not in self.OMISSION_COLUMN_INDEXES:
                if isinstance(field, np.ndarray):
                    expanded_iteration_row.extend(field.tolist())
                elif isinstance(field, list):
                    expanded_iteration_row.extend(field)
                elif isinstance(field, np.number):
                    expanded_iteration_row.append(field.item())
                else:
                    expanded_iteration_row.append(field)
        if len(expanded_iteration_row) == len(self._expanded_header):
            self._csv_writer.writerow(expanded_iteration_row)
        else:
            self._reporter.report_error("One iteration has a shape different to "
                                        "the first one used as reference for the "
                                        f"header: expected {len(self._expanded_header)}"
                                        f", found {len(expanded_iteration_row)}")

    def close(self, total_episodes_count: int):
        self._csv_file.close()


class _ShelveSync(_DataSync):
    def __init__(self, dataset_id: str, rollout_dir: Path, reporter: _ProgressReport):
        super().__init__(dataset_id, rollout_dir, reporter)
        self._shelve = shelve.open(self._file_name)
        self._shelve["num_episodes"] = 0

    @classmethod
    def _file_extension(cls):
        return "shelve"

    def process_episode(self, episode_index: int, episode: List[List[Any]]):
        self._shelve[str(episode_index)] = episode

    def process_iteration(self, iteration_index: int, iteration: List[Any]):
        # Shelve can work at episode level only
        pass

    def close(self, total_episodes_count: int):
        self._shelve["num_episodes"] = total_episodes_count
        self._shelve.close()


class _JsonSync(_DataSync):
    def __init__(self, dataset_id: str, rollout_dir: Path, reporter: _ProgressReport):
        super().__init__(dataset_id, rollout_dir, reporter)
        self._writers = [JsonWriter(str(self.file_name),
                                    IOContext(worker_index=i),
                                    reporter.output_max_file_size,
                                    reporter.output_compress_columns)
                         for i in range(reporter.num_workers)]
        self._batch_builder = SampleBatchBuilder()
        self._current_writer_index = 0

    @property
    def _current_writer(self):
        return self._writers[self._current_writer_index]

    @classmethod
    def _file_extension(cls):
        return "json"

    def process_episode(self, episode_index: int, episode: List[List[Any]]):
        if self._batch_builder.count >= self._reporter.train_batch_size:
            sample_batch = self._batch_builder.build_and_reset()
            self._current_writer.write(sample_batch)
            self._current_writer_index += 1
            if self._current_writer_index >= len(self._writers):
                self._current_writer_index = 0

    def process_iteration(self, iteration_index: int, iteration: List[Any]):
        values = {k: v for k, v in
                  zip(self.RLLIB_HEADER, iteration[:len(self.RLLIB_HEADER)])}
        self._batch_builder.add_values(**values)

    def close(self, total_episodes_count: int):
        if self._batch_builder.count > 0:
            sample_batch = self._batch_builder.build_and_reset()
            self._current_writer.write(sample_batch)
        for w in self._writers:
            if w.cur_file:
                w.cur_file.close()


class _H5Sync(_DataSync):
    def __init__(self, dataset_id: str, rollout_dir: Path, reporter: _ProgressReport):
        super().__init__(dataset_id, rollout_dir, reporter)
        self._dataset_file = h5py.File(self._file_name, "x")
        self._datasets = {}
        self._columns_by_index = {}
        self._converter_by_index = {}
        self._episode_data = {}

    @classmethod
    def _file_extension(cls):
        return "hdf5"

    def process_episode(self, episode_index: int, episode: List[List[Any]]):
        def get_scalar_type(i, v):
            if isinstance(v, bool):
                self._converter_by_index[i] = np.bool_
                return "?"
            elif isinstance(v, int):
                self._converter_by_index[i] = np.int_
                return "i"
            else:
                self._converter_by_index[i] = np.float_
                return "f"

        if not self._datasets:
            first_iteration = episode[0][:len(self.D4RL_HEADER)]
            for i, meta_column, first_cell in zip(range(len(self.D4RL_HEADER)),
                                               self.D4RL_HEADER, first_iteration):
                if i not in self.D4RL_OMISSION_COLUMN_INDEXES:
                    self._columns_by_index[i] = meta_column
                    maxshape = [None]
                    shape = [len(episode)]
                    dtype = "f"
                    if isinstance(first_cell, np.ndarray):
                        shape += list(first_cell.shape)
                        maxshape += list(first_cell.shape)
                        dtype = first_cell.dtype
                        self._converter_by_index[i] = None
                    elif isinstance(first_cell, list):
                        shape.append(len(first_cell))
                        maxshape.append(len(first_cell))
                        dtype = get_scalar_type(i, first_cell[0])
                    elif isinstance(first_cell, (np.number, np.bool_)):
                        dtype = first_cell.dtype
                        self._converter_by_index[i] = None
                    else:
                        dtype = get_scalar_type(i, first_cell)

                    self._datasets[meta_column] = self._dataset_file.create_dataset(
                        meta_column, shape=tuple(shape), dtype=dtype,
                        maxshape=tuple(maxshape))
                    self._episode_data[meta_column] = []
        else:
            self._flush_episode()
            for ds in self._datasets.values():
                ds.resize((ds.shape[0] + len(episode)), axis=0)

    def process_iteration(self, iteration_index: int, iteration: List[Any]):
        for i, field in enumerate(iteration):
            if i in self._columns_by_index:
                meta_column = self._columns_by_index[i]
                value = (field if not self._converter_by_index[i] else
                         self._converter_by_index[i](field))
                self._episode_data[meta_column].append(value)

    def _flush_episode(self):
        if self._episode_data:
            for meta_column, data in list(self._episode_data.items()):
                if data:
                    self._datasets[meta_column][-len(data):] = np.array(data)
                    self._episode_data[meta_column].clear()

    def close(self, total_episodes_count: int):
        self._flush_episode()
        self._dataset_file.close()


class _Dataset:
    def __init__(self, external_shelve_file: Optional[str], dataset_id: str,
                 rollout_dir: Path, reporter: _ProgressReport):
        self.external_shelve_file = external_shelve_file
        self.include_shelve = external_shelve_file is None
        self._dataset_id = dataset_id
        self._rollout_dir = rollout_dir
        self._reporter = reporter
        self.csv_sync = _CSVSync(self._dataset_id, self._rollout_dir, self._reporter)
        self.json_sync = _JsonSync(self._dataset_id, self._rollout_dir, self._reporter)
        self.h5_sync = _H5Sync(self._dataset_id, self._rollout_dir, self._reporter)
        self._syncs: List[_DataSync] = [self.csv_sync, self.json_sync, self.h5_sync]
        self.shelve_sync = None
        if self.include_shelve:
            self.shelve_sync = _ShelveSync(self._dataset_id, self._rollout_dir,
                                           self._reporter)
            self._syncs.append(self.shelve_sync)
        self._rewards = []
        self._episode_lengths = []
        self._episodes_count = 0

    def process_episode(self, episode_index: int, episode: List[Any]):
        self._episodes_count += 1
        episode_length = len(episode)
        self._episode_lengths.append(episode_length)
        episode_total_reward = 0
        for sync in self._syncs:
            sync.process_episode(episode_index, episode)
        for iteration_index, iteration in enumerate(episode):
            for sync in self._syncs:
                sync.process_iteration(iteration_index, iteration)
            reward_value = iteration[_DataSync.REWARD_INDEX]
            if not isinstance(reward_value, (list, np.ndarray)):
                if isinstance(reward_value, np.number):
                    reward_value = reward_value.item()
                episode_total_reward += reward_value
            else:
                self._reporter.report_error("We got an iterable field for reward")
        self._rewards.append(episode_total_reward)

    def get_reward_mean(self) -> float:
        return np.mean(self._rewards).item()

    def get_episode_length_mean(self) -> float:
        return np.mean(self._episode_lengths).item()

    def get_total_episodes_count(self) -> int:
        return self._episodes_count

    def get_total_iterations_count(self) -> int:
        return sum(self._episode_lengths)

    def get_files_generated(self) -> List[str]:
        files_generated = [s.file_name for s in self._syncs]
        if not self.include_shelve:
            files_generated.append(self.external_shelve_file)
        return files_generated

    def close(self):
        for sync in self._syncs:
            sync.close(self.get_total_episodes_count())

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def run(args, parser):
    reporter = _ProgressReport()
    start_time = time.perf_counter()
    last_time = start_time
    e_level = args.expert_level
    m_level = args.expert_level / 2
    r_level = -sys.float_info.max
    datasets_names = {"e": "Expert", "m": "Medium",
                      "r": "Random", "em": "Expert-Medium",
                      "mr": "Medium-Random"}
    datasets_times = {}
    datasets = {}
    generate_em_dataset = False
    generate_mr_dataset = False
    policies_to_train = {"e": e_level}
    if not args.no_medium:
        policies_to_train["m"] = m_level
    if not args.no_random:
        policies_to_train["r"] = r_level
    if not args.no_expert_medium:
        generate_em_dataset = True
        policies_to_train["m"] = m_level
    if not args.no_medium_random:
        generate_mr_dataset = True
        policies_to_train["m"] = m_level
        policies_to_train["r"] = r_level
    args.use_shelve = True
    timeout = args.train_timeout
    run_config = None
    em_dataset = None
    mr_dataset = None
    for p, level in policies_to_train.items():
        policy_name = datasets_names[p]
        reporter.report(f"Initiating experiments of Policy for level {policy_name}")
        reporter.report(f"Starting Training with level {level}")
        callback = _ExperimentCallback(policy_name, timeout, level)
        experiments = {}
        def experiment_handler(experiment_name, experiment):
            if not experiments:
                reporter.experiment = experiment_name
                reporter.directory = experiment["local_dir"]
                reporter.run = experiment["run"]
                if "config" in experiment and "train_batch_size" in experiment["config"]:
                    reporter.train_batch_size = experiment["config"]["train_batch_size"]
                if "config" in experiment and "num_workers" in experiment["config"]:
                    reporter.num_workers = experiment["config"]["num_workers"]
                if "config" in experiment and "output_max_file_size" in experiment["config"]:
                    reporter.output_max_file_size = experiment["config"]["output_max_file_size"]
                if "config" in experiment and "output_compress_columns" in experiment["config"]:
                    reporter.output_compress_columns = experiment["config"]["output_compress_columns"]
            experiment["config"]["evaluation_interval"] = 1
            experiment["config"]["evaluation_reward_threshold"] = level
            experiment["stop"] = callback
            experiments[experiment_name] = experiment
            return experiment

        train.run(args, parser, [callback], experiment_handler, False)
        checkpoint = callback.last_checkpoint.value
        reporter.report(f"Starting Rollouts with checkpoint {checkpoint}")
        rollout_dir = Path(callback.last_checkpoint.value).parent
        if generate_em_dataset and p in ["e", "m"] and not em_dataset:
            em_dataset = _Dataset(None, "em", rollout_dir, reporter)
            datasets["em"] = em_dataset
        if generate_mr_dataset and p in ["m", "r"] and not mr_dataset:
            mr_dataset = _Dataset(None, "mr", rollout_dir, reporter)
            datasets["mr"] = mr_dataset
        file_name = str(rollout_dir.joinpath(f"{p}-rollout.shelve"))
        args.out = file_name
        if not run_config:
            run_config = list(experiments.values())[0]["run"]
        args.run = run_config
        rollout.run(args, parser, checkpoint)
        with _Dataset(file_name, p, rollout_dir, reporter) as current_dataset:
            datasets[p] = current_dataset
            with shelve.open(file_name) as rollouts:
                if rollouts["num_episodes"]:
                    try:
                        episodes_count = rollouts["num_episodes"]
                        for episode_index in range(episodes_count):
                            episode = rollouts[str(episode_index)]
                            current_dataset.process_episode(episode_index, episode)
                            if (generate_em_dataset and p in ["e", "m"]
                                and episode_index <= (episodes_count / 2)):
                                em_dataset.process_episode(episode_index, episode)
                            if (generate_mr_dataset and p in ["m", "r"]
                                and episode_index <= (episodes_count / 2)):
                                mr_dataset.process_episode(episode_index, episode)
                    except Exception as ex:
                        log.error("Error parsing rollouts file!!!", exc_info=ex)
                else:
                    log.error("No rollouts data found!!!")
        reporter.report(f"Rollouts for Policy for level {policy_name} where save in:")
        reporter.report(f"- Shelve: {file_name}")
        reporter.report(f"- CSV: {current_dataset.csv_sync.file_name}")
        p_finish_time = time.perf_counter()
        p_total_minutes = (p_finish_time - last_time) / 60
        last_time = p_finish_time
        datasets_times[p] = p_total_minutes
        reporter.report(f"Took: {p_total_minutes} minutes")

    datasets_reward_mean = {}
    datasets_episode_len_mean = {}
    datasets_episodes_count = {}
    datasets_iterations_count = {}
    datasets_files_generated = {}
    for p, ds in datasets.items():
        datasets_reward_mean[p] = ds.get_reward_mean()
        datasets_episode_len_mean[p] = ds.get_episode_length_mean()
        datasets_episodes_count[p] = ds.get_total_episodes_count()
        datasets_iterations_count[p] = ds.get_total_iterations_count()
        datasets_files_generated[p] = ds.get_files_generated()

    if em_dataset:
        em_dataset.close()
    if mr_dataset:
        mr_dataset.close()

    reporter.report("Datasets generated!!!")
    reporter.report("Datasets stats:")
    stats_header = OrderedDict()
    stats_header["Dataset"] = "s"
    stats_header["Mean Reward"] = "f"
    stats_header["Mean Episode Length"] = "f"
    stats_header["Episode Count"] = "d"
    stats_header["Iterations Count"] = "d"
    format_width = max([len(s) for s in stats_header.keys()]) + 2
    base_header_format = f"{{:{format_width}s}}"
    header_format = base_header_format
    for _ in range(len(stats_header)-1):
        header_format += (" | " + base_header_format)
    reporter.report(header_format.format(*list(stats_header.keys())))
    int_format = f"{{:{format_width}d}}"
    float_format = f"{{:{format_width}.4f}}"
    def get_format(format_id):
        if format_id == "f":
            return float_format
        elif format_id == "d":
            return int_format
        else:
            return base_header_format

    stats_types = list(stats_header.values())
    stats_format = get_format(stats_types.pop(0))
    for t in stats_types:
        stats_format += (" | " + get_format(t))
    for p in datasets_reward_mean:
        row = [datasets_names[p],
               datasets_reward_mean[p],
               datasets_episode_len_mean[p],
               datasets_episodes_count[p],
               datasets_iterations_count[p],
               ]
        reporter.report(stats_format.format(*row))
    for p in datasets_files_generated:
        reporter.report(f"Dataset files for {datasets_names[p]}:")
        reporter.report(f"  - Shelve: {datasets_files_generated[p][0]}")
        reporter.report(f"  - CSV: {datasets_files_generated[p][1]}")
    finish_time = time.perf_counter()
    total_minutes = (finish_time - start_time) / 60
    reporter.report(f"The whole process took: {total_minutes} minutes.")
    for p in datasets_times:
        reporter.report(f"- {datasets_names[p]} took: {datasets_times[p]} minutes.")
    reporter.save()
    ray.shutdown()



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
