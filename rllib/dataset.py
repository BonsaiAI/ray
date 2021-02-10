#!/usr/bin/env python

import argparse
import csv
import logging
import math
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import List

import numpy as np
from pathlib import Path
import shelve

import ray
from ray.tune import Callback
from ray.tune.checkpoint_manager import Checkpoint
from ray.tune.config_parser import make_parser
from ray.tune.stopper import TimeoutStopper
from ray.tune.trial import Trial
from ray.rllib import train, rollout
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
        self.directory = "/tmp"
        self.run = "alg"
        self.experiment = "env"

    def report(self, message):
        print(message)
        self._messages.append(message)

    def save(self):
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{self.run}_{self.experiment}_{now_str}.txt"
        file_path = Path(self.directory).absolute().joinpath(file_name)
        with open(file_path, "w") as file:
            file.write("\n".join(self._messages))


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
    datasets_reward_mean = {}
    datasets_episode_len_mean = {}
    datasets_episodes_count = {}
    datasets_iterations_count = {}
    datasets_files_generated = {}
    datasets_times = {}
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
    em_csv_file = None
    mr_csv_file = None
    em_csv_writer = None
    mr_csv_writer = None
    em_shelve = None
    mr_shelve = None
    em_rewards = []
    em_episode_lengths = []
    em_episodes_counts = []
    em_iterations_counts = []
    mr_rewards = []
    mr_episode_lengths = []
    mr_episodes_counts = []
    mr_iterations_counts = []
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
            experiment["config"]["evaluation_interval"] = 1
            experiment["config"]["evaluation_reward_threshold"] = level
            experiment["stop"] = callback
            experiments[experiment_name] = experiment
            return experiment

        train.run(args, parser, [callback], experiment_handler, False)
        checkpoint = callback.last_checkpoint.value
        reporter.report(f"Starting Rollouts with checkpoint {checkpoint}")
        rollout_dir = Path(callback.last_checkpoint.value).parent
        extra_csv_writers = []
        extra_shelves = []
        extra_rewards = []
        extra_episode_lengths = []
        extra_episodes_counts = []
        extra_iterations_counts = []
        if generate_em_dataset and p in ["e", "m"] and not em_csv_file:
            em_file_name = str(rollout_dir.joinpath("em-rollout.shelve"))
            em_csv_file_name = str(rollout_dir.joinpath("em-rollout.csv"))
            datasets_files_generated["em"] = [em_file_name, em_csv_file_name]
            em_csv_file = open(em_csv_file_name, 'w', newline='')
            em_csv_writer = csv.writer(em_csv_file)
            em_shelve = shelve.open(em_file_name)
            em_shelve["num_episodes"] = 0
            extra_csv_writers.append(em_csv_writer)
            extra_shelves.append(em_shelve)
            extra_rewards.append(em_rewards)
            extra_episode_lengths.append(em_episode_lengths)
            extra_episodes_counts.append(em_episodes_counts)
            extra_iterations_counts.append(em_iterations_counts)
        if generate_mr_dataset and p in ["m", "r"] and not mr_csv_file:
            mr_file_name = str(rollout_dir.joinpath("mr-rollout.shelve"))
            mr_csv_file_name = str(rollout_dir.joinpath("mr-rollout.csv"))
            datasets_files_generated["mr"] = [mr_file_name, mr_csv_file_name]
            mr_csv_file = open(mr_csv_file_name, 'w', newline='')
            mr_csv_writer = csv.writer(mr_csv_file)
            mr_shelve = shelve.open(mr_file_name)
            mr_shelve["num_episodes"] = 0
            extra_csv_writers.append(mr_csv_writer)
            extra_shelves.append(mr_shelve)
            extra_rewards.append(mr_rewards)
            extra_episode_lengths.append(mr_episode_lengths)
            extra_episodes_counts.append(mr_episodes_counts)
            extra_iterations_counts.append(mr_iterations_counts)
        file_name = str(rollout_dir.joinpath(f"{p}-rollout.shelve"))
        args.out = file_name
        if not run_config:
            run_config = list(experiments.values())[0]["run"]
        args.run = run_config
        rollout.run(args, parser, checkpoint)
        csv_file_name = str(rollout_dir.joinpath(f"{p}-rollout.csv"))
        datasets_files_generated[p] = [file_name, csv_file_name]
        required_header = ["obs", "action", "next_obs", "reward", "done"]
        with open(csv_file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            with shelve.open(file_name) as rollouts:
                if rollouts["num_episodes"]:
                    try:
                        first_iteration = rollouts[str(0)][0][:len(required_header)]
                        expanded_header = []
                        for meta_column, first_cell in zip(required_header, first_iteration):
                            if isinstance(first_cell, (list, np.ndarray)):
                                for i in range(len(first_cell)):
                                    column = f"{meta_column}.{i}"
                                    expanded_header.append(column)
                            else:
                                expanded_header.append(meta_column)
                        writer.writerow(expanded_header)
                        episode_rewards = []
                        episode_lengths = []
                        episodes_count = rollouts["num_episodes"]
                        if extra_shelves:
                            for sh in extra_shelves:
                                sh["num_episodes"] += math.ceil(episodes_count / 2)
                        datasets_episodes_count[p] = episodes_count
                        if extra_episodes_counts:
                            for eec in extra_episodes_counts:
                                eec.append(episodes_count)
                        iterations_count = 0
                        for episode_index in range(episodes_count):
                            episode = rollouts[str(episode_index)]
                            episode_len = len(episode)
                            episode_lengths.append(episode_len)
                            iterations_count += episode_len
                            episode_total_reward = 0
                            if extra_shelves and episode_index <= (episodes_count / 2):
                                for sh in extra_shelves:
                                    sh[str(episode_index)] = episode
                            for iteration in episode:
                                expanded_iteration_row = []
                                for i, field in enumerate(iteration[:len(required_header)]):
                                    is_iterable_field = False
                                    if isinstance(field, np.ndarray):
                                        expanded_iteration_row.extend(field.tolist())
                                        is_iterable_field = True
                                    elif isinstance(field, list):
                                        expanded_iteration_row.extend(field)
                                        is_iterable_field = True
                                    elif isinstance(field, np.number):
                                        expanded_iteration_row.append(field.item())
                                    else:
                                        expanded_iteration_row.append(field)
                                    if i == 3:
                                        if not is_iterable_field:
                                            episode_total_reward += expanded_iteration_row[-1]
                                        else:
                                            log.error("We got an iterable field for reward")
                                if len(expanded_iteration_row) == len(expanded_header):
                                    writer.writerow(expanded_iteration_row)
                                    if extra_csv_writers and episode_index <= (
                                        episodes_count / 2):
                                        for csvw in extra_csv_writers:
                                            csvw.writerow(expanded_iteration_row)
                                else:
                                    log.error("One iteration has a shape different to "
                                              "the first one used as reference for the header: "
                                              f"expected {len(expanded_header)}, found "
                                              f"{len(expanded_iteration_row)}")
                            episode_rewards.append(episode_total_reward)
                        avg_reward = np.mean(episode_rewards)
                        datasets_reward_mean[p] = avg_reward
                        avg_length = np.mean(episode_lengths)
                        datasets_episode_len_mean[p] = avg_length
                        datasets_iterations_count[p] = iterations_count
                        if extra_rewards:
                            for er in extra_rewards:
                                er.append(avg_reward)
                        if extra_episode_lengths:
                            for eel in extra_episode_lengths:
                                eel.append(avg_length)
                        if extra_iterations_counts:
                            for eic in extra_iterations_counts:
                                eic.append(iterations_count)
                    except Exception as ex:
                        log.error("Error parsing rollouts file!!!", exc_info=ex)
                else:
                    log.error("No rollouts data found!!!")
        reporter.report(f"Rollouts for Policy for level {policy_name} where save in:")
        reporter.report(f"- Shelve: {file_name}")
        reporter.report(f"- CSV: {csv_file_name}")
        p_finish_time = time.perf_counter()
        p_total_minutes = (p_finish_time - last_time) / 60
        last_time = p_finish_time
        datasets_times[p] = p_total_minutes
        reporter.report(f"Took: {p_total_minutes} minutes")

    if em_rewards:
        p = "em"
        avg_reward = np.mean(em_rewards)
        datasets_reward_mean[p] = avg_reward
        avg_length = np.mean(em_episode_lengths)
        datasets_episode_len_mean[p] = avg_length
        datasets_iterations_count[p] = sum(em_iterations_counts)
        datasets_episodes_count[p] = sum(em_episodes_counts)
    if mr_rewards:
        p = "mr"
        avg_reward = np.mean(mr_rewards)
        datasets_reward_mean[p] = avg_reward
        avg_length = np.mean(mr_episode_lengths)
        datasets_episode_len_mean[p] = avg_length
        datasets_iterations_count[p] = sum(mr_iterations_counts)
        datasets_episodes_count[p] = sum(mr_episodes_counts)

    if em_csv_file:
        em_csv_file.close()
    if mr_csv_file:
        mr_csv_file.close()
    if em_shelve:
        em_shelve.close()
    if mr_shelve:
        mr_shelve.close()

    reporter.report("Datasets generated!!!")
    reporter.report("Datasets reward stats:")
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
