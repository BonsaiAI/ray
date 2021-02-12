#!/usr/bin/env python

import argparse

from ray.rllib import train
from ray.rllib import rollout
from ray.rllib import dataset

EXAMPLE_USAGE = """
Example usage for training:
    rllib train --run DQN --env CartPole-v0

Example usage for rollout:
    rllib rollout /trial_dir/checkpoint_1/checkpoint-1 --run DQN

Example usage for dataset:
    rllib dataset 250 -f tuned_examples/moab-ppo.yaml --steps 1000000
"""


def cli():
    parser = argparse.ArgumentParser(
        description="Train or Run an RLlib Trainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE)
    subcommand_group = parser.add_subparsers(
        help="Commands to train or run an RLlib agent.", dest="command")

    # see _SubParsersAction.add_parser in
    # https://github.com/python/cpython/blob/master/Lib/argparse.py
    train_parser = train.create_parser(
        lambda **kwargs: subcommand_group.add_parser("train", **kwargs))
    rollout_parser = rollout.create_parser(
        lambda **kwargs: subcommand_group.add_parser("rollout", **kwargs))
    dataset_parser = dataset.create_parser(
        lambda **kwargs: subcommand_group.add_parser("dataset", **kwargs))
    options = parser.parse_args()

    if options.command == "train":
        train.run(options, train_parser)
    elif options.command == "rollout":
        rollout.run(options, rollout_parser)
    elif options.command == "dataset":
        dataset.run(options, dataset_parser)
    else:
        parser.print_help()
