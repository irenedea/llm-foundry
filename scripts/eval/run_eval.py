# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import os
import subprocess

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
REGRESSIONS_DIR = os.path.join(DIR_PATH, 'yamls')

from mcli import RunConfig, create_run


def get_configs(cluster: str, wandb_entity: str,
                wandb_project: str):
    hf_8bit_eval = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'hf_8bit_eval.yaml'))
    hf_lora_eval = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'hf_lora_eval.yml'))
    mpt_eval = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'mpt_eval.yaml'))
    hf_eval = RunConfig.from_file(
        os.path.join(REGRESSIONS_DIR, 'hf_eval.yaml'))

    all_configs = [
        hf_eval, hf_8bit_eval, hf_lora_eval, mpt_eval
    ]

    commit_hash = subprocess.check_output(['git', 'rev-parse',
                                           'HEAD']).strip().decode('utf-8')
    timestamp = datetime.datetime.now().strftime('%m-%d-%Y::%H:%M:%S')
    wandb_group = f'{timestamp}::{commit_hash}'

    # make general changes
    wandb_config = {
        'entity': wandb_entity,
        'project': wandb_project,
        'group': wandb_group
    }
    for config in all_configs:
        config.cluster = cluster
        config.parameters['loggers'] = config.parameters.get('loggers', {})
        config.parameters['loggers']['wandb'] = wandb_config

    return all_configs, []


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--cluster', type=str)
    # parser.add_argument('--wandb-entity', type=str)
    # parser.add_argument('--wandb-project', type=str)

    args = parser.parse_args()

    run_configs, _ = get_configs(cluster='r1z1', wandb_entity='mosaic-ml', wandb_project='irene-test')
    for run_config in run_configs:
        run = create_run(run_config)
