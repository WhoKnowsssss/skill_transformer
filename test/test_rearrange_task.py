#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gc
import itertools
import json
import os
import os.path as osp
import time
from glob import glob

import pytest
import torch
import yaml

import habitat
import habitat.datasets.rearrange.run_episode_generator as rr_gen
import habitat.tasks.rearrange.rearrange_sim
import habitat.tasks.rearrange.rearrange_task
import habitat.utils.env_utils
from habitat.config.default import get_config
from habitat.core.embodied_task import Episode
from habitat.core.environments import get_env_class
from habitat.core.logging import logger
from habitat.datasets.rearrange.rearrange_dataset import RearrangeDatasetV0
from habitat.tasks.rearrange.multi_task.composite_task import CompositeTask
from habitat_baselines.config.default import get_config as baselines_get_config
from habitat_baselines.rl.ddppo.ddp_utils import find_free_port
from habitat_baselines.run import run_exp

CFG_TEST = "configs/tasks/rearrange/pick.yaml"
GEN_TEST_CFG = "habitat/datasets/rearrange/configs/test_config.yaml"
EPISODES_LIMIT = 6


def check_json_serialization(dataset: habitat.Dataset):
    start_time = time.time()
    json_str = dataset.to_json()
    logger.info(
        "JSON conversion finished. {} sec".format((time.time() - start_time))
    )
    decoded_dataset = RearrangeDatasetV0()
    decoded_dataset.from_json(json_str)
    decoded_dataset.config = dataset.config
    assert len(decoded_dataset.episodes) == len(dataset.episodes)
    episode = decoded_dataset.episodes[0]
    assert isinstance(episode, Episode)

    # The strings won't match exactly as dictionaries don't have an order for the keys
    # Thus we need to parse the json strings and compare the serialized forms
    assert json.loads(decoded_dataset.to_json()) == json.loads(
        json_str
    ), "JSON dataset encoding/decoding isn't consistent"


def test_rearrange_dataset():
    dataset_config = get_config(CFG_TEST).DATASET
    if not RearrangeDatasetV0.check_config_paths_exist(dataset_config):
        pytest.skip(
            "Please download ReplicaCAD RearrangeDataset Dataset to data folder."
        )

    dataset = habitat.make_dataset(
        id_dataset=dataset_config.TYPE, config=dataset_config
    )
    assert dataset
    dataset.episodes = dataset.episodes[0:EPISODES_LIMIT]
    check_json_serialization(dataset)


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("habitat_baselines/config/rearrange/**/*.yaml", recursive=True),
    ),
)
def test_rearrange_baseline_envs(test_cfg_path):
    """
    Test the Habitat Baseline environments
    """
    config = baselines_get_config(test_cfg_path)
    config.defrost()
    config.TASK_CONFIG.GYM.OBS_KEYS = None
    config.TASK_CONFIG.GYM.DESIRED_GOAL_KEYS = []
    config.freeze()

    env_class = get_env_class(config.TASK_CONFIG.ENV_TASK)

    env = habitat.utils.env_utils.make_env_fn(
        env_class=env_class, config=config
    )

    with env:
        for _ in range(10):
            env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                _, _, done, info = env.step(action=action)


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("configs/tasks/rearrange/*"),
    ),
)
def test_rearrange_tasks(test_cfg_path):
    """
    Test the underlying Habitat Tasks
    """
    if not osp.isfile(test_cfg_path):
        return

    config = get_config(test_cfg_path)

    with habitat.Env(config=config) as env:
        for _ in range(5):
            env.reset()


@pytest.mark.parametrize(
    "test_cfg_path",
    list(
        glob("configs/tasks/rearrange/*"),
    ),
)
def test_composite_tasks(test_cfg_path):
    """
    Test for the Habitat composite tasks.
    """
    if not osp.isfile(test_cfg_path):
        return

    config = get_config(test_cfg_path, ["SIMULATOR.CONCUR_RENDER", False])
    if "TASK_SPEC" not in config.TASK:
        return

    with habitat.Env(config=config) as env:
        if not isinstance(env.task, CompositeTask):
            return

        pddl_path = osp.join(
            config.TASK.TASK_SPEC_BASE_PATH, config.TASK.TASK_SPEC + ".yaml"
        )
        with open(pddl_path, "r") as f:
            domain = yaml.safe_load(f)
        if "solution" not in domain:
            return
        n_stages = len(domain["solution"])

        for task_idx in range(n_stages):
            env.reset()
            env.task.jump_to_node(task_idx, env.current_episode)
            env.step(env.action_space.sample())
            env.reset()


# NOTE: set 'debug_visualization' = True to produce videos showing receptacles and final simulation state
@pytest.mark.parametrize("debug_visualization", [False])
@pytest.mark.parametrize("num_episodes", [2])
@pytest.mark.parametrize("config", [GEN_TEST_CFG])
def test_rearrange_episode_generator(
    debug_visualization, num_episodes, config
):
    cfg = rr_gen.get_config_defaults()
    cfg.merge_from_file(config)
    dataset = RearrangeDatasetV0()
    with rr_gen.RearrangeEpisodeGenerator(
        cfg=cfg, debug_visualization=debug_visualization
    ) as ep_gen:
        start_time = time.time()
        dataset.episodes += ep_gen.generate_episodes(num_episodes)

    # test serialization of freshly generated dataset
    check_json_serialization(dataset)

    logger.info(
        f"successful_ep = {len(dataset.episodes)} generated in {time.time()-start_time} seconds."
    )


@pytest.mark.parametrize(
    "test_cfg_path,mode",
    list(
        itertools.product(
            glob("habitat_baselines/config/tp_srl_test/*"),
            ["eval"],
        )
    ),
)
def test_tp_srl(test_cfg_path, mode):
    # For testing with world_size=1
    os.environ["MAIN_PORT"] = str(find_free_port())

    run_exp(
        test_cfg_path,
        mode,
        ["EVAL.SPLIT", "train"],
    )

    # Needed to destroy the trainer
    gc.collect()

    # Deinit processes group
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
