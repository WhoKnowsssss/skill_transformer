#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type, Union

from habitat import Config, Env, RLEnv, make_dataset


def make_env_fn(
    config: Config, env_class: Union[Type[Env], Type[RLEnv]]
) -> Union[Env, RLEnv]:
    r"""Creates an env of type env_class with specified config and rank.
    This is to be passed in as an argument when creating VectorEnv.

    Args:
        config: root exp config that has core env config node as well as
            env-specific config node.
        env_class: class type of the env to be created.

    Returns:
        env object created according to specification.
    """
    if "TASK_CONFIG" in config:
        config = config.TASK_CONFIG
    dataset = make_dataset(config.DATASET.TYPE, config=config.DATASET)
    env = env_class(config=config, dataset=dataset)
    env.seed(config.SEED)
    return env
