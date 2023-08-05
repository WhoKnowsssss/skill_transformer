#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from builtins import breakpoint
import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional

from matplotlib import pyplot as plt
import wandb


import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm
from gym import spaces
from torch import device, nn
from torch.optim.lr_scheduler import LambdaLR
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp

from habitat import Config, VectorEnv, logger
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.environments import get_env_class
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from habitat_baselines.transformer_policy.dataset import RollingDataset
from habitat_baselines.utils.common import (
    cosine_decay,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)
from habitat_baselines.common.construct_vector_env import construct_envs
from habitat.utils.render_wrapper import overlay_frame


@baseline_registry.register_trainer(name="transformer")
class TransformerTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    envs: VectorEnv

    def __init__(self, config=None):
        super().__init__(config)
        self.transformer_policy = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1

        self.using_velocity_ctrl = (
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ) == ["VELOCITY_CONTROL"]

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_transformer_policy(self) -> None:
        r"""Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.transformer_policy = policy.from_config(
            self.config,
            observation_space,
            self.policy_action_space,
        )
        self.obs_space = observation_space
        self.transformer_policy.to(self.device)

        if (
            self.config.RL.TRANSFORMER.pretrained_encoder
            or self.config.RL.TRANSFORMER.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.TRANSFORMER.pretrained_weights,
                map_location="cpu",
            )

        if self.config.RL.TRANSFORMER.pretrained:
            prefix = (
                "module."
                if "module.net.state_encoder.skill_embedding"
                in pretrained_state["state_dict"].keys()
                else ""
            )
            self.transformer_policy.load_state_dict(
                {
                    k[k.find(prefix) + len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.TRANSFORMER.pretrained_encoder:
            prefix = "net.visual_encoder."
            self.transformer_policy.net.visual_encoder.load_state_dict(
                {
                    k[k.find(prefix) + len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if prefix in k
                }
            )

        if not self.config.RL.TRANSFORMER.train_encoder:
            self._static_encoder = True
            for (
                param
            ) in self.transformer_policy.net.visual_encoder.parameters():
                param.requires_grad_(False)

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        if is_slurm_batch_job():
            add_signal_handlers()

        world_rank, world_size = None, None
        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.TRANSFORMER.distrib_backend
            )
            world_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if rank0_only():
                logger.info(
                    "Initialized Skill Transformer with {} workers, id={}".format(
                        world_size, world_rank
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                world_rank * self.config.RL.TRANSFORMER.num_workers
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._init_envs(self.config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
        else:
            self.policy_action_space = action_space

        self._setup_transformer_policy()

        logger.info(
            "agent number of parameters: {}".format(
                sum(
                    param.numel()
                    for param in self.transformer_policy.parameters()
                )
            )
        )

        if self._is_distributed:
            self.init_distributed(find_unused_params=True)
            torch.distributed.barrier()

        manager = mp.Manager()
        self.dataset_context = manager.dict()
        self.train_dataset = RollingDataset(
            self.config.RL.TRAJECTORY_DATASET,
            self.config.RL.TRANSFORMER.context_length,
            (world_size, world_rank, self.config.TASK_CONFIG.SEED),
            self.dataset_context,
            rank0_only(),
        )

        self.test_dataset = RollingDataset(
            self.config.RL.VALIDATION_DATASET,
            self.config.RL.TRANSFORMER.context_length,
            (world_size, world_rank, self.config.TASK_CONFIG.SEED),
            self.dataset_context,
            rank0_only(),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            pin_memory=True,
            batch_size=self.config.RL.TRANSFORMER.batch_size,
            num_workers=self.config.RL.TRANSFORMER.num_workers,
            collate_fn=self.train_dataset.get_batch,
            persistent_workers=True,
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            pin_memory=True,
            batch_size=self.config.RL.TRANSFORMER.batch_size,
            num_workers=1,
            collate_fn=self.test_dataset.get_batch,
            persistent_workers=True,
        )

        self.optimizer = torch.optim.AdamW(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.transformer_policy.parameters(),
                )
            ),
            lr=self.config.RL.TRANSFORMER.lr,
            eps=self.config.RL.TRANSFORMER.eps,
        )

        self.envs.close()

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def init_distributed(self, find_unused_params: bool = True) -> None:

        if torch.cuda.is_available():
            self.transformer_policy = (
                torch.nn.parallel.DistributedDataParallel(
                    self.transformer_policy,
                    device_ids=[self.device],
                    output_device=self.device,
                    find_unused_parameters=find_unused_params,
                )
            )
        else:
            self.transformer_policy = (
                torch.nn.parallel.DistributedDataParallel(
                    self.transformer_policy,
                    find_unused_parameters=find_unused_params,
                )
            )
        self.transformer_policy = (
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.transformer_policy
            ).to(self.device)
        )

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.transformer_policy.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

        if len(os.listdir(self.config.CHECKPOINT_FOLDER)) > 17:
            fn = os.listdir(self.config.CHECKPOINT_FOLDER)
            fn = [
                int(f.split(".")[1])
                if f != ".habitat-resume-state.pth"
                else -1
                for f in fn
            ]
            fn.sort()
            os.remove(
                os.path.join(
                    self.config.CHECKPOINT_FOLDER,
                    "ckpt.{}.pth".format(fn[1]),
                )
            )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        writer.add_scalars(
            "losses",
            losses,
            self.num_updates_done,
        )
        writer.add_scalar(
            f"learning_rate",
            self.optimizer.param_groups[0]["lr"],
            self.num_updates_done,
        )

    @rank0_only
    def _testing_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        writer.add_scalars(
            "test/losses",
            losses,
            self.num_updates_done,
        )

    def _run_epoch(self, split: str, epoch_num: int = 0):
        is_train = not (
            (self.num_updates_done % self.config.TEST_INTERVAL == 0)
            and not self.evaluated
        )
        is_train = True
        self.transformer_policy.train(True) #is_train

        if is_train:
            pbar = tqdm(enumerate(self.train_loader))
            self.evaluated = False
        else:
            pbar = tqdm(enumerate(self.test_loader))
            self.evaluated = True
        losses = []
        for it, (x, y, r, t) in pbar:
            # place data on the correct device

            x = TensorDict.from_tree(x).map_in_place(
                lambda v: v.to(self.device)
            )
            y = y.to(self.device)
            r = r.to(self.device)
            t = t.to(self.device)

            # forward the model
            with torch.set_grad_enabled(True):
                loss, loss_dict = self.transformer_policy(x, y, y, r, t)
                loss = (
                    loss.mean()
                )  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss_dict)

            # backprop and update the parameters
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.transformer_policy.parameters(),
                self.config.RL.TRANSFORMER.grad_norm_clip,
            )
            if is_train:
                self.optimizer.step()
                # report progress
                # if rank0_only():
                pbar.set_description(
                    f"Epoch {epoch_num+1} iter {it}: Train loss {loss.detach().item():.5f}."
                )
            else:
                # if rank0_only():
                pbar.set_description(
                    f"TEST epoch {epoch_num} iter {it}: TEST loss {loss.detach().item():.5f}."
                )

        losses = {
            k: torch.from_numpy(
                np.array([np.mean([d[k] for d in losses])])
            ).to(self.device)
            for k in loss_dict.keys()
        }
        print("LR:::", self.optimizer.param_groups[0]["lr"])
        return losses, is_train

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.
        Returns:
            None
        """

        self._init_train()
        self.evaluated = False

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler_after = LambdaLR(
            optimizer=self.optimizer,
            lr_lambda=lambda x: cosine_decay(self.percent_done()),
        )
        lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1,
            total_epoch=self.config.RL.TRANSFORMER.warmup_updates,  # 100
            after_scheduler=lr_scheduler_after,
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            if self._is_distributed:
                prefix = (
                    "module."
                    if "module.net.state_encoder.skill_embedding"
                    not in resume_state["state_dict"].keys()
                    else ""
                )
                self.transformer_policy.load_state_dict(
                    {
                        prefix + k: v
                        for k, v in resume_state["state_dict"].items()
                    }
                )
            else:
                prefix = (
                    "module."
                    if "module.net.state_encoder.skill_embedding"
                    in resume_state["state_dict"].keys()
                    else ""
                )
                self.transformer_policy.load_state_dict(
                    {
                        k[k.find(prefix) + len(prefix) :]: v
                        for k, v in resume_state["state_dict"].items()
                    }
                )
            self.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler_after.load_state_dict(resume_state["lr_sched_state"])
            lr_scheduler.total_epoch = 0

            requeue_stats = resume_state["requeue_stats"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            count_checkpoints = requeue_stats["count_checkpoints"]

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress(AttributeError)
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_updates_done=self.num_updates_done,
                    )
                    save_resume_state(
                        dict(
                            state_dict=self.transformer_policy.state_dict(),
                            optim_state=self.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler_after.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():

                    requeue_job()

                    return

                self.train_dataset.set_epoch(self.num_updates_done)
                self.test_dataset.set_epoch(0)

                with self.transformer_policy.join():
                    loss, is_train = self._run_epoch(
                        "train", epoch_num=self.num_updates_done
                    )
                for k in loss:
                    torch.distributed.all_reduce(loss[k])
                    loss[k] = loss[k].cpu().item() / 4
                if is_train:
                    self._training_log(writer, loss, prev_time)
                    self.num_updates_done += 1
                    if self.config.RL.TRANSFORMER.use_linear_lr_decay:
                        lr_scheduler.step()  # type: ignore
                else:
                    self._testing_log(writer, loss, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.
        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging
        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if (
            len(config.VIDEO_RENDER_VIEWS) > 0
            and len(self.config.VIDEO_OPTION) > 0
        ):
            config.defrost()
            for render_view in config.VIDEO_RENDER_VIEWS:
                uuid = config.TASK_CONFIG.SIMULATOR[render_view].UUID
                config.TASK_CONFIG.GYM.OBS_KEYS.append(uuid)
                config.SENSORS.append(render_view)
            config.TASK_CONFIG.SIMULATOR.DEBUG_RENDER = True
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)
        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_policy_action_space = self.envs.orig_action_spaces[0]
        if is_continuous_action_space(action_space):
            # Assume NONE of the actions are discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = (1,)
            discrete_actions = True

        self._setup_transformer_policy()

        prefix = (
            "module."
            if "module.net.state_encoder.skill_embedding"
            in ckpt_dict["state_dict"].keys()
            else ""
        )
        self.transformer_policy.load_state_dict(
            {
                k[k.find(prefix) + len(prefix) :]: v
                for k, v in ckpt_dict["state_dict"].items()
            }
        )
        self.actor_critic = self.transformer_policy

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.num_recurrent_layers,
            self.actor_critic.hidden_state_hxs_dim,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        evals_per_ep = self.config.EVAL.EVALS_PER_EP
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with TEST_EPISODE_COUNT"

        pbar = tqdm(total=number_of_eval_episodes * evals_per_ep)
        self.actor_critic.eval()

        first_nav_success_counter_temp = {}
        second_nav_success_counter_temp = {}
        envs_to_pause_list = []
        skill_data = np.ones((self.envs.num_envs, 5000, 2)) * -1
        skill_accuracy = []
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            if self.config.RL.TRANSFORMER.model_type == "reward_conditioned":
                rtgs = self.config.RL.TRANSFORMER.return_to_go - current_episode_reward
            else:
                rtgs = None
            with torch.inference_mode():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                    envs_to_pause=envs_to_pause_list,
                    rtgs=rtgs,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if (
                current_episodes_info[0].episode_id
                not in first_nav_success_counter_temp.keys()
            ):
                first_nav_success_counter_temp[
                    current_episodes_info[0].episode_id
                ] = False
                second_nav_success_counter_temp[
                    current_episodes_info[0].episode_id
                ] = False
            if batch["obj_start_gps_compass"][0, 0] < 1.5:
                first_nav_success_counter_temp[
                    current_episodes_info[0].episode_id
                ] = True
            if (
                batch["obj_goal_gps_compass"][0, 0] < 1.5
                and batch["is_holding"][0] == 1
            ):
                second_nav_success_counter_temp[
                    current_episodes_info[0].episode_id
                ] = True

            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions.cpu()
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            outputs = self.envs.step(step_data)
            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            policy_info = self.actor_critic.get_policy_info(infos, dones)
            oracle_skill = self.get_oracle_skill(batch)
            timestep_mask = np.argmax(np.all(skill_data == -1, -1), -1)
            for i in range(len(policy_info)):
                infos[i].update(policy_info[i])
                skill_data[i,timestep_mask[i],0] = policy_info[i]["cur_skill"]
                skill_data[i,timestep_mask[i],1] = oracle_skill[i]
                infos[i].update(
                    {
                    "skill_data": "{}".format(policy_info[i]["cur_skill"]),
                    "oracle_skill": "{}".format(oracle_skill[i]),
                    }
                )
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs

            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                    envs_to_pause_list.append(i)

                if len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])
                    rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats
                    mask = skill_data[i,:,0] != -1
                    skill_data[i, :, :] = -1

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id
                            + f"_{len(stats_episodes)}",
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                    num_episodes = len(stats_episodes)
                    aggregated_stats = {}
                    for stat_key in ["composite_stage_goals.stage_0_5_success", "composite_stage_goals.stage_1_success"]:
                        aggregated_stats[stat_key] = (
                            sum(v[stat_key] for v in stats_episodes.values())
                            / num_episodes
                        )

                    if num_episodes % 10 == 0:
                        for k, v in aggregated_stats.items():
                            print(f"Average episode {k}: {v:.4f}")
                        for k, v in metrics.items():
                            print(f"Average episode {k}: {v:.4f}")

                    metrics = {
                        k: v
                        for k, v in aggregated_stats.items()
                        if k != "reward"
                    }
                    for k, v in metrics.items():
                        writer.add_scalar(f"eval_metrics/{k}", v, num_episodes)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        self.envs.close()


    def get_oracle_skill(self, observations):
        mask_hold = observations['all_predicates'][:,10].bool()
        mask_goal = (observations["obj_goal_gps_compass"][:,0] < 1.5)
        mask_start = (observations["obj_start_gps_compass"][:,0] < 1.5)
        mask_in = torch.any(observations['all_predicates'][:,:5], dim=-1)
        mask_pick = torch.any(observations['receptacle_state'][:,:5], dim=-1)
        # =========== rules ===========
        skills = torch.zeros(len(mask_hold))
        skills[mask_hold & mask_goal] = 2
        skills[mask_hold & ~mask_goal] = 4
        skills[~mask_hold & ~mask_start] = 0
        skills[~mask_hold & mask_start & ~mask_in] = 1
        skills[~mask_hold & mask_start & mask_in & mask_pick] = 1
        skills[~mask_hold & mask_start & mask_in & ~mask_pick] = 5

        return skills

    # where we defined the skills
    @property
    def skill_dict(self):
        return {
            "nav": 0,
            "nav_goal": 4,
            "pick": 1,
            "pick_offset": 3,
            "place": 2,
            "open_cab": 5,
            "open_fridge": 6,
            "reset_arm": -1,
            "wait": 7,
            "wait": 8,
            "wait": 9,
        }
    
    def get_skill_accuracy(self, planner):
        skill_dict = {v: k for k,v in self.skill_dict.items()}

        accuracy = []
        for i in range(len(planner)):
            if planner[i,0] == 6:
                planner[i,0] = 5
            if planner[i,0] == 3:
                planner[i,0] = 1
            accuracy.append(float(skill_dict[planner[i,0]] == skill_dict[planner[i,1]]))

        return torch.mean(torch.tensor(accuracy))