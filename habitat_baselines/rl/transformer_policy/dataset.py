import os, pickle, time
from threading import Thread
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch

from torch.utils.data import (
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    DistributedSampler,
    get_worker_info,
)

from habitat import Config, logger
from habitat_baselines.transformer_policy.dataset_utils import producer
from habitat_baselines.common.tensor_dict import TensorDict


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = actions.shape[-1]
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        assert (
            len(self.actions) > self.block_size
        ), "No enough transitions in this dataset"
        return len(self.actions) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size
        done_idx = min(
            self.done_idxs[np.searchsorted(self.done_idxs, idx)],
            idx + block_size,
        )
        idx = done_idx - block_size

        if idx < 0:
            idx, done_idx = 0, block_size
            print(
                f"ERROR on indexing, idx: {idx}, done_idx: {done_idx}, {self.done_idxs}"
            )
        states = self.data[idx:done_idx]
        assert len(states) == self.block_size, "Error on states length"

        actions = torch.tensor(
            self.actions[idx:done_idx], dtype=torch.float32
        ).unsqueeze(
            1
        )  # (block_size, 1)
        rtgs = torch.tensor(
            self.rtgs[idx:done_idx], dtype=torch.float32
        ).unsqueeze(1)
        timesteps = torch.tensor(
            self.timesteps[idx : idx + 1], dtype=torch.int64
        ).unsqueeze(1)

        return states, actions, rtgs, timesteps

    def __getindex__(self, idx):
        block_size = self.block_size
        try:
            done_idx = min(
                self.done_idxs[np.searchsorted(self.done_idxs, idx)],
                idx + block_size,
            )
        except:
            done_idx = min(
                self.done_idxs[np.searchsorted(self.done_idxs, idx) - 1],
                idx + block_size,
            )
        return torch.arange(done_idx - block_size, done_idx)

    @classmethod
    def from_config(
        cls,
        buffer: Tuple,
        context_length: int = 30,
    ):
        obss, actions, done_idxs, rtg, timesteps = buffer
        return cls(obss, context_length, actions, done_idxs, rtg, timesteps)


class RollingDataset(IterableDataset):
    class DatasetIterator:
        dataset: StateActionReturnDataset

        def __init__(
            self,
            config: Config,
            context_length: int,
            sampler_params: Tuple,
            dataset_context: Dict,
            world_rank: bool,
        ):
            self.config = config
            self.context_length = context_length
            self.dataset_context = dataset_context
            self.world_rank = world_rank
            num_replicas, rank, self.seed = sampler_params
            assert (num_replicas is None) == (
                rank is None
            ), "Local or Distributed Training? "
            if num_replicas is None:
                self._is_distributed = False
            else:
                self.num_replicas = num_replicas
                self.rank = rank
                self._is_distributed = True

            self.dataset_context["num_iterated"] = 0
            self.dataset_context["num_init"] = 0
            self.num_iterated_epoch = 0
            self.queue = deque()
            self.producer = None
            self.batch_idx = []

        def init_dataset(self):
            assert hasattr(
                self, "seed_epoch"
            ), "Set epoch before Dataloader loads"

            while len(self.queue) == 0:
                time.sleep(1)
            self.dataset = StateActionReturnDataset.from_config(
                self.queue.popleft(), self.context_length
            )
            self.dataset_context["num_init"] += 1

            self.sampler = RandomSampler(self.dataset)  # RandomSampler

        def __iter__(self):
            self.num_iterated_epoch = 0
            self.batch_idx = []

            worker_info = get_worker_info()
            self.num_workers = (
                worker_info.num_workers - 1 if worker_info is not None else 0
            )
            self.id = worker_info.id if worker_info is not None else 0

            rng = np.random.default_rng(self.seed + self.id)
            if self.producer is None:
                self.producer = Thread(
                    target=producer,
                    args=(
                        self.config,
                        rng,
                        self.queue,
                        False,
                        self.context_length,
                    ),
                )  # config, np.RNG, queue, verbose
                self.producer.start()

            self.init_dataset()
            self.sampler_iterator = iter(self.sampler)

            return self

        def __next__(self):
            self.num_iterated_epoch += 1

            try:
                idx = next(self.sampler_iterator)
            except StopIteration:
                raise StopIteration

            idx_list = self.dataset.__getindex__(idx)
            self.batch_idx.append(idx_list)
            return None

        def set_epoch(self, epoch):
            if self._is_distributed:
                try:
                    self.sampler.set_epoch(epoch)
                except:
                    pass
            self.seed_epoch = epoch

    def __init__(
        self,
        config: Config,
        context_length: int,
        sampler_params: Tuple,
        dataset_context: dict,
        world_rank: bool,
    ):
        self.iterator = self.DatasetIterator(
            config, context_length, sampler_params, dataset_context, world_rank
        )

    def __iter__(self):
        return iter(self.iterator)

    def set_epoch(self, epoch):
        self.iterator.set_epoch(epoch)

    def get_batch(self, data):
        B = len(self.iterator.batch_idx)
        batch_idx = torch.cat(self.iterator.batch_idx)
        self.iterator.batch_idx = []
        states = self.iterator.dataset.data[batch_idx]
        actions = self.iterator.dataset.actions[batch_idx]  # (block_size, 1)
        rtgs = self.iterator.dataset.rtgs[batch_idx]
        timesteps = self.iterator.dataset.timesteps[
            batch_idx[:: self.iterator.context_length]
        ]

        return (
            states,
            actions.view(B, -1, actions.shape[-1]),
            rtgs.view(B, -1, 1),
            timesteps.view(-1, 1, 1),
        )
