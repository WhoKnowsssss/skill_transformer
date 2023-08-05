import torch
import torch.nn as nn
import math
import numpy as np
from habitat_baselines.utils.common import (
    iterate_action_space_recursively,
    CustomNormal,
    CustomFixedCategorical,
    ActionDistributionNet,
    get_num_discrete_action_logits,
    get_num_continuous_action_logits,
    get_num_actions,
)
from habitat_baselines.utils.common import (
    CategoricalNet,
    GaussianNet,
    get_num_actions,
)
import gym.spaces as spaces


class _SumTensors(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors):
        ctx.num_inputs = len(tensors)

        return torch.stack(tensors, -1).sum(-1)

    @staticmethod
    def backward(ctx, grad_out):
        return tuple(grad_out for _ in range(ctx.num_inputs))


def sum_tensor_list(tensors):
    if len(tensors) == 1:
        return tensors[0]
    elif len(tensors) == 2:
        return tensors[0] + tensors[1]
    else:
        return _SumTensors.apply(*tensors)


class ActionDistribution:
    def __init__(
        self,
        action_space,
        box_mu_act,
        logits,
        std,
        boundaries,
        boundaries_mean,
    ):
        if std is None:
            self.params = logits
        else:
            self.params = torch.cat((logits, std), -1)

        self.distributions = []
        self.action_slices = []
        self.action_dtypes = []
        logits_offset = 0
        std_offset = 0
        action_offset = 0
        self.dtype = torch.int64
        self.logits = logits
        self.boundaries_mean = boundaries_mean
        self.boundaries = boundaries
        for space in iterate_action_space_recursively(action_space):
            if isinstance(space, spaces.Box):
                numel = int(np.prod(space.shape))
                mu = logits[..., logits_offset : logits_offset + numel]
                if box_mu_act == "tanh":
                    mu = torch.tanh(mu)
                self.distributions.append(
                    CustomNormal(mu, std[..., std_offset : std_offset + numel])
                )
                std_offset += numel

                self.action_slices.append(
                    slice(action_offset, action_offset + numel)
                )
                self.dtype = torch.float32
                self.action_dtypes.append(torch.float32)
            elif isinstance(space, spaces.Discrete):
                numel = space.n
                self.distributions.append(
                    CustomFixedCategorical(
                        logits=logits[
                            ..., logits_offset : logits_offset + numel
                        ]
                    )
                )
                self.action_slices.append(
                    slice(action_offset, action_offset + 1)
                )
                self.action_dtypes.append(torch.int64)

            logits_offset += numel
            action_offset = self.action_slices[-1].stop

    def sample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        action = torch.cat(
            [
                dist.sample(sample_shape).to(self.dtype)
                for dist in self.distributions
            ],
            -1,
        )
        return action

    def mean(self):
        return torch.cat(
            [
                dist.mode().to(self.dtype)
                if isinstance(dist, CustomFixedCategorical)
                else dist.mean
                for dist in self.distributions
            ],
            -1,
        )

    def log_probs(self, action):
        all_log_probs = []
        for dist, _slice, dtype in zip(
            self.distributions, self.action_slices, self.action_dtypes
        ):
            all_log_probs.append(dist.log_probs(action[..., _slice].to(dtype)))

        return sum_tensor_list(all_log_probs)

    def entropy(self):
        return sum_tensor_list([dist.entropy() for dist in self.distributions])

    def unnormalize_actions(self, action):
        action = torch.clone(action)
        action[:, :7] = self.boundaries_mean[action[:, :7].to(torch.long)]
        action[:, 7] = (
            (action[:, 7] == 1).int()
            + 2 * (action[:, 7] == 0).int()
            + 3 * (action[:, 7] == 2).int()
            - 2
        )
        mask = action[:, 7:8] == -1
        return torch.cat([action, torch.zeros_like(mask.float())], dim=-1)

    def normalize_actions(self, action):
        action = torch.clone(action[:, :10])
        action[:, :7] = torch.bucketize(action[:, :7], self.boundaries) - 1
        action[:, 7] = (
            (action[:, 7] == 0).int()
            + 2 * (action[:, 7] == -1).int()
            + 3 * (action[:, 7] == 1).int()
            - 1
        )
        return action


class MixedDistributionNet(ActionDistributionNet):
    def __init__(
        self,
        num_inputs: int,
        config,
        action_space,
    ) -> None:
        super().__init__()

        self.action_activation = config.action_activation
        self.use_softplus = config.use_softplus
        use_std_param = config.use_std_param
        self.clamp_std = config.clamp_std
        self.min_std = config.min_log_std
        self.max_std = config.max_log_std
        std_init = config.log_std_init
        self.temperature = config.temperature
        self.scheduled_std = False

        n_actions = get_num_actions(action_space)
        ac_spaces = {
            "arm": spaces.Dict({k: spaces.Discrete(21 ) for k in range(7)})
            if config.discrete_arm
            else spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32),
            "gripper": spaces.Discrete(3),
            "locomotion": spaces.Dict(
                {k: spaces.Discrete(21 ) for k in range(2)}
            )
            if config.discrete_base
            else spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
        }

        self.boundaries_mean = torch.linspace(-1, 1, 21 ).cuda()
        self.boundaries = torch.linspace(-1.025, 1.025, 22).cuda()

        self.action_space = spaces.Dict(ac_spaces)

        self.num_discrete_logits = get_num_discrete_action_logits(
            self.action_space
        )
        self.num_continuous_logits = get_num_continuous_action_logits(
            self.action_space
        )

        if use_std_param:
            self.std = torch.nn.parameter.Parameter(
                torch.randn(self.num_continuous_logits) * 0.01 + std_init
            )
        elif self.scheduled_std:
            if self.use_log_std:
                self.min_std = math.exp(self.min_std)
                self.max_std = math.exp(self.max_std)
                std_init = math.exp(std_init)
                self.use_log_std = False

            self.std_init = std_init
            self.register_buffer("std", torch.full((), self.std_init))
            self.update(0.0)
        else:
            self.std = None
            self.std_head = nn.Linear(num_inputs, self.num_continuous_logits)

        num_outputs = self.num_continuous_logits + self.num_discrete_logits

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

        if not use_std_param:
            nn.init.constant_(self.std_head.bias[:], std_init)

        ###HACK

        if False:
            for param in self.parameters():
                param.requires_grad = False
            self.residual_action_net = GaussianNet(
                num_inputs,
                n_actions,
                config,
            )

    def update(self, training_progress: float):
        if not self.scheduled_std:
            return

        init_var = math.pow(self.std_init, 2)
        final_var = init_var * 0.1
        new_var = (init_var - final_var) * (
            1.0 - min(max(training_progress / (2 / 3), 0.0), 1.0)
        ) + final_var

        self.std.fill_(math.sqrt(new_var))

    def forward(self, x, return_logits=False) -> ActionDistribution:

        logits = self.linear(x)

        if return_logits:
            return logits

        if self.std is not None:
            std = (self.std).repeat(x.shape[0], 1)
        else:
            std = self.std_head(x)

        if self.clamp_std:
            std = torch.clamp(std, min=self.min_std, max=self.max_std)
        std = torch.exp(std)
        if self.use_softplus:
            std = torch.nn.functional.softplus(std)

        logits[:, :self.num_discrete_logits] = logits[:, :self.num_discrete_logits] / self.temperature

        if not hasattr(self, "residual_action_net"):
            return ActionDistribution(
                self.action_space,
                self.action_activation,
                logits,
                std,
                self.boundaries,
                self.boundaries_mean,
            )
        else:
            return Residual_Action_Distribution(
                ActionDistribution(
                    self.action_space,
                    self.action_activation,
                    logits,
                    std,
                    self.boundaries,
                    self.boundaries_mean,
                ),
                self.residual_action_net(x),
            )


class Residual_Action_Distribution:
    def __init__(self, main, residual) -> None:
        self.main = main
        self.residual = residual

    def sample(self, sample_shape=None):
        return self.main.mean() + self.residual.sample()

    def mean(self):
        return self.main.mean() + self.residual.mean

    def log_probs(self, actions):
        return self.residual.log_probs(actions - self.main.mean())

    def entropy(self):
        return self.residual.entropy()


def linear_decay(epoch: int, total_num_updates: int) -> float:
    r"""Returns a multiplicative factor for linear value decay

    Args:
        epoch: current epoch number
        total_num_updates: total number of

    Returns:
        multiplicative factor that decreases param value linearly
    """
    return 1 - (epoch / float(total_num_updates))
