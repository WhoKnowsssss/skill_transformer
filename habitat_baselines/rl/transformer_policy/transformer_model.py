import math
import logging
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class ActionNorm(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        assert mean.shape[0] == std.shape[0], "Shape Must Match"
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def unnormalize(self, y):
        return y * self.std + self.mean


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        if config.reg_flags["attention_dropout"]:
            self.attn_drop = nn.Dropout(config.attn_pdrop)
            self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        # if attention_mask is not None:
        #     att = att.masked_fill(attention_mask.repeat(T,self.n_head,1,1).transpose(0,2) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        if hasattr(self, "attn_drop"):
            att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        if hasattr(self, "resid_drop"):
            y = self.resid_drop(self.proj(y))
        else:
            y = self.proj(y)
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        if config.reg_flags["attention_layernorm"]:
            self.ln1 = nn.LayerNorm(config.n_embd)
        if config.reg_flags["feedforward_layernorm"]:
            self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        if config.reg_flags["feedforward_dropout"]:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Dropout(config.resid_pdrop),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
            )

    def forward(self, x):
        if hasattr(self, "ln1"):
            x = x + self.attn(self.ln1(x))
        else:
            x = x + self.attn(x)
        if hasattr(self, "ln2"):
            x = x + self.mlp(self.ln2(x))
        else:
            x = x + self.mlp(x)

        return x


class ActionInference(nn.Module):
    """the action inference module that outputs encoded embedding used to infer low-level actions"""

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 3

        config.block_size = config.block_size * self.num_inputs

        self.reg_flags = config.reg_flags

        self.block_size = config.block_size

        self.n_embd = config.n_embd
        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        if config.num_skills != 0:
            self.skill_embedding = nn.Parameter(
                torch.randn(config.num_skills, config.n_embd)
            )
        if self.reg_flags["outer_dropout"]:
            self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        # decoder head
        if self.reg_flags["outer_layernorm"]:
            self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        if config.num_states[0] == 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.num_states[1], config.n_embd), nn.Tanh()
            )
        else:
            self.state_encoder = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(i, config.n_embd // 2), nn.Tanh())
                    for i in [config.num_states[1]]
                ]
            )

        if self.model_type == "reward_conditioned":
            self.ret_emb = nn.Sequential(
                nn.Linear(1, config.n_embd), nn.Tanh()
            )

        self.action_embeddings = nn.Sequential(
            nn.Linear(config.vocab_size, config.n_embd), nn.Tanh()
        )
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

        self.boundaries_mean = torch.linspace(-1, 1, 21).cuda()
        self.boundaries = torch.linspace(-1.025, 1.025, 22).cuda()

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # state, action, and return
    def forward(self, states, actions, rtgs=None):
        # states: (batch, block_size, 4*84*84)
        # actions: (batch, block_size, 8)
        # rtgs: (batch, block_size, 1)

        if (
            states.shape[-1]
            == self.n_embd // 2 + self.config.num_states[1] + 1
        ):
            states, skill_set = torch.split(
                states, [self.n_embd // 2 + self.config.num_states[1], 1], -1
            )
        else:
            skill_set = None

        state_inputs = list(
            torch.split(
                states, [self.n_embd // 2, self.config.num_states[1]], -1
            )
        )

        for i in range(1, len(state_inputs)):
            state_inputs[i] = self.state_encoder[i - 1](
                state_inputs[i].type(torch.float32)
            )

        if actions is not None and self.model_type == "reward_conditioned":
            rtg_embeddings = self.ret_emb(rtgs.type(torch.float32))
            actions = torch.clone(actions)
            actions[:, :, [10]] = 0
            actions = actions.type(torch.float32)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)

            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    self.num_inputs * states.shape[1],
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=action_embeddings.device,
            )

            token_embeddings[:, :: self.num_inputs, :] = rtg_embeddings

            token_embeddings[:, 1 :: self.num_inputs, :] = torch.cat(
                [state_inputs[0], state_inputs[-1]], dim=-1
            )

            token_embeddings[
                :, (self.num_inputs - 1) :: self.num_inputs, :
            ] = action_embeddings

        elif actions is not None and self.model_type == "bc":
            actions = torch.clone(actions)
            actions[:, :, [10]] = 0
            actions = actions.type(torch.float32)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)
            if skill_set is not None:
                token_embeddings = torch.zeros(
                    (
                        states.shape[0],
                        (self.num_inputs) * states.shape[1],
                        self.config.n_embd,
                    ),
                    dtype=torch.float32,
                    device=action_embeddings.device,
                )
                token_embeddings[:, :: (self.num_inputs), :] = (
                    self.skill_embedding[skill_set.long()]
                    .repeat(1, 1, 1, 1)
                    .view(skill_set.shape[0], -1, self.config.n_embd)
                )
                token_embeddings[:, 1 :: (self.num_inputs), :] = torch.cat(
                    [state_inputs[0], state_inputs[-1]], dim=-1
                )

                token_embeddings[
                    :, (self.num_inputs - 1) :: (self.num_inputs), :
                ] = action_embeddings
            else:
                raise NotImplementedError
        elif actions is not None and self.model_type == "bc_no_skill":
            actions = torch.clone(actions)
            actions[:, :, [10]] = 0
            actions = actions.type(torch.float32)
            action_embeddings = self.action_embeddings(
                actions
            )  # (batch, block_size, n_embd)
            token_embeddings = torch.zeros(
                (
                    states.shape[0],
                    (self.num_inputs - 1) * states.shape[1],
                    self.config.n_embd,
                ),
                dtype=torch.float32,
                device=action_embeddings.device,
            )
            token_embeddings[:, 0 :: (self.num_inputs - 1), :] = torch.cat(
                [state_inputs[0], state_inputs[-1]], dim=-1
            )

            token_embeddings[
                :, 1 :: (self.num_inputs - 1), :
            ] = action_embeddings
        else:
            raise NotImplementedError

        batch_size = states.shape[0]

        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]
        x = token_embeddings + position_embeddings
        if self.reg_flags["outer_dropout"]:
            x = self.drop(x)
        x = self.blocks(x)
        if self.reg_flags["outer_layernorm"]:
            x = self.ln_f(x)

        if actions is not None and self.model_type == "reward_conditioned":
            return x[:, (self.num_inputs - 2) :: (self.num_inputs), :]
        elif actions is not None and self.model_type == "bc":
            if skill_set is not None:
                return x[:, 1 :: (self.num_inputs), :]
            return x[:, (self.num_inputs - 3) :: (self.num_inputs - 1), :]
        elif actions is not None and self.model_type == "bc_no_skill":
            return x[:, (self.num_inputs - 3) :: (self.num_inputs - 1), :]
        else:
            raise NotImplementedError()


class SkillInference(nn.Module):
    """the skill inference module that outputs encoded embedding used to infer high-level skills"""
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        self.num_inputs = 1

        config.block_size = config.block_size * self.num_inputs

        self.block_size = config.block_size

        self.n_embd = config.n_embd
        # input embedding stem
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e",
            sum(p.numel() for p in self.parameters()),
        )

        if config.num_states[0] == 0:
            self.state_encoder = nn.Sequential(
                nn.Linear(config.num_states[1], config.n_embd), nn.Tanh()
            )
        else:
            self.state_encoder = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(i, config.n_embd // 2), nn.Tanh())
                    for i in [config.num_states[1]]
                ]
            )
        self.output_head = nn.Linear(config.n_embd, config.num_skills)
        self.output_head2 = nn.Linear(config.n_embd, 6)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states):
        state_inputs = list(
            torch.split(
                states, [self.n_embd // 2, self.config.num_states[1]], -1
            )
        )

        for i in range(1, len(state_inputs)):
            state_inputs[i] = self.state_encoder[i - 1](
                state_inputs[i].type(torch.float32)  # [..., 3:13]
            )

        token_embeddings = torch.zeros(
            (
                states.shape[0],
                states.shape[1],
                self.config.n_embd,
            ),
            dtype=torch.float32,
            device=states.device,
        )

        token_embeddings[:, :: self.num_inputs, :] = torch.cat(
            [state_inputs[0], state_inputs[-1]], dim=-1
        )

        position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.output_head(x), self.output_head2(x)
