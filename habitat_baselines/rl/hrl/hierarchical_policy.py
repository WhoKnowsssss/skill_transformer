import os.path as osp
from collections import defaultdict
from typing import Dict, List

import gym.spaces as spaces
import torch

from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    CompositeSuccess,
)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl import (  # noqa: F401.
    FixedHighLevelPolicy,
    HighLevelPolicy,
    StripsHighLevelPolicy,
)
from habitat_baselines.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PickOffsetSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import get_num_actions
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem


@baseline_registry.register_policy
class HierarchicalPolicy(Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: ActionSpace,
        num_envs: int,
    ):
        super().__init__()

        self._action_space = action_space
        self._num_envs: int = num_envs
        self._hidden_size = full_config.RL.PPO.hidden_size

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}
        self._idx_to_name: Dict[int, str] = {}

        task_spec_file = osp.join(
            full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
            full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",
        )
        domain_file = full_config.TASK_CONFIG.TASK.PDDL_DOMAIN_DEF

        self._pddl_problem = PddlProblem(
            domain_file,
            task_spec_file,
            config,
        )

        for i, (skill_id, use_skill_name) in enumerate(
            config.USE_SKILLS.items()
        ):
            if use_skill_name == "":
                # Skip loading this skill if no name is provided
                continue
            skill_config = config.DEFINED_SKILLS[use_skill_name]

            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i
            self._idx_to_name[i] = skill_id

        self._call_high_level: torch.Tensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )
        self._cur_skills: torch.Tensor = torch.full(
            (self._num_envs,), -1, dtype=torch.long
        )

        self._is_nav_goal: torch.Tensor = torch.full(
            (self._num_envs,), -1, dtype=torch.long
        )

        high_level_cls = eval(config.high_level_policy.name)
        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            self._pddl_problem,
            num_envs,
            self._name_to_idx,
        )
        self._stop_action_idx, _ = find_action_range(
            action_space, "REARRANGE_STOP"
        )

    @property
    def hidden_state_hxs_dim(self):
        return self._hidden_size

    def eval(self):
        pass

    def get_policy_info(self, infos, dones):
        policy_infos = []
        for i, info in enumerate(infos):
            cur_skill_idx = self._cur_skills[i].item()
            policy_info = {
                "cur_skill": self._idx_to_name[cur_skill_idx],
            }
            if self._is_nav_goal[0] == 1 and cur_skill_idx == 4:
                policy_info["cur_skill"] = 'nav_goal'

            did_skill_fail = dones[i] and not info[CompositeSuccess.cls_uuid]
            for skill_name, idx in self._name_to_idx.items():
                policy_info[f"failed_skill_{skill_name}"] = (
                    did_skill_fail if idx == cur_skill_idx else 0.0
                )
            policy_infos.append(policy_info)

        return policy_infos

    @property
    def num_recurrent_layers(self):
        return self._skills[0].num_recurrent_layers

    @property
    def should_load_agent_state(self):
        return False

    def parameters(self):
        return self._skills[0].parameters()

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)

    def _broadcast_skill_ids(
        self, skill_ids, sel_dat, should_adds=None
    ) -> Dict[int, List[int]]:
        # skill id -> [batch ids]
        grouped_skills = defaultdict(list)
        if should_adds is None:
            should_adds = [True for _ in range(len(skill_ids))]
        for i, (cur_skill, should_add) in enumerate(
            zip(skill_ids, should_adds)
        ):
            if should_add:
                cur_skill = cur_skill.item()
                grouped_skills[cur_skill].append(i)
        for k, v in grouped_skills.items():
            grouped_skills[k] = (
                v,
                {dat_k: dat[v] for dat_k, dat in sel_dat.items()},
            )
        return dict(grouped_skills)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):

        self._high_level_policy.apply_mask(masks)

        should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool, device=masks.device
        )
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool, device=masks.device
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
            # Only decide on skill termination if the episode is active.
            should_adds=masks,
        )

        # Check if skills should terminate.
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            if skill_id == -1:
                # Policy has not prediced a skill yet.
                should_terminate[batch_ids] = 1.0
                continue
            try:
                (
                    should_terminate[batch_ids],
                    bad_should_terminate[batch_ids],
                ) = self._skills[skill_id].should_terminate(
                    **dat,
                    batch_idx=batch_ids,
                )
            except Exception as e:
                print(e)
                breakpoint()
        should_terminate = should_terminate.cpu()
        bad_should_terminate = bad_should_terminate.cpu()
        self._call_high_level = should_terminate

        # Always call high-level if the episode is over.
        self._call_high_level = self._call_high_level | (~masks).view(-1).cpu()

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
        if self._call_high_level.sum() > 0:
            (
                new_skills,
                new_skill_args,
                hl_terminate,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
            )
            if new_skills[0] == 4 and 'TARGET_goal0|0' in new_skill_args[0]:
                self._is_nav_goal[0] = 1
            else:
                self._is_nav_goal[0] = 0

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                should_adds=self._call_high_level,
            )

            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                rnn_hidden_states[batch_ids] *= 0.0
                prev_actions[batch_ids] *= 0
            self._cur_skills = (
                (~self._call_high_level) * self._cur_skills
            ) + (self._call_high_level * new_skills)

        # Compute the actions from the current skills
        actions = torch.zeros(
            (self._num_envs, get_num_actions(self._action_space)),
            device=masks.device,
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
            },
        )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            tmp_actions, tmp_rnn = self._skills[skill_id].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
            )
            # LL skills are not allowed to terminate the overall episode.
            actions[batch_ids] = tmp_actions
            rnn_hidden_states[batch_ids] = tmp_rnn
        actions[:, self._stop_action_idx] = 0.0

        should_terminate = bad_should_terminate | hl_terminate
        if should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0

        return (None, actions, None, rnn_hidden_states)

    @classmethod
    def from_config(
        cls,
        config,
        observation_space,
        action_space,
        orig_action_space,
        **kwargs,
    ):
        return cls(
            config.RL.POLICY,
            config,
            observation_space,
            orig_action_space,
            config.NUM_ENVIRONMENTS,
        )
