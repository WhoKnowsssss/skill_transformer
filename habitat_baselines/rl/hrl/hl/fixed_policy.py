from typing import List, Tuple

import torch
import yaml

from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy


class FixedHighLevelPolicy(HighLevelPolicy):
    """
    :property _solution_actions: List of tuples were first tuple element is the
        action name and the second is the action arguments.
    """

    _solution_actions: List[Tuple[str, List[str]]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sol_actions = self._pddl_problem.solution
        if sol_actions is None:
            raise ValueError(
                f"The ground truth task planner only works when the task solution is hard-coded in the PDDL problem file."
            )
        self._solution_actions = []
        for i, sol_step in enumerate(sol_actions):
            sol_action = [
                sol_step.name,
                [x.name for x in sol_step.param_values],
            ]
            self._solution_actions.append(sol_action)
            if self._config.add_arm_rest and i < (len(sol_actions) - 1):
                self._solution_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        self._solution_actions.append(parse_func("wait(30)"))

        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ):
        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                if self._next_sol_idxs[batch_idx] >= len(
                    self._solution_actions
                ):
                    baselines_logger.info(
                        f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
                    )
                    immediate_end[batch_idx] = True
                    use_idx = len(self._solution_actions) - 1
                else:
                    use_idx = self._next_sol_idxs[batch_idx].item()

                skill_name, skill_args = self._solution_actions[use_idx]
                baselines_logger.info(
                    f"Got next element of the plan with {skill_name}, {skill_args}"
                )
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end
