from typing import Any, List, Tuple

import torch
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat.tasks.rearrange.multi_task.composite_sensors import (
    GlobalPredicatesSensor,
)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import (
    FRIDGE_TYPE,
    CAB_TYPE,
)
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func


class StripsHighLevelPolicy(HighLevelPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._preds_list = self._pddl_problem.get_possible_predicates()
        self._next_sol_idxs = torch.zeros(self._num_envs, dtype=torch.int32)
        self._env_to_plan = {env_i: None for env_i in range(self._num_envs)}

    def _is_in_closed_recep(self, search_name, pred_vals, env_i):
        matches = [
            (i, x)
            for i, x in enumerate(self._preds_list)
            if x.name == "in"
            and x._arg_values[0].name == search_name
            and pred_vals[env_i, i].item() == 1.0
        ]

        for i, x in matches:
            recep = x.arg_values[1]
            match_closed = [
                (i, x)
                for i, x in enumerate(self._preds_list)
                if x.name.startswith("closed_") and x.arg_values[0] == recep
            ]
            if len(match_closed) != 1:
                raise ValueError()
            if pred_vals[env_i, match_closed[0][0]] == 1.0:
                return recep
        return None

    def _create_ac_list(self, obj_in_recep, goal_in_recep):
        sol_actions = [
            "nav(goal0|0, ROBOT_0)",
            "pick(goal0|0, ROBOT_0)",
            "nav(TARGET_goal0|0, ROBOT_0)",
            "place(goal0|0, TARGET_goal0|0, ROBOT_0)",
        ]

        def get_recep_ac(recep, obj_name):
            if recep.expr_type.name == FRIDGE_TYPE:
                name = "open_fridge"
            else:
                name = "open_cab"
            return f"{name}({recep.name}, {obj_name}, ROBOT_0)"

        if obj_in_recep is not None:
            sol_actions.insert(1, get_recep_ac(obj_in_recep, "goal0|0"))
            if 'open_cab' in sol_actions[1]:
                sol_actions[2] = 'pick_offset' + sol_actions[2].split('pick')[-1]
        if goal_in_recep is not None:
            sol_actions.insert(
                # Must add relative to the end in case we added an open after the first nav.
                len(sol_actions) - 1,
                get_recep_ac(goal_in_recep, "TARGET_goal0|0"),
            )

        final_sol_actions = []
        for i, sol_step in enumerate(sol_actions):
            final_sol_actions.append(parse_func(sol_step))
            if self._config.add_arm_rest and i < (len(sol_actions) - 1):
                final_sol_actions.append(parse_func("reset_arm(0)"))

        # Add a wait action at the end.
        final_sol_actions.append(parse_func("wait(30)"))
        return final_sol_actions

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        pred_vals = observations[GlobalPredicatesSensor.cls_uuid]
        for env_i, plan in self._env_to_plan.items():
            if plan is None:
                self._env_to_plan[env_i] = self._create_ac_list(
                    self._is_in_closed_recep("goal0|0", pred_vals, env_i),
                    self._is_in_closed_recep(
                        "TARGET_goal0|0", pred_vals, env_i
                    ),
                )

        next_skill = torch.zeros(self._num_envs)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                plan = self._env_to_plan[batch_idx]
                if self._next_sol_idxs[batch_idx] >= len(plan):
                    immediate_end[batch_idx] = True
                    use_idx = len(plan) - 1
                else:
                    use_idx = self._next_sol_idxs[batch_idx].item()

                skill_name, skill_args = plan[use_idx]
                # print(skill_name, self._env_to_plan)
                # breakpoint()
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]

                skill_args_data[batch_idx] = skill_args

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end

    def apply_mask(self, mask: torch.Tensor) -> None:
        for i, mask_ele in enumerate(mask):
            if mask_ele == 0.0:
                self._env_to_plan[i] = None
        self._next_sol_idxs *= mask.cpu().view(-1)
