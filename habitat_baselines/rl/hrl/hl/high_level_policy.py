from typing import Any, List, Tuple

import torch


class HighLevelPolicy:

    def __init__(
        self,
        config,
        pddl_problem,
        num_envs,
        skill_name_to_idx,
    ):
        super().__init__()
        self._config = config
        self._num_envs = num_envs
        self._pddl_problem = pddl_problem

        self._entities_list = self._pddl_problem.get_ordered_entities_list()
        self._action_ordering = self._pddl_problem.get_ordered_actions()
        self._skill_name_to_idx = skill_name_to_idx
        self._skill_idx_to_name = {v: k for k, v in skill_name_to_idx.items()}

    def get_next_skill(
        self, observations, rnn_hidden_states, prev_actions, masks, plan_masks
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """
        raise NotImplementedError()

    def apply_mask(self, mask: torch.Tensor) -> None:
        pass
