from dataclasses import dataclass

import gym.spaces as spaces
import torch

from habitat.tasks.rearrange.rearrange_sensors import (
    TargetGoalGpsCompassSensor,
    TargetStartGpsCompassSensor,
)
from habitat.tasks.rearrange.sub_tasks.nav_to_obj_sensors import (
    NavGoalPointGoalSensor,
)
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy


class NavSkillPolicy(NnSkillPolicy):
    @dataclass(frozen=True)
    class NavArgs:
        obj_idx: int
        is_target: bool

    def __init__(
        self,
        wrap_policy,
        config,
        action_space: spaces.Space,
        filtered_obs_space: spaces.Space,
        filtered_action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
            should_keep_hold_state=True,
        )

    def _get_filtered_obs(self, observations, cur_batch_idx) -> TensorDict:
        ret_obs = super()._get_filtered_obs(observations, cur_batch_idx)

        if NavGoalPointGoalSensor.cls_uuid in ret_obs:
            idx_dict = {
                TargetGoalGpsCompassSensor.cls_uuid: [],
                TargetStartGpsCompassSensor.cls_uuid: [],
            }
            for idx, i in enumerate(cur_batch_idx):
                if self._cur_skill_args[i].is_target:
                    replace_sensor = TargetGoalGpsCompassSensor.cls_uuid
                else:
                    replace_sensor = TargetStartGpsCompassSensor.cls_uuid
                idx_dict[replace_sensor].append(idx)
            for k, v in idx_dict.items():
                ret_obs[NavGoalPointGoalSensor.cls_uuid][v] = observations[k][v]
            self.left_dist = ret_obs[NavGoalPointGoalSensor.cls_uuid][:,0]
        return ret_obs

    def _get_multi_sensor_index(self, batch_idx):
        return [self._cur_skill_args[i].obj_idx for i in batch_idx]

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        prob_done = (self._did_want_done[batch_idx] > 0.0).to(masks.device)
        # mask = ~(observations['is_holding'].reshape(-1)).bool()
        # prob_done[mask] = prob_done[mask] & (self.left_dist[mask] < 1.)
        return prob_done# & (self.left_dist < .5)

    def _parse_skill_arg(self, skill_arg):
        targ_name, targ_idx = skill_arg[-2].split("|")
        return NavSkillPolicy.NavArgs(
            obj_idx=int(targ_idx), is_target=targ_name.startswith("TARGET")
        )
