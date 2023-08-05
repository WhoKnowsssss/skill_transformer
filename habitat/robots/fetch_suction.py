import numpy as np

from habitat.robots import FetchRobot


class FetchSuctionRobot(FetchRobot):
    def _get_fetch_params(self):
        params = super()._get_fetch_params()
        params.gripper_init_params = None
        params.gripper_closed_state = np.array([0.0], dtype=np.float32)
        params.gripper_open_state = np.array([0.0], dtype=np.float32)
        params.gripper_joints = [23]

        params.ee_link = 23
        return params
