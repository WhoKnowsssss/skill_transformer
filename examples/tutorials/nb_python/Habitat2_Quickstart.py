# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     collapsed_sections: []
#     name: Habitat 2.0 Quick Start Tutorial
#     provenance: []
#   jupytext:
#     cell_metadata_filter: -all
#     formats: nb_python//py:percent,colabs//ipynb
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Overview
# This tutorial covers the basics of using Habitat 2.0 including: setting up
# the environment, creating custom environments, and creating new episode
# datasets.

# %%
# Play a teaser video
try:
    from IPython.display import HTML

    HTML(
        '<iframe src="https://drive.google.com/file/d/1ltrse38i8pnJPGAXlThylcdy8PMjUMKh/preview" width="640" height="480" allow="autoplay"></iframe>'
    )
except Exception:
    pass

# %%
# %%capture
# @title Install Dependencies (if on Colab) { display-mode: "form" }
# @markdown (double click to show code)

import os

# Colab installation
if "COLAB_GPU" in os.environ:
    print("Setting up Habitat")
    # !curl -L https://raw.githubusercontent.com/facebookresearch/habitat-sim/main/examples/colab_utils/colab_install.sh | NIGHTLY=true bash -s
# %%
# Imports
import os

if "COLAB_GPU" in os.environ:
    print("Setting Habitat base path")
    # %env HABLAB_BASE_CFG_PATH=/content/habitat-lab
    import importlib

    import PIL

    importlib.reload(PIL.TiffTags)

import os

import gym
import gym.spaces as spaces
import numpy as np

import habitat
import habitat.utils.gym_definitions as habitat_gym
from habitat.core.embodied_task import Measure
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes
from habitat.tasks.rearrange.rearrange_sensors import RearrangeReward
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_sim.utils import viz_utils as vut

# Quiet the Habitat simulator logging
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def insert_render_options(config):
    # Added settings to make rendering higher resolution for better visualization
    config.defrost()
    config.SIMULATOR.THIRD_RGB_SENSOR.WIDTH = 512
    config.SIMULATOR.THIRD_RGB_SENSOR.HEIGHT = 512
    config.SIMULATOR.CONCUR_RENDER = False
    config.SIMULATOR.AGENT_0.SENSORS.append("THIRD_RGB_SENSOR")
    config.freeze()
    return config


import importlib

# If the import block fails due to an error like "'PIL.TiffTags' has no attribute
# 'IFD'", then restart the Colab runtime instance and rerun this cell and the previous cell.
import PIL

importlib.reload(PIL.TiffTags)  # To potentially avoid PIL problem


# %% [markdown]
# # Local installation
# Follow the steps on the [Habitat Lab README](https://github.com/facebookresearch/habitat-lab/tree/challenge_tasks#installation).

# %% [markdown]
# # Quickstart
#
# Start with a minimal environment interaction loop using the Habitat API. This sets up the environment, takes random episodes, and then saves a video once the episode ends.
#
# If this is your first time running Habitat 2.0 code, the datasets will automatically download which include the ReplicaCAD scenes, episode datasets, and object assets. To manually download this data, run `python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets`.

# %%
with habitat.Env(
    config=insert_render_options(
        habitat.get_config(
            os.path.join(
                habitat_gym.base_dir,
                "configs/tasks/rearrange/pick.yaml",
            )
        )
    )
) as env:
    observations = env.reset()  # noqa: F841

    print("Agent acting inside environment.")
    count_steps = 0
    # To save the video
    video_file_path = "data/example_interact.mp4"
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)

    while not env.episode_over:
        observations = env.step(env.action_space.sample())  # noqa: F841
        info = env.get_metrics()

        render_obs = observations_to_image(observations, info)
        render_obs = overlay_frame(render_obs, info)

        video_writer.append_data(render_obs)

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))

    video_writer.close()
    if vut.is_notebook():
        vut.display_video(video_file_path)


# %% [markdown]
# ## Gym API
# You can also use environments through the Gym API. For more information about how to use the Gym API and the supported tasks, see [this tutorial](https://github.com/facebookresearch/habitat-lab/blob/main/examples/tutorials/colabs/habitat2_gym_tutorial.ipynb).

# %%
env = gym.make("HabitatRenderPick-v0")

video_file_path = "data/example_interact.mp4"
video_writer = vut.get_fast_video_writer(video_file_path, fps=30)

done = False
env.reset()
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())
    video_writer.append_data(env.render(mode="rgb_array"))

video_writer.close()
if vut.is_notebook():
    vut.display_video(video_file_path)


# %% [markdown]
# # Defining New Tasks
#
# We will define a task for the robot to navigate to and then pick up a target object in the environment. To support a new task we need:
# * A task of type `RearrangeTask` which implements the reset function.
# * Sensor definitions to populate the observation space.
# * Measurement definitions to define the reward, termination condition, and additional logging information.
#
# For other examples of task, sensor, and measurement definitions, [see here
# for existing tasks](https://github.com/facebookresearch/habitat-lab/tree/main/habitat/tasks/rearrange/sub_tasks). Tasks, sensors, and measurements are connected through a config file that defines the task.

# %%
@registry.register_task(name="RearrangeDemoNavPickTask-v0")
class NavPickTaskV1(RearrangeTask):
    """
    Primarily this is used to implement the episode reset functionality.
    Can also implement custom episode step functionality.
    """

    def reset(self, episode):
        self.target_object_index = np.random.randint(
            0, self._sim.get_n_targets()
        )
        start_pos = self._sim.pathfinder.get_random_navigable_point()
        self._sim.robot.base_pos = start_pos

        # Put any reset logic here.
        return super().reset(episode)


@registry.register_sensor
class TargetStartSensor(Sensor):
    """
    Relative position from end effector to target object start position.
    """

    cls_uuid: str = "relative_object_to_end_effector"

    def __init__(self, sim, config, *args, task, **kwargs):
        self._sim = sim
        self._task = task
        self._config = config
        super().__init__(**kwargs)

    def _get_uuid(self, *args, **kwargs):
        return TargetStartSensor.cls_uuid

    def _get_sensor_type(self, *args, **kwargs):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args, **kwargs):
        return spaces.Box(
            shape=(3,),
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            dtype=np.float32,
        )

    def get_observation(self, observations, episode, *args, **kwargs):
        global_T = self._sim.robot.ee_transform
        T_inv = global_T.inverted()
        start_pos = self._sim.get_target_objs_start()[
            self._task.target_object_index
        ]
        relative_start_pos = T_inv.transform_point(start_pos)
        return np.asarray(relative_start_pos)


@registry.register_measure
class DistanceToTargetObject(Measure):
    """
    Gets the Euclidean distance to the target object from the end-effector.
    """

    cls_uuid: str = "distance_to_object"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return DistanceToTargetObject.cls_uuid

    def reset_metric(self, *args, episode, **kwargs):
        self.update_metric(*args, episode=episode, **kwargs)

    def update_metric(self, *args, task, episode, **kwargs):
        ee_pos = self._sim.robot.ee_transform.translation

        idxs, _ = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()[idxs[task.target_object_index]]

        # Metric information is stored in the `self._metric` variable.
        self._metric = np.linalg.norm(scene_pos - ee_pos, ord=2, axis=-1)


@registry.register_measure
class NavPickReward(RearrangeReward):
    """
    For every new task, you NEED to implement a reward function.
    `RearrangeReward` automatically includes penalties for collisions into the reward function.
    """

    cls_uuid: str = "navpick_reward"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(sim=sim, config=config, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavPickReward.cls_uuid

    def reset_metric(self, *args, task, episode, **kwargs):
        # Measurements can be computed from other measurements.
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToTargetObject.cls_uuid,
            ],
        )
        self.update_metric(*args, task=task, episode=episode, **kwargs)

    def update_metric(self, *args, task, episode, **kwargs):
        ee_to_object_distance = task.measurements.measures[
            DistanceToTargetObject.cls_uuid
        ].get_metric()

        self._metric = -ee_to_object_distance


@registry.register_measure
class NavPickSuccess(Measure):
    """
    For every new task, you NEED to implement a "success" condition.
    """

    cls_uuid: str = "navpick_success"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return NavPickSuccess.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self.update_metric(
            *args,
            episode=episode,
            task=task,
            observations=observations,
            **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        # Check that the agent is holding the correct object.
        abs_targ_obj_idx = self._sim.scene_obj_ids[task.target_object_index]
        self._metric = abs_targ_obj_idx == self._sim.grasp_mgr.snap_idx


# %% [markdown]
# We now add all the previously defined task, sensor, and measurement
# definitions to a config file to finish defining the new Habitat task. For
# examples of more configs [see here](https://github.com/facebookresearch/habitat-lab/tree/main/configs/tasks/rearrange).
#
# This config also defines the action space through the `TASK.ACTIONS` key. You
# can substitute different base control actions from
# [here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/tasks/rearrange/actions.py),
# different arm control actions [from
# here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/tasks/rearrange/actions.py),
# and different grip actions [from here](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/tasks/rearrange/grip_actions.py).

# %%
cfg_txt = """
ENVIRONMENT:
    # Number of steps within an episode.
    MAX_EPISODE_STEPS: 200
DATASET:
    TYPE: RearrangeDataset-v0
    SPLIT: train
    # The dataset to use. Later we will generate our own dataset.
    DATA_PATH: data/datasets/replica_cad/rearrange/v1/{split}/all_receptacles_10k_1k.json.gz
    SCENES_DIR: "data/replica_cad/"
TASK:
    TYPE: RearrangeDemoNavPickTask-v0

    # Sensors for the observation space.
    TARGET_START_SENSOR:
        TYPE: "TargetStartSensor"
    JOINT_SENSOR:
        TYPE: "JointSensor"
        DIMENSIONALITY: 7
    SENSORS: ["TARGET_START_SENSOR", "JOINT_SENSOR"]

    # Measurements
    ROBOT_FORCE:
        TYPE: "RobotForce"
        MIN_FORCE: 20.0
    FORCE_TERMINATE:
        TYPE: "ForceTerminate"
        # Maximum amount of allowed force in Newtons.
        MAX_ACCUM_FORCE: 5000.0
    DISTANCE_TO_TARGET_OBJECT:
        TYPE: "DistanceToTargetObject"
    NAV_PICK_REWARD:
        TYPE: "NavPickReward"
        SCALING_FACTOR: 0.1

        # General Rearrange Reward config
        CONSTRAINT_VIOLATE_PEN: 10.0
        FORCE_PEN: 0.001
        MAX_FORCE_PEN: 1.0
        FORCE_END_PEN: 10.0

    NAV_PICK_SUCCESS:
        TYPE: "NavPickSuccess"

    MEASUREMENTS:
        # The measurements returned in the info dictionary
        - "ROBOT_FORCE"
        - "FORCE_TERMINATE"
        - "DISTANCE_TO_TARGET_OBJECT"
        - "NAV_PICK_REWARD"
        - "NAV_PICK_SUCCESS"
    ACTIONS:
        # Define the action space.
        ARM_ACTION:
            TYPE: "ArmAction"
            ARM_CONTROLLER: "ArmRelPosAction"
            GRIP_CONTROLLER: "MagicGraspAction"
            ARM_JOINT_DIMENSIONALITY: 7
            GRASP_THRESH_DIST: 0.15
            DISABLE_GRIP: False
            DELTA_POS_LIMIT: 0.0125
            EE_CTRL_LIM: 0.015
        BASE_VELOCITY:
            TYPE: "BaseVelAction"
            LIN_SPEED: 12.0
            ANG_SPEED: 12.0
            ALLOW_DYN_SLIDE: True
            END_ON_STOP: False
            ALLOW_BACK: True
            MIN_ABS_LIN_SPEED: 1.0
            MIN_ABS_ANG_SPEED: 1.0
    POSSIBLE_ACTIONS:
        - ARM_ACTION
        - BASE_VELOCITY

SIMULATOR:
    ADDITIONAL_OBJECT_PATHS:
        - "data/objects/ycb/configs/"
    DEBUG_RENDER: False
    ACTION_SPACE_CONFIG: v0
    AGENTS: ['AGENT_0']
    CONCUR_RENDER: False
    AUTO_SLEEP: False
    AGENT_0:
        HEIGHT: 1.5
        IS_SET_START_STATE: False
        RADIUS: 0.1
        SENSORS: ['HEAD_RGB_SENSOR']
        START_POSITION: [0, 0, 0]
        START_ROTATION: [0, 0, 0, 1]
    HEAD_RGB_SENSOR:
        WIDTH: 128
        HEIGHT: 128

    # Agent setup
    ARM_REST: [0.6, 0.0, 0.9]
    CTRL_FREQ: 120.0
    AC_FREQ_RATIO: 4
    ROBOT_URDF: ./data/robots/hab_fetch/robots/hab_fetch.urdf
    ROBOT_TYPE: "FetchRobot"
    FORWARD_STEP_SIZE: 0.25

    # Grasping
    HOLD_THRESH: 0.09
    GRASP_IMPULSE: 1000.0

    HABITAT_SIM_V0:
        ALLOW_SLIDING: True
        ENABLE_PHYSICS: True
        GPU_DEVICE_ID: 0
        GPU_GPU: False
        PHYSICS_CONFIG_FILE: ./data/default.physics_config.json
    TYPE: RearrangeSim-v0
"""
nav_pick_cfg_path = "data/nav_pick_demo.yaml"
with open(nav_pick_cfg_path, "w") as f:
    f.write(cfg_txt)

# %% [markdown]
# The new task can then be imported via the yaml file.

# %%
with habitat.Env(
    config=insert_render_options(habitat.get_config(nav_pick_cfg_path))
) as env:
    env.reset()

    print("Agent acting inside environment.")
    count_steps = 0
    # To save the video
    video_file_path = "data/example_interact.mp4"
    video_writer = vut.get_fast_video_writer(video_file_path, fps=30)

    while not env.episode_over:
        action = env.action_space.sample()
        observations = env.step(action)  # noqa: F841
        info = env.get_metrics()

        render_obs = observations_to_image(observations, info)
        render_obs = overlay_frame(render_obs, info)

        video_writer.append_data(render_obs)

        count_steps += 1
    print("Episode finished after {} steps.".format(count_steps))

    video_writer.close()
    if vut.is_notebook():
        vut.display_video(video_file_path)

# %% [markdown]
# # Dataset Generation
# The previously defined task uses an included default `all_receptacles_10k_1k.json.gz` dataset which places objects on any receptacle. The episode `.json.gz` dataset defines where
# objects are placed and their rearrangement target positions. New episode
# datasets are generated with the [run_episode_generator.py](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/datasets/rearrange/run_episode_generator.py) script. In this example, we will define a new episode dataset where a single object spawns on the table with its goal also on the table.

# %%
dataset_cfg_txt = """
---
dataset_path: "data/replica_cad/replicaCAD.scene_dataset_config.json"
additional_object_paths:
  - "data/objects/ycb/configs/"
scene_sets:
  -
    name: "v3_sc"
    included_substrings:
      - "v3_sc"
    excluded_substrings: []
    comment: "This set (v3_sc) selects all 105 ReplicaCAD variations with static furniture."

object_sets:
  -
    name: "kitchen"
    included_substrings:
      - "002_master_chef_can"
      - "003_cracker_box"
    excluded_substrings: []
    comment: "Leave included_substrings empty to select all objects."

receptacle_sets:
  -
    name: "table"
    included_object_substrings:
      - "frl_apartment_table_01"
    excluded_object_substrings: []
    included_receptacle_substrings:
      - ""
    excluded_receptacle_substrings: []
    comment: "The empty substrings act like wildcards, selecting all receptacles for all objects."

scene_sampler:
  type: "subset"
  params:
    scene_sets: ["v3_sc"]
  comment: "Samples from ReplicaCAD 105 variations with static furniture."


object_samplers:
  -
    name: "kitchen_counter"
    type: "uniform"
    params:
      object_sets: ["kitchen"]
      receptacle_sets: ["table"]
      num_samples: [1, 1]
      orientation_sampling: "up"

object_target_samplers:
  -
    name: "kitchen_counter_targets"
    type: "uniform"
    params:
      object_samplers: ["kitchen_counter"]
      receptacle_sets: ["table"]
      num_samples: [1, 1]
      orientation_sampling: "up"
"""
nav_pick_cfg_path = "data/nav_pick_dataset.yaml"
with open(nav_pick_cfg_path, "w") as f:
    f.write(dataset_cfg_txt)

# %%
# !python -m habitat.datasets.rearrange.run_episode_generator --run --config data/nav_pick_dataset.yaml --num-episodes 10 --out data/nav_pick.json.gz

# %% [markdown]
# To use this dataset set `DATASET.DATA_PATH = data/nav_pick.json.gz` in the task config. See the full set of possible objects, receptacles, and scenes with `python -m habitat.datasets.rearrange.run_episode_generator --list`
