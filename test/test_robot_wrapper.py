#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from os import path as osp

import numpy as np
import pytest

import habitat.robots.fetch_robot as fetch_robot
import habitat.robots.franka_robot as franka_robot
import habitat_sim
import habitat_sim.agent

default_sim_settings = {
    # settings shared by example.py and benchmark.py
    "max_frames": 1000,
    "width": 640,
    "height": 480,
    "default_agent": 0,
    "sensor_height": 1.5,
    "hfov": 90,
    "color_sensor": True,  # RGB sensor (default: ON)
    "semantic_sensor": False,  # semantic sensor (default: OFF)
    "depth_sensor": False,  # depth sensor (default: OFF)
    "ortho_rgba_sensor": False,  # Orthographic RGB sensor (default: OFF)
    "ortho_depth_sensor": False,  # Orthographic depth sensor (default: OFF)
    "ortho_semantic_sensor": False,  # Orthographic semantic sensor (default: OFF)
    "fisheye_rgba_sensor": False,
    "fisheye_depth_sensor": False,
    "fisheye_semantic_sensor": False,
    "equirect_rgba_sensor": False,
    "equirect_depth_sensor": False,
    "equirect_semantic_sensor": False,
    "seed": 1,
    "silent": False,  # do not print log info (default: OFF)
    # settings exclusive to example.py
    "save_png": False,  # save the pngs to disk (default: OFF)
    "print_semantic_scene": False,
    "print_semantic_mask_stats": False,
    "compute_shortest_path": False,
    "compute_action_shortest_path": False,
    "scene": "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
    "test_scene_data_url": "http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip",
    "goal_position": [5.047, 0.199, 11.145],
    "enable_physics": False,
    "enable_gfx_replay_save": False,
    "physics_config_file": "./data/default.physics_config.json",
    "num_objects": 10,
    "test_object_index": 0,
    "frustum_culling": True,
}

# build SimulatorConfiguration
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    if "scene_dataset_config_file" in settings:
        sim_cfg.scene_dataset_config_file = settings[
            "scene_dataset_config_file"
        ]
    sim_cfg.frustum_culling = settings.get("frustum_culling", False)
    if "enable_physics" in settings:
        sim_cfg.enable_physics = settings["enable_physics"]
    if "physics_config_file" in settings:
        sim_cfg.physics_config_file = settings["physics_config_file"]
    if not settings["silent"]:
        print("sim_cfg.physics_config_file = " + sim_cfg.physics_config_file)
    if "scene_light_setup" in settings:
        sim_cfg.scene_light_setup = settings["scene_light_setup"]
    sim_cfg.gpu_device_id = 0
    if not hasattr(sim_cfg, "scene_id"):
        raise RuntimeError(
            "Error: Please upgrade habitat-sim. SimulatorConfig API version mismatch"
        )
    sim_cfg.scene_id = settings["scene"]

    # define default sensor parameters (see src/esp/Sensor/Sensor.h)
    sensor_specs = []

    def create_camera_spec(**kw_args):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        camera_sensor_spec.resolution = [settings["height"], settings["width"]]
        camera_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(camera_sensor_spec, k, kw_args[k])
        return camera_sensor_spec

    if settings["color_sensor"]:
        color_sensor_spec = create_camera_spec(
            uuid="color_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(color_sensor_spec)

    if settings["depth_sensor"]:
        depth_sensor_spec = create_camera_spec(
            uuid="depth_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(depth_sensor_spec)

    if settings["semantic_sensor"]:
        semantic_sensor_spec = create_camera_spec(
            uuid="semantic_sensor",
            hfov=settings["hfov"],
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.PINHOLE,
        )
        sensor_specs.append(semantic_sensor_spec)

    if settings["ortho_rgba_sensor"]:
        ortho_rgba_sensor_spec = create_camera_spec(
            uuid="ortho_rgba_sensor",
            sensor_type=habitat_sim.SensorType.COLOR,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_rgba_sensor_spec)

    if settings["ortho_depth_sensor"]:
        ortho_depth_sensor_spec = create_camera_spec(
            uuid="ortho_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_depth_sensor_spec)

    if settings["ortho_semantic_sensor"]:
        ortho_semantic_sensor_spec = create_camera_spec(
            uuid="ortho_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
            sensor_subtype=habitat_sim.SensorSubType.ORTHOGRAPHIC,
        )
        sensor_specs.append(ortho_semantic_sensor_spec)

    # TODO Figure out how to implement copying of specs
    def create_fisheye_spec(**kw_args):
        fisheye_sensor_spec = habitat_sim.FisheyeSensorDoubleSphereSpec()
        fisheye_sensor_spec.uuid = "fisheye_sensor"
        fisheye_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        fisheye_sensor_spec.sensor_model_type = (
            habitat_sim.FisheyeSensorModelType.DOUBLE_SPHERE
        )

        # The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
        # Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
        # Camera Model, The International Conference on 3D Vision (3DV), 2018
        # You can find the intrinsic parameters for the other lenses in the same table as well.
        fisheye_sensor_spec.xi = -0.27
        fisheye_sensor_spec.alpha = 0.57
        fisheye_sensor_spec.focal_length = [364.84, 364.86]

        fisheye_sensor_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        # The default principal_point_offset is the middle of the image
        fisheye_sensor_spec.principal_point_offset = None
        # default: fisheye_sensor_spec.principal_point_offset = [i/2 for i in fisheye_sensor_spec.resolution]
        fisheye_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(fisheye_sensor_spec, k, kw_args[k])
        return fisheye_sensor_spec

    if settings["fisheye_rgba_sensor"]:
        fisheye_rgba_sensor_spec = create_fisheye_spec(
            uuid="fisheye_rgba_sensor"
        )
        sensor_specs.append(fisheye_rgba_sensor_spec)
    if settings["fisheye_depth_sensor"]:
        fisheye_depth_sensor_spec = create_fisheye_spec(
            uuid="fisheye_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(fisheye_depth_sensor_spec)
    if settings["fisheye_semantic_sensor"]:
        fisheye_semantic_sensor_spec = create_fisheye_spec(
            uuid="fisheye_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(fisheye_semantic_sensor_spec)

    def create_equirect_spec(**kw_args):
        equirect_sensor_spec = habitat_sim.EquirectangularSensorSpec()
        equirect_sensor_spec.uuid = "equirect_rgba_sensor"
        equirect_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        equirect_sensor_spec.resolution = [
            settings["height"],
            settings["width"],
        ]
        equirect_sensor_spec.position = [0, settings["sensor_height"], 0]
        for k in kw_args:
            setattr(equirect_sensor_spec, k, kw_args[k])
        return equirect_sensor_spec

    if settings["equirect_rgba_sensor"]:
        equirect_rgba_sensor_spec = create_equirect_spec(
            uuid="equirect_rgba_sensor"
        )
        sensor_specs.append(equirect_rgba_sensor_spec)

    if settings["equirect_depth_sensor"]:
        equirect_depth_sensor_spec = create_equirect_spec(
            uuid="equirect_depth_sensor",
            sensor_type=habitat_sim.SensorType.DEPTH,
            channels=1,
        )
        sensor_specs.append(equirect_depth_sensor_spec)

    if settings["equirect_semantic_sensor"]:
        equirect_semantic_sensor_spec = create_equirect_spec(
            uuid="equirect_semantic_sensor",
            sensor_type=habitat_sim.SensorType.SEMANTIC,
            channels=1,
        )
        sensor_specs.append(equirect_semantic_sensor_spec)

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
        ),
    }

    # override action space to no-op to test physics
    if sim_cfg.enable_physics:
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            )
        }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def simulate(sim, dt, get_observations=False):
    r"""Runs physics simulation at 60FPS for a given duration (dt) optionally collecting and returning sensor observations."""
    observations = []
    target_time = sim.get_world_time() + dt
    while sim.get_world_time() < target_time:
        sim.step_physics(1.0 / 60.0)
        if get_observations:
            observations.append(sim.get_sensor_observations())
    return observations


@pytest.mark.skipif(
    not osp.exists("data/robots/hab_fetch"),
    reason="Test requires Fetch robot URDF and assets.",
)
@pytest.mark.skipif(
    not habitat_sim.built_with_bullet,
    reason="Robot wrapper API requires Bullet physics.",
)
@pytest.mark.parametrize("fixed_base", [True, False])
def test_fetch_robot_wrapper(fixed_base):
    """Test the fetch robot."""
    # set this to output test results as video for easy investigation
    produce_debug_video = False
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # loading the physical scene
    hab_cfg = make_cfg(cfg_settings)

    with habitat_sim.Simulator(hab_cfg) as sim:
        obj_template_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [0.0, -1.0, 2.0]

        # add a ground plane
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(
            cube_handle
        )
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        # compute a navmesh on the ground plane
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings, True)
        sim.navmesh_visualization = True

        # add the robot to the world via the wrapper
        robot_path = "data/robots/hab_fetch/robots/hab_fetch.urdf"
        fetch = fetch_robot.FetchRobot(robot_path, sim, fixed_base=fixed_base)
        fetch.reconfigure()
        assert fetch.get_robot_sim_id() == 1  # 0 is the ground plane
        print(fetch.get_link_and_joint_names())
        observations += simulate(sim, 1.0, produce_debug_video)

        # retract the arm
        observations += fetch._interpolate_arm_control(
            [1.2299035787582397, 2.345386505126953],
            [
                fetch.params.arm_joints[1],
                fetch.params.arm_joints[3],
            ],
            1,
            30,
            produce_debug_video,
        )

        # ready the arm
        observations += fetch._interpolate_arm_control(
            [-0.45, 0.1],
            [
                fetch.params.arm_joints[1],
                fetch.params.arm_joints[3],
            ],
            1,
            30,
            produce_debug_video,
        )

        # setting arm motor positions
        fetch.arm_motor_pos = np.zeros(len(fetch.params.arm_joints))
        observations += simulate(sim, 1.0, produce_debug_video)

        # set base ground position from navmesh
        # NOTE: because the navmesh floats above the collision geometry we should see a pop/settle with dynamics and no fixed base
        target_base_pos = sim.pathfinder.snap_point(fetch.sim_obj.translation)
        fetch.base_pos = target_base_pos
        assert fetch.base_pos == target_base_pos
        observations += simulate(sim, 1.0, produce_debug_video)
        if fixed_base:
            assert np.allclose(fetch.base_pos, target_base_pos)
        else:
            assert not np.allclose(fetch.base_pos, target_base_pos)

        # arm joint queries and setters
        print(f" Arm joint velocities = {fetch.arm_velocity}")
        fetch.arm_joint_pos = np.ones(len(fetch.params.arm_joints))
        fetch.arm_motor_pos = np.ones(len(fetch.params.arm_joints))
        print(f" Arm joint positions (should be ones) = {fetch.arm_joint_pos}")
        print(f" Arm joint limits = {fetch.arm_joint_limits}")
        fetch.arm_motor_pos = fetch.arm_motor_pos
        observations += simulate(sim, 1.0, produce_debug_video)

        # test gripper state
        fetch.open_gripper()
        observations += simulate(sim, 1.0, produce_debug_video)
        assert fetch.is_gripper_open
        assert not fetch.is_gripper_closed
        fetch.close_gripper()
        observations += simulate(sim, 1.0, produce_debug_video)
        assert fetch.is_gripper_closed
        assert not fetch.is_gripper_open

        # halfway open
        fetch.set_gripper_target_state(0.5)
        observations += simulate(sim, 0.5, produce_debug_video)
        assert not fetch.is_gripper_open
        assert not fetch.is_gripper_closed

        # kinematic open/close (checked before simulation)
        fetch.gripper_joint_pos = fetch.params.gripper_open_state
        assert np.allclose(
            fetch.gripper_joint_pos,
            fetch.params.gripper_open_state,
        )
        assert fetch.is_gripper_open
        observations += simulate(sim, 0.2, produce_debug_video)
        fetch.gripper_joint_pos = fetch.params.gripper_closed_state
        assert fetch.is_gripper_closed
        observations += simulate(sim, 0.2, produce_debug_video)

        # end effector queries
        print(f" End effector link id = {fetch.ee_link_id}")
        print(f" End effector local offset = {fetch.ee_local_offset}")
        print(f" End effector transform = {fetch.ee_transform}")
        print(
            f" End effector translation (at current state) = {fetch.calculate_ee_forward_kinematics(fetch.sim_obj.joint_positions)}"
        )
        invalid_ef_target = np.array([100.0, 200.0, 300.0])
        print(
            f" Clip end effector target ({invalid_ef_target}) to reach = {fetch.clip_ee_to_workspace(invalid_ef_target)}"
        )

        # produce some test debug video
        if produce_debug_video:
            from habitat_sim.utils import viz_utils as vut

            vut.make_video(
                observations,
                "color_sensor",
                "color",
                "test_fetch_robot_wrapper__fixed_base=" + str(fixed_base),
                open_vid=True,
            )


@pytest.mark.skipif(
    not osp.exists("data/robots/franka_panda"),
    reason="Test requires Franka robot URDF and assets.",
)
def test_franka_robot_wrapper():
    """Test the franka robot."""
    # set this to output test results as video for easy investigation
    produce_debug_video = False
    observations = []
    cfg_settings = default_sim_settings.copy()
    cfg_settings["scene"] = "NONE"
    cfg_settings["enable_physics"] = True

    # loading the physical scene
    hab_cfg = make_cfg(cfg_settings)

    with habitat_sim.Simulator(hab_cfg) as sim:
        obj_template_mgr = sim.get_object_template_manager()
        rigid_obj_mgr = sim.get_rigid_object_manager()

        # setup the camera for debug video (looking at 0,0,0)
        sim.agents[0].scene_node.translation = [0.0, -1.0, 2.0]

        # add a ground plane
        cube_handle = obj_template_mgr.get_template_handles("cubeSolid")[0]
        cube_template_cpy = obj_template_mgr.get_template_by_handle(
            cube_handle
        )
        cube_template_cpy.scale = np.array([5.0, 0.2, 5.0])
        obj_template_mgr.register_template(cube_template_cpy)
        ground_plane = rigid_obj_mgr.add_object_by_template_handle(cube_handle)
        ground_plane.translation = [0.0, -0.2, 0.0]
        ground_plane.motion_type = habitat_sim.physics.MotionType.STATIC

        # compute a navmesh on the ground plane
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings, True)
        sim.navmesh_visualization = True

        # add the robot to the world via the wrapper
        robot_path = "data/robots/franka_panda/panda_arm.urdf"
        franka = franka_robot.FrankaRobot(urdf_path=robot_path, sim=sim)
        franka.reconfigure()
        assert franka.get_robot_sim_id() == 1  # 0 is the ground plane
        print(franka.get_link_and_joint_names())
        observations += simulate(sim, 1.0, produce_debug_video)

        # move the arm
        observations += franka._interpolate_arm_control(
            [1.2299035787582397, 2.345386505126953],
            [
                franka.params.arm_joints[1],
                franka.params.arm_joints[3],
            ],
            1,
            30,
            produce_debug_video,
        )

        # move the arm
        observations += franka._interpolate_arm_control(
            [-0.45, 0.1],
            [
                franka.params.arm_joints[1],
                franka.params.arm_joints[3],
            ],
            1,
            30,
            produce_debug_video,
        )

        # setting arm motor positions
        franka.arm_motor_pos = np.zeros(len(franka.params.arm_joints))
        observations += simulate(sim, 1.0, produce_debug_video)
        assert np.allclose(
            franka.arm_motor_pos,
            np.zeros(len(franka.params.arm_joints)),
        )

        # arm joint queries and setters
        print(f" Arm joint velocities = {franka.arm_velocity}")
        franka.arm_joint_pos = np.ones(len(franka.params.arm_joints))
        franka.arm_motor_pos = np.ones(len(franka.params.arm_joints))
        print(
            f" Arm joint positions (should be ones) = {franka.arm_joint_pos}"
        )
        print(f" Arm joint limits = {franka.arm_joint_limits}")
        franka.arm_motor_pos = franka.arm_motor_pos
        observations += simulate(sim, 1.0, produce_debug_video)

        # end effector queries
        print(f" End effector link id = {franka.ee_link_id}")
        print(f" End effector local offset = {franka.ee_local_offset}")
        print(f" End effector transform = {franka.ee_transform}")
        print(
            f" End effector translation (at current state) = {franka.calculate_ee_forward_kinematics(franka.sim_obj.joint_positions)}"
        )
        invalid_ef_target = np.array([100.0, 200.0, 300.0])
        print(
            f" Clip end effector target ({invalid_ef_target}) to reach = {franka.clip_ee_to_workspace(invalid_ef_target)}"
        )

        # produce some test debug video
        if produce_debug_video:
            from habitat_sim.utils import viz_utils as vut

            vut.make_video(
                observations,
                "color_sensor",
                "color",
                "test_franka_robot_wrapper",
                open_vid=True,
            )
