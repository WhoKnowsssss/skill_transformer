import os, pickle, time
from this import d
from typing import Any, ClassVar, Dict, List, Tuple, Union, Optional
import itertools as its
from collections import deque

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
import numba


from habitat import Config, logger
from habitat_baselines.common.tensor_dict import TensorDict


def read_dataset(
    config: Config,
    verbose: bool,
    rng: np.random.Generator,
    context_length: int = 30,
):
    obss = []
    actions = []
    done_idxs = []
    stepwise_returns = []

    skill_dict = {
        "nav": 0,
        "pick": 1,
        "pick_offset": 3,
        "place": 2,
        "open_cab": 5,
        "open_fridge": 5,
        "reset_arm": -1,
        "wait": -1,
    }

    paths = config.trajectory_dir
    dataset_size = config.dataset_size
    if not isinstance(paths, list):
        paths = [paths]
    filenames = []
    for path in paths:
        if dataset_size != -1:
            filenames_folder = os.listdir(path)[:dataset_size]
        else:
            filenames_folder = os.listdir(path)
        filenames_folder = [
            os.path.join(path, filenames_folder[i])
            for i in range(len(filenames_folder))
        ]
        filenames.extend(filenames_folder)

    if verbose:
        logger.info("Trajectory Files: {}".format(filenames))

    transitions_per_buffer = np.zeros(len(filenames), dtype=int)
    num_trajectories = 0
    previous_done = 0
    while len(obss) < config.files_per_load:
        buffer_num = rng.choice(np.arange(len(filenames)), 1, replace=False)[0]
        i = transitions_per_buffer[buffer_num]
        if verbose:
            logger.info("Loading from buffer {}".format(buffer_num, i))
        file = filenames[buffer_num]
        try:
            file = os.readlink(file)
        except Exception as e:
            pass
        if os.path.exists(file):
            import time

            s = time.perf_counter()
            try:
                dataset_raw = torch.load(
                    file, map_location=torch.device("cpu")
                )
            except Exception as e:
                print("skip", e)
                continue

            temp_obs = np.array(dataset_raw["obs"])
            temp_actions = torch.stack(dataset_raw["actions"]).numpy()[:, :12]
            temp_stepwise_returns = torch.cat(dataset_raw["rewards"]).numpy()
            temp_dones = torch.cat(dataset_raw["masks"]).numpy()
            temp_infos = np.array(dataset_raw["infos"])

            # ==================== Categorize Gripper Action ===================
            # categorize gripper action into a one-time "status change"
            # action to prevent label imbalance.
            temp_actions = np.clip(temp_actions, -1, 1)

            temp_actions[:, 10] = 0
            temp_pick_action = np.stack(
                [temp_obs[i]["is_holding"] for i in range(len(temp_obs))]
            )
            change = temp_pick_action[1:-1] - temp_pick_action[:-2]
            ii = np.where(change < 0)[0]
            try:
                ii = np.concatenate(
                    [
                        np.arange(iii - 1, min(iii + 1, len(temp_actions) - 1))
                        for iii in ii
                    ]
                )
                temp_actions[ii, 10] = 1
            except:
                pass
            ii = np.where(change > 0)[0]
            try:
                ii = np.concatenate(
                    [
                        np.arange(iii - 1, min(iii + 1, len(temp_actions) - 1))
                        for iii in ii
                    ]
                )
                temp_actions[ii, 10] = 2
            except:
                pass

            # ==================== Planner Targets ==================
            try:
                for i in range(len(temp_obs)):
                    temp_obs[i]["skill"] = skill_dict[temp_infos[i]["skill"]]

                    # replace hard-coded skills with skill before it
                    if temp_obs[i]["skill"] == -1:
                        if temp_obs[i - 1]["skill"] != 5:
                            temp_obs[i]["skill"] = temp_obs[i - 1]["skill"]
                        else:
                            temp_obs[i]["skill"] = 3

                    # separate nav_to_start and nav_to_goal
                    if (
                        temp_obs[i]["skill"] == 0
                        and temp_obs[i]["is_holding"] == 1
                    ):
                        temp_obs[i]["skill"] = 4

                    if temp_obs[i]["skill"] == 4 or temp_obs[i]["skill"] == 0:
                        temp_actions[i, :7] = 0
            except:
                continue

            # ======================== Add Missing Keys ========================
            if "all_predicates" not in temp_obs[0].keys():
                temp_missing_obs = torch.zeros((1, 47))
                for i in range(len(temp_obs)):
                    temp_obs[i]["all_predicates"] = temp_missing_obs[0, :35]

            # delete hacks used in expert
            if "abs_obj_start_sensor" in temp_obs[0].keys():
                for i in range(len(temp_obs)):
                    try:
                        temp_obs[i].pop("abs_obj_start_sensor")
                        temp_obs[i].pop("ee_pos")
                        temp_obs[i].pop("obj_start_offset_sensor")
                        temp_obs[i].pop("obj_goal_pos_sensor")
                    except KeyError:
                        pass

            obss += [temp_obs]
            actions += [temp_actions]
            done_idxs += [temp_done_idxs + previous_done]
            previous_done += len(temp_actions)
            stepwise_returns += [temp_stepwise_returns]

    actions = np.concatenate(actions)
    obss = np.concatenate(obss).tolist()
    stepwise_returns = np.concatenate(stepwise_returns)
    done_idxs = np.concatenate(done_idxs)

    rtg, timesteps = _timesteps_rtg(done_idxs, stepwise_returns)

    if verbose:
        logger.info(
            "In this load, max rtg is {}, max timestep is {}. ".format(
                rtg.max().round(2), timesteps.max()
            )
        )

    obss = TensorDict.from_tree(default_collate(obss))
    actions = torch.from_numpy(actions).to(torch.float32)
    rtg = torch.from_numpy(rtg).to(torch.float32)
    timesteps = torch.from_numpy(timesteps).to(torch.int64)
    return obss, actions, done_idxs, rtg, timesteps


@numba.jit(nopython=True, parallel=True)
def _timesteps_rtg(done_idxs, stepwise_returns):
    rtg = np.zeros_like(stepwise_returns)
    timesteps = np.zeros(len(stepwise_returns), dtype=np.int64)
    start_index = np.concatenate(
        (np.array([0], dtype=np.int64), done_idxs[:-1])
    )
    for i in numba.prange(len(done_idxs)):
        start = start_index[i]
        done = done_idxs[i]
        curr_traj_returns = stepwise_returns[start:done]

        for j in numba.prange(done - start):
            rtg[j + start] = np.sum(curr_traj_returns[j:])

        timesteps[start:done] = np.arange(done - start)
    return rtg, timesteps


def producer(
    config: Config,
    rng: np.random.Generator,
    deque: deque,
    verbose: bool,
    context_length: int = 30,
):
    import time

    while True:
        if len(deque) < config.queue_size:
            s = time.perf_counter()
            deque.append(read_dataset(config, verbose, rng, context_length))
            print(
                "dataset loaded, load time: ",
                time.perf_counter() - s,
                " seconds. ",
            )
        else:
            time.sleep(1)
