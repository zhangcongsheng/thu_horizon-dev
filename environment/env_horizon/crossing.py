# -*- coding: utf-8 -*-
import math
import os
import platform
import random
import warnings
import PIL
from collections import OrderedDict
from math import cos, pi, sin

import gym

# import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.utils import seeding
from matplotlib.path import Path

sys_type = platform.system()
if sys_type == "Linux":
    """
    NOTE: run without screen, need to comment out when rendering In Linux

    """
    pass
    # matplotlib.use('Agg')
elif sys_type == "Windows":
    from matplotlib.backends.backend_agg import FigureCanvasAgg


from environment.env_horizon.dynamics_and_models import (
    VehicleDynamics,
    ReferencePath,
    environment_model,
)
from environment.env_horizon.endtoend_env_utils import (
    shift_coordination,
    rotate_coordination,
    points2line,
    rotate_and_shift_coordination,
    deal_with_phi,
    line2points,
)
from environment.env_horizon.misc_ic import (
    L,
    W,
    VEHICLE_MODE_DICT,
    BIKE_MODE_DICT,
    PERSON_MODE_DICT,
    VEH_NUM,
    BIKE_NUM,
    PERSON_NUM,
    ROUTE_JUNCTION,
    turning_v_info,
    turning_b_info,
    turning_p_info,
    DETECT_RANGE,
    map_info,
    CROSS_TASKS,
    task2staticpath,
    MODE2TASK,
    GOAL_POINT_R,
)
from environment.env_horizon.traffic import Traffic

warnings.filterwarnings("ignore")

VEH_CLASSES = ["dl", "du", "dr", "rd", "rl", "ru", "ur", "ud", "ul", "lu", "lr", "ld"]
BIKE_CLASSES = ["du_b", "dr_b", "rl_b", "ru_b", "ud_b", "ul_b", "lr_b", "ld_b"]
PERSON_CLASSES = ["du_p", "rl_p", "ud_p", "lr_p"]


def convert_observation_to_space(observation):
    """[summary]

    Parameters
    ----------
    observation : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    NotImplementedError
        [description]
    """
    if isinstance(observation, dict):

        space = gym.spaces.Dict(
            OrderedDict([(key, convert_observation_to_space(value)) for key, value in observation.items()])
        )
    elif isinstance(observation, np.ndarray):
        obs_shape = (observation.shape[0] - 3,)
        low = np.full(obs_shape, -float("inf"))
        high = np.full(obs_shape, float("inf"))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class HorizonCrossingEnv(gym.Env):
    def __init__(
        self,
        training_task2=None,  # tuple (from_edge, to_edge)
        num_future_data=0,
        mode="training",
        multi_display=False,
        expected_v=None,
        **kwargs,
    ):
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.fill_num = None
        self.edge_areas = None
        self.edge_lines = None
        self.junction_areas = None
        """
        # task1 -- multi-lane task
        # task2 -- crossing (in edge, out edge)
        # task4 -- general (junction or edge, edge, in edge, out edge)
        """
        if training_task2 is None:
            self.task_mode = "random"
            self.training_task2 = self.random_select_task()
            # print(self.training_task2)
        else:
            self.task_mode = "fixed"
            self.training_task2 = training_task2

        self.from_edge, self.to_edge = self.training_task2
        self.turning_direction = ROUTE_JUNCTION[self.from_edge][1]

        self.ref_path = ReferencePath(training_task4=("junction", None, self.from_edge, self.to_edge),
                                      expected_v=expected_v)

        self._struct_road_area()

        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = num_future_data
        self.env_model = environment_model(
            training_task4=("junction", None, self.from_edge, self.to_edge),
            num_future_data=num_future_data, expected_v=expected_v
        )
        self.init_state = {}
        self.action_number = 2
        self.exp_v = expected_v
        assert self.exp_v is not None
        self.ego_l, self.ego_w = L, W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = self._reset_init_state()

        self.obs = None
        self.action = None
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.turning_direction]
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.turning_direction]
        self.person_mode_dict = PERSON_MODE_DICT[self.turning_direction]
        self.veh_num = VEH_NUM[self.turning_direction]
        self.bike_num = BIKE_NUM[self.turning_direction]
        self.person_num = PERSON_NUM[self.turning_direction]
        self.virtual_red_light_vehicle = False
        self.light_phase_list = CROSS_TASKS[self.from_edge]["tl"]
        self.done_type = "not_done_yet"
        self.reward_info = None
        self.ego_info_dim = 6
        self.per_veh_info_dim = 5
        self.per_bike_info_dim = 5
        self.per_person_info_dim = 5
        self.per_tracking_info_dim = 3
        self.mode = mode
        self._define_predefined_dicts()
        sumocfg_dir = os.path.join(
            os.path.dirname(__file__),
            "sumo_file/cfg/IC_{}.sumocfg".format(CROSS_TASKS[self.from_edge]["main_crossing"]),
        )

        print(f"sumocfg dir: {sumocfg_dir}")
        if not multi_display:
            self.traffic = Traffic(
                self.step_length,
                mode=self.mode,
                init_n_ego_dict=self.init_state,
                training_task=("junction", None, self.from_edge, self.to_edge),
                sumocfg_dir=sumocfg_dir,
            )
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)  # 创建observation space
            # plt.figure(figsize=(16.2, 7.8))
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def random_select_task():
        """随机选择一个task"""
        task_name = random.choice(list(CROSS_TASKS.keys()))
        from_edge = CROSS_TASKS[task_name]["from_edge"]
        to_edge = CROSS_TASKS[task_name]["to_edge"]
        return from_edge, to_edge

    def _define_predefined_dicts(self):
        self.predefined_veh_class_dict = {k: turning_v_info[self.from_edge][k]["lanes"] for k in VEH_CLASSES}
        self.predefined_bike_class_dict = {k: turning_b_info[self.from_edge][k]["lanes"] for k in BIKE_CLASSES}
        self.predefined_person_class_dict = {k: turning_p_info[self.from_edge][k]["lanes"] for k in PERSON_CLASSES}

    def reset(self, **kwargs):  # kwargs include three keys
        if self.task_mode == "random":
            self.training_task2 = self.random_select_task()
            self.from_edge, self.to_edge = self.training_task2

        self.ref_path = ReferencePath(
            training_task4=("junction", None, self.from_edge, self.to_edge),
            expected_v=self.exp_v, **kwargs)
        if self.task_mode == "random":
            self._struct_road_area()
            self.light_phase_list = CROSS_TASKS[self.from_edge]["tl"]

        self.init_state = self._reset_init_state()
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()
        ego_dynamics = self._get_ego_dynamics(
            [
                self.init_state["ego"]["v_x"],
                self.init_state["ego"]["v_y"],
                self.init_state["ego"]["r"],
                self.init_state["ego"]["x"],
                self.init_state["ego"]["y"],
                self.init_state["ego"]["phi"],
            ],
            [
                0,
                0,
                self.dynamics.vehicle_params["miu"],
                self.dynamics.vehicle_params["miu"],
            ],
        )

        self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.action = None
        self.reward_info = None
        self.done_type = "not_done_yet"

        plt.cla()

        if self.turning_direction == "right":
            self.virtual_red_light_vehicle = False
        elif self.mode == "training" or self.mode == "debug":
            if np.random.random() > 0.85:  # 10%的概率不管红绿灯是哪一个阶段，都放置红灯虚拟车
                self.virtual_red_light_vehicle = True
            else:
                self.virtual_red_light_vehicle = False
        else:
            self.virtual_red_light_vehicle = False
        return self.obs

    def close(self):
        """don't call __del__"""
        pass
        # try:
        #     del self.traffic
        # except Exception as e:
        #     print("----> error in delete traffic")
        #     print(e)

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self.compute_reward(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()
        all_info = self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({"final_rew": reward})
        all_info.update(
            {
                "reward_info": self.reward_info,
                "ref_index": self.ref_path.ref_index,
                "veh_num": self.veh_num,
                "bike_num": self.bike_num,
                "person_num": self.person_num,
                "fill_num_bikes": self.fill_num["bikes"],
                "fill_num_persons": self.fill_num["persons"],
                "fill_num_vehicles": self.fill_num["vehicles"],
            }
        )
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        """
        通过observation 设定 observation space
        """
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):
        out = dict(
            v_x=next_ego_state[0],
            v_y=next_ego_state[1],
            r=next_ego_state[2],
            x=next_ego_state[3],
            y=next_ego_state[4],
            phi=next_ego_state[5],
            l=self.ego_l,
            w=self.ego_w,
            alpha_f=next_ego_params[0],
            alpha_r=next_ego_params[1],
            miu_f=next_ego_params[2],
            miu_r=next_ego_params[3],
        )
        miu_f, miu_r = out["miu_f"], out["miu_r"]
        F_zf, F_zr = (
            self.dynamics.vehicle_params["F_zf"],
            self.dynamics.vehicle_params["F_zr"],
        )
        C_f, C_r = (
            self.dynamics.vehicle_params["C_f"],
            self.dynamics.vehicle_params["C_r"],
        )
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params["g"] / (abs(out["v_x"]) + 1e-8)  # 横摆角速度的上界

        l, w, x, y, phi = out["l"], out["w"], out["x"], out["y"], out["phi"]

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)

        corner_point = cal_corner_point_of_ego_car()
        out.update(
            dict(
                alpha_f_bound=alpha_f_bound,
                alpha_r_bound=alpha_r_bound,
                r_bound=r_bound,
                Corner_point=corner_point,
            )
        )
        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles["ego"]  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.v_light = self.traffic.v_light

        all_info = dict(
            all_vehicles=self.all_vehicles,
            ego_dynamics=self.ego_dynamics,
            v_light=self.v_light,
        )
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done
        """
        if self.traffic.collision_flag:
            return "collision", 1
        if self._break_road_constrain():
            return "break_road_constrain", 1
        elif self._deviate_too_much():
            return "deviate_too_much", 1
        elif self._break_stability():
            return "break_stability", 1
        elif self._break_red_light():
            return "break_red_light", 1
        elif self._is_achieve_goal():
            return "good_done", 1
        else:
            return "not_done_yet", 0

    def _deviate_too_much(self):
        delta_y, delta_phi, delta_v = self.obs[self.ego_info_dim : self.ego_info_dim + 3]
        return True if abs(delta_y) > 15 else False

    def _break_road_constrain(self):
        # NOTE: now all point is out of lane, return will be true
        ego_corner_point = self.ego_dynamics["Corner_point"]
        results = True
        for ra in self.edge_areas:
            path_area = Path(line2points(*ra))
            if any(path_area.contains_points(ego_corner_point)):
                results = False

        for ja in self.junction_areas:
            path_area = Path(line2points(*ja))
            if any(path_area.contains_points(ego_corner_point)):
                results = False

        return results

    def _break_stability(self):
        # alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
        #                                  self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        # alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics["r_bound"]
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        # TODO 使用 r_bound是否足够
        if -r_bound < self.ego_dynamics["r"] < r_bound:
            return False
        else:
            return True

    @staticmethod
    def _break_red_light():
        # TODO
        # return True if self.v_light != 0 and self.v_light != 1 and self.ego_dynamics['y'] >
        # -CROSSROAD_SIZE/2 and self.turning_direction != 'right' else False
        return False

    def _is_achieve_goal(self):
        x = self.ego_dynamics["x"]
        y = self.ego_dynamics["y"]

        xt, yt = self.ref_path.target_points

        _distance = np.square(x - xt) + np.square(y - yt)

        if _distance <= np.square(GOAL_POINT_R / 2):
            return True
        else:
            return False

    def _action_transformation_for_end2end(self, action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25 * a_x_norm - 0.75  # [-3, 1.5]
        # if self.v_light != 0 and self.ego_dynamics['y'] < -25 and self.training_task != 'right':
        #     scaled_steer = 0.
        #     scaled_a_x = -3.
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics["v_x"]
        current_v_y = self.ego_dynamics["v_y"]
        current_r = self.ego_dynamics["r"]
        current_x = self.ego_dynamics["x"]
        current_y = self.ego_dynamics["y"]
        current_phi = self.ego_dynamics["phi"]
        steer, a_x = trans_action
        state = np.array(
            [[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]],
            dtype=np.float32,
        )
        action = np.array([[steer, a_x]], dtype=np.float32)

        state = torch.Tensor(state)
        action = torch.Tensor(action)

        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)

        next_ego_state, next_ego_params = (
            next_ego_state.numpy()[0],
            next_ego_params.numpy()[0],
        )
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.0
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self, exit_="D"):
        """
        获得环境的状态 （绝对坐标）
        """
        ego_x = self.ego_dynamics["x"]
        ego_y = self.ego_dynamics["y"]
        ego_phi = self.ego_dynamics["phi"]
        ego_v_x = self.ego_dynamics["v_x"]
        """
        状态量 =（自车的状态，tracking error, 周车的状态）
        """
        vehs_vector = self._construct_veh_vector_short(exit_)
        ego_vector = self._construct_ego_vector_short()
        tracking_error = self.ref_path.tracking_error_vector(
            np.array([ego_x], dtype=np.float32),
            np.array([ego_y], dtype=np.float32),
            np.array([ego_phi], dtype=np.float32),
            np.array([ego_v_x], dtype=np.float32),
            self.num_future_data,
        ).numpy()[0]
        self.per_tracking_info_dim = 3

        vector = np.concatenate((ego_vector, tracking_error, vehs_vector), axis=0)
        # vector = self.convert_vehs_to_rela(vector)

        return vector

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics["v_x"]
        ego_v_y = self.ego_dynamics["v_y"]
        ego_r = self.ego_dynamics["r"]
        ego_x = self.ego_dynamics["x"]
        ego_y = self.ego_dynamics["y"]
        ego_phi = self.ego_dynamics["phi"]
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        self.ego_info_dim = 6
        return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector_short(self, exit_="D"):
        ego_x = self.ego_dynamics["x"]
        ego_y = self.ego_dynamics["y"]
        ego_phi = self.ego_dynamics["phi"]
        v_light = self.v_light
        vehs_vector = []

        def filter_interested_participants(vs, turning_direction):
            """
            对周车的状态进行划分，排序、选择，得到最后感兴趣的车辆
            规则： 周车的行驶路线（筛选会和自车有冲突），然后选择最近的，不足时使用虚拟车补足。
            rule-based attention
            """

            veh_class_dict = {k: [] for k in VEH_CLASSES}
            bike_class_dict = {k: [] for k in BIKE_CLASSES}
            person_class_dict = {k: [] for k in PERSON_CLASSES}

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num], 0
                else:
                    fill_number = num - len(sorted_list)
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list, fill_number

            def compute_pad_position():
                xp = ego_x - DETECT_RANGE * np.cos(np.deg2rad(ego_phi))
                yp = ego_y - DETECT_RANGE * np.sin(np.deg2rad(ego_phi))

                index, (xpp, ypp, phipp) = self.ref_path.find_closest_point4junction(xp, yp)
                return {"x": xpp, "y": ypp, "phi": phipp}

            for v in vs:
                if v["type"] in ["bicycle_1", "bicycle_2", "bicycle_3"]:
                    v.update(partici_type=0.0)
                    lane_b = v["lane"]

                    for bike_key in bike_class_dict.keys():
                        if lane_b in self.predefined_bike_class_dict[bike_key]:
                            bike_class_dict[bike_key].append(v)
                            break

                elif v["type"] == "DEFAULT_PEDTYPE":
                    v.update(partici_type=1.0)
                    lane_p = v["lane"]

                    for person_key in person_class_dict.keys():
                        if lane_p in self.predefined_person_class_dict[person_key]:
                            person_class_dict[person_key].append(v)
                            break

                else:
                    v.update(partici_type=2.0)
                    lane_v = v["lane"]
                    for veh_key in veh_class_dict.keys():
                        if lane_v in self.predefined_veh_class_dict[veh_key]:
                            veh_class_dict[veh_key].append(v)
                            break

            # fetch bicycle in range
            for bike_key in bike_class_dict.keys():
                # filtering
                bike_class_dict[bike_key] = list(
                    filter(
                        lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"]) <= np.square(DETECT_RANGE),
                        bike_class_dict[bike_key]
                    )
                )

                # sorting
                bike_class_dict[bike_key] = sorted(
                    bike_class_dict[bike_key],
                    key=lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"]),
                )

            pad_bvp = compute_pad_position()
            pad_bike_dict = dict(
                type="bicycle_1",
                x=pad_bvp["x"],
                y=pad_bvp["y"],
                v=0,
                phi=pad_bvp["phi"],
                w=0.48,
                l=2,
                route="padding",
                partici_type=0.0,
            )

            tmp_b = OrderedDict()
            fill_num_b = OrderedDict()
            for mode, num in BIKE_MODE_DICT[turning_direction].items():
                tmp_b[mode], fill_num_b[mode] = slice_or_fill(bike_class_dict[mode], pad_bike_dict, num)

            # fetch person in range

            for person_key in person_class_dict.keys():
                person_class_dict[person_key] = list(
                    filter(
                        lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"]) <= np.square(DETECT_RANGE),
                        person_class_dict[person_key]
                    )
                )

                person_class_dict[person_key] = sorted(
                    person_class_dict[person_key],
                    key=lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"]),
                )

            pad_person_dict = dict(
                type="DEFAULT_PEDTYPE",
                x=pad_bvp["x"],
                y=pad_bvp["y"],
                v=0,
                phi=pad_bvp["phi"],
                w=0.525,
                l=0.75,
                road="pad",
                partici_type=1.0,
            )

            tmp_p = OrderedDict()
            fill_num_p = OrderedDict()
            for mode, num in PERSON_MODE_DICT[turning_direction].items():
                tmp_p[mode], fill_num_p[mode] = slice_or_fill(person_class_dict[mode], pad_person_dict, num)

            # if self.turning_direction != 'right':
            # if (v_light >1 and ego_y < -CROSSROAD_SIZE/2) \
            #         or (self.virtual_red_light_vehicle and ego_y < -CROSSROAD_SIZE/2):

            def in_in_edge():
                results = False
                for ra in self.edge_in_area:
                    path_area = Path(line2points(*ra))
                    if path_area.contains_point((ego_x, ego_y)):
                        results = True
                return results

            ioo = in_in_edge()
            if (v_light not in self.light_phase_list and ioo) or (self.virtual_red_light_vehicle and ioo):
                if self.turning_direction == "left":
                    for possible_path in turning_v_info[self.from_edge]["dl"]["path"]:
                        vlv_x, vlv_y, vlv_a = map_info[self.from_edge][possible_path[0]]["vlv"]  # TODO 此处可能不通用
                        vlv_pad = dict(x=vlv_x, y=vlv_y, phi=vlv_a)
                        veh_class_dict["dl"].append(
                            dict(
                                type="car_1",
                                x=vlv_pad["x"],
                                y=vlv_pad["y"],
                                v=0.0,
                                phi=vlv_pad["phi"],
                                l=5,
                                w=2.5,
                                route=None,
                                partici_type=2.0,
                            )
                        )

                elif self.turning_direction == "straight":
                    for possible_path in turning_v_info[self.from_edge]["du"]["path"]:
                        vlv_x, vlv_y, vlv_a = map_info[self.from_edge][possible_path[0]]["vlv"]
                        vlv_pad = dict(x=vlv_x, y=vlv_y, phi=vlv_a)
                        veh_class_dict["du"].append(
                            dict(
                                type="car_1",
                                x=vlv_pad["x"],
                                y=vlv_pad["y"],
                                v=0.0,
                                phi=vlv_pad["phi"],
                                l=5,
                                w=2.5,
                                route=None,
                                partici_type=2.0,
                            )
                        )

                elif self.turning_direction == "right":
                    for possible_path in turning_v_info[self.from_edge]["dr"]["path"]:
                        vlv_x, vlv_y, vlv_a = map_info[self.from_edge][possible_path[0]]["vlv"]
                        vlv_pad = dict(x=vlv_x, y=vlv_y, phi=vlv_a)
                        veh_class_dict["dr"].append(
                            dict(
                                type="car_1",
                                x=vlv_pad["x"],
                                y=vlv_pad["y"],
                                v=0.0,
                                phi=vlv_pad["phi"],
                                l=5,
                                w=2.5,
                                route=None,
                                partici_type=2.0,
                            )
                        )

                else:
                    raise KeyError("Unkonwn turning direction: {}".format(self.turning_direction))

            # fetch veh in range
            for veh_key in veh_class_dict.keys():
                veh_class_dict[veh_key] = list(
                    filter(
                        lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"]) <= np.square(DETECT_RANGE),
                        veh_class_dict[veh_key],
                    )
                )

                veh_class_dict[veh_key] = sorted(
                    veh_class_dict[veh_key],
                    key=lambda v: np.square(ego_y - v["y"]) + np.square(ego_x - v["x"])
                )

            pad_veh_dict = dict(
                type="car_1",
                x=pad_bvp["x"],
                y=pad_bvp["y"],
                v=0,
                phi=pad_bvp["phi"],
                w=2.5,
                l=5,
                route="pad",
                partici_type=2.0,
            )

            tmp_v = OrderedDict()
            fill_num_v = OrderedDict()
            for mode, num in VEHICLE_MODE_DICT[turning_direction].items():
                tmp_v[mode], fill_num_v[mode] = slice_or_fill(veh_class_dict[mode], pad_veh_dict, num)

            tmp1 = dict(tmp_b, **tmp_p)
            tmp = dict(tmp1, **tmp_v)
            fill_num = dict(
                bikes=sum([fill_num_b[mode] for _, mode in enumerate(fill_num_b)]),
                persons=sum([fill_num_p[mode] for _, mode in enumerate(fill_num_p)]),
                vehicles=sum([fill_num_v[mode] for _, mode in enumerate(fill_num_v)]),
            )
            return tmp, fill_num

        list_of_interested_veh_dict = []
        self.interested_vehs, self.fill_num = filter_interested_participants(self.all_vehicles, self.turning_direction)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi, veh_partici_type = (
                veh["x"],
                veh["y"],
                veh["v"],
                veh["phi"],
                veh["partici_type"],
            )
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi, veh_partici_type])

        return np.array(vehs_vector, dtype=np.float32)

    @staticmethod
    def recover_orig_position_fn(transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # useless
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _reset_init_state(self):
        """自车初始化的状态

        Returns
        -------
        dict
            自车状态字典
        """
        end_index = self.ref_path.joint_idx[-2] - 60  # TODO general ?
        start_index = self.ref_path.joint_idx[-3] - 300

        start_index = max(start_index, 0)
        end_index = min(end_index, self.ref_path.joint_idx[-1])

        # print(start_index, end_index)
        if np.random.rand() > 0.5:
            random_index = np.random.randint(start_index, self.ref_path.joint_idx[-3])
        else:
            random_index = np.random.randint(self.ref_path.joint_idx[-3], end_index)

        x, y, phi = self.ref_path.indexs2points(random_index)

        # v = 7 + 6 * np.random.random()
        # if np.random.random() < 0.1:
        #     v = 0.0
        # else:
        #     v = 1.5 * self.exp_v * np.random.random()

        v = 1.0 * self.exp_v * np.random.random()

        routeID = "self"
        return dict(
            ego=dict(
                v_x=v,
                v_y=0,
                r=0,
                x=x.numpy(),
                y=y.numpy(),
                phi=phi.numpy(),
                l=self.ego_l,
                w=self.ego_w,
                routeID=routeID,
            )
        )

    def compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]

        # extract infos for each kind of participants
        start = 0
        end = start + self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data + 1)
        obses_ego = obses[:, start:end]
        start = end
        end = start + self.per_bike_info_dim * self.bike_num
        obses_bike = obses[:, start:end]
        start = end
        end = start + self.per_person_info_dim * self.person_num
        obses_person = obses[:, start:end]
        start = end
        end = start + self.per_veh_info_dim * self.veh_num
        obses_veh = obses[:, start:end]

        reward, _, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(
            obses_ego, obses_bike, obses_person, obses_veh, actions
        )
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def _struct_road_area(self):
        """
        求解出 路 的几何范围，用一组点框出来。
        十字路口使用的是sumo junction的shape，
        edge, 是使用edge的shape 结和lane宽度算出的范围
        """
        self.edge_areas = []
        self.edge_lines = []
        self.junction_areas = []

        self.edge_in_area = []

        def mixlist2list(inlist):
            outlist = []
            for l in inlist:
                if isinstance(l, list):
                    outlist.extend(l)
                else:
                    outlist.append(l)
            return outlist

        def edge2roadline(edgeID, io="in"):
            lane_list_whole = map_info[edgeID]["lane_list"]
            lane_list = [ll for ll in lane_list_whole if map_info[edgeID][ll]["type"] != "x"]
            road_line_list = []
            road_area_list = []

            def centerline2points(start_point, end_point, width=3.75, io="in"):
                A, B = start_point, end_point
                AB = (B[0] - A[0], B[1] - A[1])
                ABL = math.sqrt(AB[0] * AB[0] + AB[1] * AB[1])
                L = min(ABL, 40)
                if io == "in":
                    A = (B[0] - AB[0] / ABL * L, B[1] - AB[1] / ABL * L)
                elif io == "out":
                    B = (A[0] + AB[0] / ABL * L, A[1] + AB[1] / ABL * L)

                start_point, end_point = A, B
                A, B = start_point, end_point
                AB = (B[0] - A[0], B[1] - A[1])
                ABL = math.sqrt(AB[0] * AB[0] + AB[1] * AB[1])

                k = (AB[1] / ABL, -AB[0] / ABL)

                a1 = (A[0] + k[0] * width / 2, A[1] + k[1] * width / 2)
                a2 = (B[0] + k[0] * width / 2, B[1] + k[1] * width / 2)
                a3 = (A[0] - k[0] * width / 2, A[1] - k[1] * width / 2)
                a4 = (B[0] - k[0] * width / 2, B[1] - k[1] * width / 2)

                roadpoint1 = (a1, a2)
                roadpoint2 = (a3, a4)

                roadarea = (a1, a2, a4, a3)
                return roadpoint1, roadpoint2, roadarea

            for l in lane_list:
                lane_shape = map_info[edgeID][l]["shape"]
                start_point = lane_shape[0]
                end_point = lane_shape[1]
                width = map_info[edgeID][l]["lane_width"]
                roadpoint1, roadpoint2, roadarea = centerline2points(start_point, end_point, width, io)
                road_line_list.append(points2line(roadpoint1))
                road_line_list.append(points2line(roadpoint2))

                road_area_list.append(points2line(roadarea))
            return road_line_list, road_area_list

        # plot road lines
        route_list = task2staticpath[self.from_edge]
        ref_route = route_list[self.ref_path.ref_index]

        for e, l in ref_route[:-1]:
            junctionID = ROUTE_JUNCTION[e][0]
            junction_points = map_info[junctionID]["shape"]
            junction_outline = points2line(junction_points)

            self.junction_areas.append(junction_outline)

            io = ROUTE_JUNCTION[e]
            edge1 = io[3]["u"]["in"]
            edge2 = io[3]["d"]["in"]
            edge3 = io[3]["l"]["in"]
            edge4 = io[3]["r"]["in"]
            edge5 = io[3]["u"]["out"]
            edge6 = io[3]["d"]["out"]
            edge7 = io[3]["l"]["out"]
            edge8 = io[3]["r"]["out"]
            edge_list_in = mixlist2list([edge1, edge2, edge3, edge4])
            edge_list_out = mixlist2list([edge5, edge6, edge7, edge8])
            # edge_list_mix = edge_list_in + edge_list_out
            ego_list_in = mixlist2list([edge2])
            for ee in edge_list_in:
                if ee:
                    road_line_list, road_area_list = edge2roadline(ee, io="in")
                    for rl in road_line_list:
                        self.edge_lines.append(rl)
                    for ra in road_area_list:
                        self.edge_areas.append(ra)

            for ee in edge_list_out:
                if ee:
                    road_line_list, road_area_list = edge2roadline(ee, io="out")
                    for rl in road_line_list:
                        self.edge_lines.append(rl)
                    for ra in road_area_list:
                        self.edge_areas.append(ra)

            for ee in ego_list_in:
                if ee:
                    road_line_list, road_area_list = edge2roadline(ee, io="in")

                    for ra in road_area_list:
                        self.edge_in_area.append(ra)

    def render(self, mode="human", save_dir=None):
        light_line_width = 3
        dotted_line_style = "--"
        solid_line_style = "-"
        extension = 40

        def is_in_plot_area(x, y, ego_x, ego_y, tolerance=DETECT_RANGE + 10.0):
            _distance2 = np.square(x - ego_x) + np.square(y - ego_y)
            if _distance2 <= np.square(tolerance):
                return True
            else:
                return False

        def draw_rotate_rec(ax, x, y, a, l, w, color, linestyle="-"):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            ax.plot(
                [RU_x + x, RD_x + x],
                [RU_y + y, RD_y + y],
                color=color,
                linestyle=linestyle,
            )
            ax.plot(
                [RU_x + x, LU_x + x],
                [RU_y + y, LU_y + y],
                color=color,
                linestyle=linestyle,
            )
            ax.plot(
                [LD_x + x, RD_x + x],
                [LD_y + y, RD_y + y],
                color=color,
                linestyle=linestyle,
            )
            ax.plot(
                [LD_x + x, LU_x + x],
                [LD_y + y, LU_y + y],
                color=color,
                linestyle=linestyle,
            )

        def plot_phi_line(type, x, y, phi, color):
            if type in ["bicycle_1", "bicycle_2", "bicycle_3"]:
                line_length = 2
            elif type == "DEFAULT_PEDTYPE":
                line_length = 1
            else:
                line_length = 5
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.0), y + line_length * sin(phi * pi / 180.0)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        def ploter(render_mode="human", save_dir=None):
            plt.cla()

            # ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            ax = plt.gca()
            fig = plt.gcf()
            fig.set_figheight(7.8)
            fig.set_figwidth(16.2)
            plt.axis("off")
            ax.axis("equal")

            # plot road lines

            for rl in self.edge_lines:
                plt.plot(rl[0], rl[1], "k")

            for ra in self.edge_areas:
                plt.fill(*ra, "lemonchiffon")

            for ja in self.junction_areas:
                plt.fill(*ja, "lightcyan")

            # plot vehicles
            ego_x = self.ego_dynamics["x"]
            ego_y = self.ego_dynamics["y"]

            for veh in self.all_vehicles:
                veh_x = veh["x"]
                veh_y = veh["y"]
                veh_phi = veh["phi"]
                veh_l = veh["l"]
                veh_w = veh["w"]
                veh_type = veh["type"]
                if veh_type in ["bicycle_1", "bicycle_2", "bicycle_3"]:
                    veh_color = "navy"
                elif veh_type == "DEFAULT_PEDTYPE":
                    veh_color = "purple"
                else:
                    veh_color = "black"
                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, veh_color)
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)

            # plot_interested vehs
            for mode, num in self.veh_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    veh_x = veh["x"]
                    veh_y = veh["y"]
                    veh_phi = veh["phi"]
                    veh_l = veh["l"]
                    veh_w = veh["w"]
                    veh_type = veh["type"]
                    # print("车辆信息", veh)
                    # veh_type = 'car_1'
                    task2color = {"left": "b", "straight": "c", "right": "m"}

                    if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")

            # plot_interested bicycle
            for mode, num in self.bicycle_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    veh_x = veh["x"]
                    veh_y = veh["y"]
                    veh_phi = veh["phi"]
                    veh_l = veh["l"]
                    veh_w = veh["w"]
                    veh_type = veh["type"]
                    # print("车辆信息", veh)
                    # veh_type = 'bicycle_1'
                    task2color = {"left": "b", "straight": "c", "right": "m"}

                    if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")

            # plot_interested person
            for mode, num in self.person_mode_dict.items():
                for i in range(num):
                    veh = self.interested_vehs[mode][i]
                    veh_x = veh["x"]
                    veh_y = veh["y"]
                    veh_phi = veh["phi"]
                    veh_l = veh["l"]
                    veh_w = veh["w"]
                    veh_type = veh["type"]

                    # print("车辆信息", veh)
                    # veh_type = 'bicycle_1'
                    task2color = {"left": "b", "straight": "c", "right": "m"}

                    if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                        plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                        task = MODE2TASK[mode]
                        color = task2color[task]
                        draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")

            # plot ego vehicle
            ego_v_x = self.ego_dynamics["v_x"]
            ego_v_y = self.ego_dynamics["v_y"]
            ego_r = self.ego_dynamics["r"]
            ego_x = self.ego_dynamics["x"]
            ego_y = self.ego_dynamics["y"]
            ego_phi = self.ego_dynamics["phi"]
            ego_l = self.ego_dynamics["l"]
            ego_w = self.ego_dynamics["w"]
            ego_alpha_f = self.ego_dynamics["alpha_f"]
            ego_alpha_r = self.ego_dynamics["alpha_r"]
            alpha_f_bound = self.ego_dynamics["alpha_f_bound"]
            alpha_r_bound = self.ego_dynamics["alpha_r_bound"]
            r_bound = self.ego_dynamics["r_bound"]

            plot_phi_line("self_car", ego_x, ego_y, ego_phi, "red")
            draw_rotate_rec(ax, ego_x, ego_y, ego_phi, ego_l, ego_w, "red")

            # plot future data (static path)
            tracking_info = self.obs[
                self.ego_info_dim : self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data + 1)
            ]
            future_path = tracking_info[self.per_tracking_info_dim :]
            for i in range(self.num_future_data):
                delta_x, delta_y, delta_phi = future_path[
                    i * self.per_tracking_info_dim : (i + 1) * self.per_tracking_info_dim
                ]
                path_x, path_y, path_phi = (
                    ego_x + delta_x,
                    ego_y + delta_y,
                    ego_phi - delta_phi,
                )
                plt.plot(path_x, path_y, "g.")
                plot_phi_line("self_car", path_x, path_y, path_phi, "g")

            delta_, _, _ = tracking_info[:3]
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color="g")

            # ax.scatter(self.ref_path.path[0][400], self.ref_path.path[1][400], color = 'k', s=100)
            # ax.scatter(self.ref_path.path[0][1020], self.ref_path.path[1][1020],color = 'k', s=100)
            # print(self.ref_path.path[0][400], self.ref_path.path[1][400])
            # print(self.ref_path.path[0][1020], self.ref_path.path[1][1020])
            indexs, points = self.ref_path.find_closest_point(
                np.array([ego_x], np.float32), np.array([ego_y], np.float32)
            )

            path_x = points[0][0].numpy()
            path_y = points[1][0].numpy()
            path_phi = points[2][0].numpy()

            plt.plot(path_x, path_y, "g.")

            delta_x = ego_x - path_x
            delta_y = ego_y - path_y
            delta_phi = ego_phi - path_phi

            if self.action is not None:
                steer, a_x = self.action[0], self.action[1]
                steer_string = r"steer: {:.2f}rad (${:.2f}\degree$)".format(steer, steer * 180 / np.pi)
                a_x_string = "a_x: {:.2f}m/s^2".format(a_x)
            else:
                steer_string = r"steer: [N/A]rad ($[N/A]\degree$)"
                a_x_string = "a_x: [N/A] m/s^2"

            if self.reward_info is not None:
                reward_string_list = []
                reward_color_list = []
                for key, val in self.reward_info.items():
                    reward_string_list.append("{}: {:.4f}".format(key, val))
                    reward_color_list.append("lavenderblush")
            else:
                reward_string_list = []
                reward_color_list = []

            text_data_list = [
                "ego_x: {:.2f}m".format(ego_x),
                "ego_y: {:.2f}m".format(ego_y),
                "path_x: {:.2f}m".format(path_x),
                "path_y: {:.2f}m".format(path_y),
                "delta_: {:.2f}m".format(delta_),
                "delta_x: {:.2f}m".format(delta_x),
                "delta_y: {:.2f}m".format(delta_y),
                r"ego_phi: ${:.2f}\degree$".format(ego_phi),
                r"path_phi: ${:.2f}\degree$".format(path_phi),
                r"delta_phi: ${:.2f}\degree$".format(delta_phi),
                "v_x: {:.2f}m/s".format(ego_v_x),
                "exp_v: {:.2f}m/s".format(self.exp_v),
                "delta_v: {:.2f}m/s".format(ego_v_x - self.exp_v),
                "v_y: {:.2f}m/s".format(ego_v_y),
                "yaw_rate: {:.2f}rad/s".format(ego_r),
                "yaw_rate bound: [{:.2f}, {:.2f}]".format(-r_bound, r_bound),
                r"$\alpha_f$: {:.2f} rad".format(ego_alpha_f),
                r"$\alpha_f$ bound: [{:.2f}, {:.2f}] ".format(-alpha_f_bound, alpha_f_bound),
                r"$\alpha_r$: {:.2f} rad".format(ego_alpha_r),
                r"$\alpha_r$ bound: [{:.2f}, {:.2f}] ".format(-alpha_r_bound, alpha_r_bound),
                steer_string,
                a_x_string,
                "done info: {}".format(self.done_type),
            ]

            text_color_list = (
                ["ivory"] * 4
                + ["lightcyan"] * 3
                + ["ivory"] * 2
                + ["lightcyan"] * 1
                + ["ivory"] * 2
                + ["lightcyan"] * 1
                + ["ivory"] * 7
                + ["salmon"] * 2
                + ["palegreen"]
            )

            text_data_list.extend(reward_string_list)
            text_color_list.extend(reward_color_list)

            def list_chunk_pad(input_list, num_col, pad_info=""):
                num_row, mod = len(input_list) // num_col, len(input_list) % num_col

                if mod > 0:
                    num_row = num_row + 1

                pad_size = num_row * num_col - len(input_list)

                if pad_size > 0:
                    pad_list = [pad_info] * pad_size
                    input_list.extend(pad_list)

                start = 0
                end = num_col
                muli_row_list = []
                for i in range(num_row):
                    muli_row_list.append(input_list[start:end])
                    start += num_col
                    end += num_col
                return muli_row_list

            cell_text = list_chunk_pad(text_data_list, 8)
            cell_color = list_chunk_pad(text_color_list, 8, "white")

            info_table = plt.table(
                cellText=cell_text,
                cellColours=cell_color,
                colWidths=[0.15] * 40,
            )
            # info_table.auto_set_font_size(False)
            # info_table.set_fontsize(10)

            if True:
                # perdiction traj of vehicle
                obses = self.obs[np.newaxis, :]

                veh_l, bik_l, per_l = [], [], []

                # extract infos for each kind of participants
                start = 0
                end = start + self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data + 1)
                obses_ego = torch.as_tensor(obses[:, start:end])
                start = end
                end = start + self.per_bike_info_dim * self.bike_num
                obses_bike = torch.as_tensor(obses[:, start:end])
                start = end
                end = start + self.per_person_info_dim * self.person_num
                obses_person = torch.as_tensor(obses[:, start:end])
                start = end
                end = start + self.per_veh_info_dim * self.veh_num
                obses_veh = torch.as_tensor(obses[:, start:end])

                for _ in range(25):
                    obses_veh = self.env_model.veh_predict(obses_veh)
                    obses_person = self.env_model.person_predict(obses_person)
                    obses_bike = self.env_model.bike_predict(obses_bike)
                    veh_l.append(obses_veh)
                    per_l.append(obses_person)
                    bik_l.append(obses_bike)

                veh_p = torch.cat(veh_l).numpy()
                bik_p = torch.cat(bik_l).numpy()
                per_p = torch.cat(per_l).numpy()

                for ite in range(veh_p.shape[1] // 5):
                    xx = veh_p[:, ite * 5]
                    yy = veh_p[:, ite * 5 + 1]
                    ax.plot(xx, yy, "r")

                for ite in range(bik_p.shape[1] // 5):
                    xx = bik_p[:, ite * 5]
                    yy = bik_p[:, ite * 5 + 1]
                    ax.plot(xx, yy, "r")

                for ite in range(per_p.shape[1] // 5):
                    xx = per_p[:, ite * 5]
                    yy = per_p[:, ite * 5 + 1]
                    ax.plot(xx, yy, "r")
            plt.tight_layout()

            if render_mode == "human":
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.show()
                plt.pause(0.001)
                return None
            elif render_mode == "save":
                """
                plt.show()
                plt.savefig(save_dir, dpi=500)
                plt.pause(0.001)
                """
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                fig.canvas.draw()
                img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # img = PIL.Image.fromarray(image_from_plot, "RGB")
                img.save(save_dir)
                return None
            elif render_mode == "rgb_array":
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                return image_from_plot

        return ploter(render_mode=mode, save_dir=save_dir)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


# def write_video(frames, title, path=''):
#     if len(frames[0].shape) == 3:
#         frames = np.stack(frames, axis=0)[:, :, :,
#                  ::-1]  # VideoWrite expects H x W x C in BGR
#     elif len(frames[0].shape) == 4:
#         frames = np.concatenate(frames, axis=0).astype(np.uint8)[:,
#                  :, :,
#                  ::-1]  # VideoWrite expects H x W x C in BGR
#     else:
#         print("error")
#
#     _, H, W, _ = frames.shape
#
#     writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 10., (W, H), True)
#     for frame in frames:
#         writer.write(frame)
#     writer.release()


def t_crossing():
    env = HorizonCrossingEnv(training_task2=("EE6", "ES8"), num_future_data=0)
    # env = HorizonCrossingEnv(training_task2=['EN4', 'EN3', 'EE2'], num_future_data=0)

    obs = env.reset()

    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            # action=2*np.random.random(2)-1
            action = env.action_space.sample()
            action = np.array([0.0, 0.3])
            obs, reward, done, info = env.step(action)
            print(obs.shape)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # extract infos for each kind of participants
            start = 0
            end = start + env.ego_info_dim + env.per_tracking_info_dim * (env.num_future_data + 1)
            obses_ego = obses[:, start:end]
            print(obses_ego)

            env.render()

            if done:
                env.reset()
    env.close()


def t_env_time():
    """
    render only FPS = 1.2255063786037046 FPS
    render and save  0.36218579088184016 FPS
    withour render 31.583721480983932 FPS

    # 71.30774831059723
    """
    import time

    env = HorizonCrossingEnv(training_task2=("EE6", "ES8"), num_future_data=0)
    env.reset()
    START = time.time()
    # frames = []
    for i in range(1000):
        a = env.action_space.sample()
        s, r, d, _ = env.step(a)
        # img = env.render(mode="save", save_dir="./pp_{}.png".format(i))
        # frames.append(img)
        if d:
            env.reset()
        #

    # write_video(frames, "test", )
    END = time.time()
    print(f"FPS = {1000 / (END - START)}")


if __name__ == "__main__":
    # t_end2end()

    t_crossing()
