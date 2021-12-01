#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/08
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: crossing.py
# =====================================
import os
import warnings
import PIL
import math
from collections import OrderedDict
from math import cos, sin, pi

import pprint
import gym
import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import logical_and
import torch
from gym.utils import seeding
from matplotlib.path import Path

# gym.envs.user_defined.toyota_env.
from environment.env_horizon.dynamics_and_models import VehicleDynamics, ReferencePath, environment_model
from environment.env_horizon.endtoend_env_utils import shift_coordination, rotate_coordination, rotate_and_shift_coordination, deal_with_phi, \
     wrapped_point_dynamics2crossing, line2points, points2line, coordi_dynamics2crossing, get_edge_info_ml
from environment.env_horizon.misc_ic import L, W, VEHICLE_MODE_DICT, BIKE_MODE_DICT, PERSON_MODE_DICT, \
    ML_PERSON_NUM, ML_VEH_NUM, ML_BIKE_NUM, EGO_ROUTE, \
    RANDOM_V_X, RANDOM_V_Y, RANDOM_R, RANDOM_X, RANDOM_Y, RANDOM_PHI, \
    DETECT_RANGE, map_info, GOAL_POINT_R, MLANE_TASKS, VEH_NUM, BIKE_NUM, PERSON_NUM
from environment.env_horizon.traffic import Traffic

warnings.filterwarnings("ignore")


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = gym.spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        obs_shape = observation.shape[0] - 3
        low = np.full((obs_shape, ), -float('inf'))
        high = np.full((obs_shape, ), float('inf'))
        space = gym.spaces.Box(low, high, dtype=np.float32)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class HorizonMultiLaneEnv(gym.Env):
    def __init__(self,
                 training_task=None,  # edge, useless
                 num_future_data=0,
                 mode='training',
                 multi_display=False,
                 expected_v=None,
                 **kwargs):
        self.dynamics = VehicleDynamics()
        self.interested_vehs = None
        self.fill_num = None

        self.edge_areas = None
        self.edge_lines = None

        self.area_right_position = None
        self.lane_num = None

        self.adj_ref_mode = kwargs['adj_ref_mode']
        self.env_edge = kwargs['env_edge']
        self.training_task1 = self.env_edge
        if self.adj_ref_mode == 'random':
            self.prob_init_adj = 0.5
        else:
            self.prob_init_adj = 0.0
        # if training_task is None:
        #     # task1: multi-lane's task
        #     self.task_mode = 'random'
        #     self.training_task1 = self.random_select_task()
        # else:
        #     self.task_mode = 'fixed'
        #     self.training_task1 = training_task  # 'EE2'

        self.current_edge = self.training_task1
        self.from_edge = self.current_edge
        self.turning_direction = 'straight'

        _, self.area_right_position, self.lane_num = get_edge_info_ml(self.current_edge)

        self.ref_path = ReferencePath(training_task4=('edge', self.current_edge, self.from_edge, None),
                                      expected_v=expected_v)
        if self.adj_ref_mode == 'random':
            adj_ref_index = np.random.choice(len(self.ref_path.path_list))
            while adj_ref_index == self.ref_path.ref_index:
                adj_ref_index = np.random.choice(len(self.ref_path.path_list))
            self.ref_path.adjacent_path = self.ref_path.path_list[adj_ref_index]

        self._struct_edge_area()

        self.detected_vehicles = None
        self.all_vehicles = None
        self.ego_dynamics = None
        self.num_future_data = num_future_data
        self.env_model = environment_model(('edge', self.current_edge, self.from_edge, None), num_future_data, expected_v=expected_v)
        self.init_state = {}
        self.action_number = 2
        self.exp_v = expected_v
        self.ego_l, self.ego_w = L, W
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_number,), dtype=np.float32)

        self.seed()
        self.v_light = None
        self.step_length = 100  # ms

        self.step_time = self.step_length / 1000.0
        self.init_state = self._reset_init_state()

        self.obs = None
        self.action = None
        self.veh_mode_dict = VEHICLE_MODE_DICT[self.turning_direction]  # TODO: attention之后修改
        self.bicycle_mode_dict = BIKE_MODE_DICT[self.turning_direction]
        self.person_mode_dict = PERSON_MODE_DICT[self.turning_direction]
        self.veh_num = VEH_NUM[self.turning_direction]
        self.bike_num = BIKE_NUM[self.turning_direction]
        self.person_num = PERSON_NUM[self.turning_direction]
        self.virtual_red_light_vehicle = False

        self.done_type = 'not_done_yet'
        self.reward_info = None
        self.ego_info_dim = 6
        self.per_veh_info_dim = 5
        self.per_bike_info_dim = 5
        self.per_person_info_dim = 5
        self.per_tracking_info_dim = 3
        self.mode = mode
        sumocfg_dir = os.path.join(os.path.dirname(__file__),
                                   "sumo_file/cfg/IC_{}.sumocfg".format(self.training_task1))

        print(f"sumocfg dir: {sumocfg_dir}")
        if not multi_display:
            self.traffic = Traffic(self.step_length,
                                   mode=self.mode,
                                   init_n_ego_dict=self.init_state,
                                   training_task=('edge', self.current_edge, self.from_edge, None),
                                   sumocfg_dir=sumocfg_dir
                                   )
            self.reset()
            action = self.action_space.sample()
            observation, _reward, done, _info = self.step(action)
            self._set_observation_space(observation)  # 观察observation的空间
            plt.ion()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @ staticmethod
    def random_select_task():
        task_name = random.choice(list(MLANE_TASKS.keys()))
        return task_name

    def reset(self, **kwargs):  # kwargs include three keys
        # if self.task_mode == 'random':
        #     self.training_task1 = self.random_select_task()
        #     self.current_edge = self.training_task1
        #     self.from_edge = self.current_edge

        _, area_right_position, self.lane_num = get_edge_info_ml(self.current_edge)
        self.ref_path = ReferencePath(training_task4=('edge', self.current_edge, self.from_edge, None),
                                      expected_v=self.exp_v
                                      **kwargs)
        if self.adj_ref_mode == 'random':
            adj_ref_index = np.random.choice(len(self.ref_path.path_list))
            while adj_ref_index == self.ref_path.ref_index:
                adj_ref_index = np.random.choice(len(self.ref_path.path_list))
            self.ref_path.adjacent_path = self.ref_path.path_list[adj_ref_index]

        # self.traffic.reset_rs_param(self.rs_param)  # 坐标变换
        self.init_state = self._reset_init_state()  # 初始化自车
        self.traffic.init_traffic(self.init_state)
        self.traffic.sim_step()  # SUMO走一步，才真正把自车放进去
        ego_dynamics = self._get_ego_dynamics([self.init_state['ego']['v_x'],
                                               self.init_state['ego']['v_y'],
                                               self.init_state['ego']['r'],
                                               self.init_state['ego']['x'],
                                               self.init_state['ego']['y'],
                                               self.init_state['ego']['phi']],
                                              [0,
                                               0,
                                               self.dynamics.vehicle_params['miu'],
                                               self.dynamics.vehicle_params['miu']]
                                              )

        self._get_all_info(ego_dynamics)  # 获取所有车辆的信息
        self.obs = self._get_obs()  # 获取obs
        self.action = None
        self.reward_info = None
        self.done_type = 'not_done_yet'

        plt.cla()  # reset清理图片

        if self.mode == 'training':
            if np.random.random() > 0.9:
                self.virtual_red_light_vehicle = True
            else:
                self.virtual_red_light_vehicle = False
        else:
            self.virtual_red_light_vehicle = False
        return self.obs

    def close(self):
        del self.traffic

    def step(self, action):
        self.action = self._action_transformation_for_end2end(action)
        reward, self.reward_info = self.compute_reward(self.obs, self.action)
        next_ego_state, next_ego_params = self._get_next_ego_state(self.action)
        ego_dynamics = self._get_ego_dynamics(next_ego_state, next_ego_params)
        self.traffic.set_own_car(dict(ego=ego_dynamics))
        self.traffic.sim_step()  # 自车状态更新生效，周车向前走一步
        all_info = self._get_all_info(ego_dynamics)
        self.obs = self._get_obs()
        self.done_type, done = self._judge_done()
        self.reward_info.update({'final_rew': reward})
        all_info.update({'reward_info': self.reward_info, 'ref_index': self.ref_path.ref_index,
                         'veh_num': self.veh_num, 'bike_num': self.bike_num, 'person_num': self.person_num,
                         'fill_num_bikes': self.fill_num['bikes'],
                         'fill_num_persons': self.fill_num['persons'],
                         'fill_num_vehicles': self.fill_num['vehicles']})
        # self.traffic.ttt
        # print('ttt', self.traffic.ttt, type(self.traffic.ttt), self.obs[[3, 4]])
        return self.obs, reward, done, all_info

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_ego_dynamics(self, next_ego_state, next_ego_params):

        out = dict(v_x=next_ego_state[0],
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
                   miu_r=next_ego_params[3],)
        miu_f, miu_r = out['miu_f'], out['miu_r']
        F_zf, F_zr = self.dynamics.vehicle_params['F_zf'], self.dynamics.vehicle_params['F_zr']
        C_f, C_r = self.dynamics.vehicle_params['C_f'], self.dynamics.vehicle_params['C_r']
        alpha_f_bound, alpha_r_bound = 3 * miu_f * F_zf / C_f, 3 * miu_r * F_zr / C_r
        r_bound = miu_r * self.dynamics.vehicle_params['g'] / (abs(out['v_x'])+1e-8)  # 横摆角速度的限制

        l, w, x, y, phi = out['l'], out['w'], out['x'], out['y'], out['phi']

        def cal_corner_point_of_ego_car():
            x0, y0, a0 = rotate_and_shift_coordination(l / 2, w / 2, 0, -x, -y, -phi)
            x1, y1, a1 = rotate_and_shift_coordination(l / 2, -w / 2, 0, -x, -y, -phi)
            x2, y2, a2 = rotate_and_shift_coordination(-l / 2, w / 2, 0, -x, -y, -phi)
            x3, y3, a3 = rotate_and_shift_coordination(-l / 2, -w / 2, 0, -x, -y, -phi)
            return (x0, y0), (x1, y1), (x2, y2), (x3, y3)
        Corner_point = cal_corner_point_of_ego_car()
        out.update(dict(alpha_f_bound=alpha_f_bound,
                        alpha_r_bound=alpha_r_bound,
                        r_bound=r_bound,
                        Corner_point=Corner_point))
        return out

    def _get_all_info(self, ego_dynamics):  # used to update info, must be called every timestep before _get_obs
        # to fetch info
        self.all_vehicles = self.traffic.n_ego_vehicles['ego']  # coordination 2
        self.ego_dynamics = ego_dynamics  # coordination 2
        self.v_light = self.traffic.v_light

        # all_vehicles
        # dict(x=x, y=y, v=v, phi=a, l=length,
        #      w=width, route=route)

        all_info = dict(all_vehicles=self.all_vehicles,
                        ego_dynamics=self.ego_dynamics,
                        v_light=self.v_light)
        return all_info

    def _judge_done(self):
        """
        :return:
         1: bad done: collision
         2: bad done: break_road_constrain
         3: good done: task succeed
         4: not done or end of an episode
        """
        if self.traffic.collision_flag:
            return 'collision', 1
        if self._break_road_constrain():
            return 'break_road_constrain', 1
        elif self._deviate_too_much():
            return 'deviate_too_much', 1
        elif self._break_stability():
            return 'break_stability', 1
        elif self._break_red_light():
            return 'break_red_light', 1
        elif self._is_achieve_goal():
            return 'good_done', 1
        elif self._near_end_of_road():
            return 'end_of_an_episode', 1
        else:
            return 'not_done_yet', 0

    def _deviate_too_much(self):
        delta_y, delta_phi, delta_v = self.obs[self.ego_info_dim:self.ego_info_dim+3]
        return True if abs(delta_y) > 15 else False

    def _break_road_constrain(self):
        # TODO now all point is out of lane, return will be true
        ego_corner_point = self.ego_dynamics['Corner_point']
        results = True
        for ra in self.edge_areas:
            path_area = Path(line2points(*ra))
            if any(path_area.contains_points(ego_corner_point)):
                results = False

        return results

    def _break_stability(self):
        alpha_f, alpha_r, miu_f, miu_r = self.ego_dynamics['alpha_f'], self.ego_dynamics['alpha_r'], \
                                         self.ego_dynamics['miu_f'], self.ego_dynamics['miu_r']
        alpha_f_bound, alpha_r_bound = self.ego_dynamics['alpha_f_bound'], self.ego_dynamics['alpha_r_bound']
        r_bound = self.ego_dynamics['r_bound']
        # if -alpha_f_bound < alpha_f < alpha_f_bound \
        #         and -alpha_r_bound < alpha_r < alpha_r_bound and \
        #         -r_bound < self.ego_dynamics['r'] < r_bound:
        if -r_bound < self.ego_dynamics['r'] < r_bound:
            return False
        else:
            return True

    def _break_red_light(self):
        # return True if self.v_light != 0 and self.v_light != 1 and self.ego_dynamics['y'] > -CROSSROAD_SIZE/2 and self.turning_direction != 'right' else False
        return False

    def _is_achieve_goal(self):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']  # °

        xt, yt, _ = self.ref_path.target_points
        _distance2end = np.sqrt(np.square(ego_x - xt) + np.square(ego_y - yt))

        _, points = self.ref_path.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
        path_x, path_y, path_phi = points[0][0].item(), points[1][0].item(), points[2][0].item()
        _distance2path = np.sqrt(np.square(ego_x - path_x) + np.square(ego_y - path_y))
        _phi2path = ego_phi - path_phi
        if _phi2path > math.pi:
            _phi2path = _phi2path - math.pi
        if _phi2path < -math.pi:
            _phi2path = _phi2path + math.pi
        _phi2path = abs(_phi2path)

        if _distance2end <= GOAL_POINT_R * 3 and _distance2path <= (3.5 - W) / 2 and _phi2path <= 15:
            return True
        else:
            return False

    def _near_end_of_road(self):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']

        xt, yt, phi_path = self.ref_path.target_points
        # _distance2end = np.sqrt(np.square(ego_x - xt) + np.square(ego_y - yt))
        _distance2end_along_path = abs((xt - ego_x) * cos(phi_path / 180 * pi)
                                       + (yt - ego_y) * sin(phi_path / 180 * pi))

        if _distance2end_along_path <= 12:
            return True

    @staticmethod
    def _action_transformation_for_end2end(action):  # [-1, 1]
        action = np.clip(action, -1.05, 1.05)
        steer_norm, a_x_norm = action[0], action[1]
        scaled_steer = 0.4 * steer_norm
        scaled_a_x = 2.25*a_x_norm - 0.75  # [-3, 1.5]
        # if self.v_light != 0 and self.ego_dynamics['y'] < -25 and self.training_task != 'right':
        #     scaled_steer = 0.
        #     scaled_a_x = -3.
        scaled_action = np.array([scaled_steer, scaled_a_x], dtype=np.float32)
        return scaled_action

    def _get_next_ego_state(self, trans_action):
        current_v_x = self.ego_dynamics['v_x']
        current_v_y = self.ego_dynamics['v_y']
        current_r = self.ego_dynamics['r']
        current_x = self.ego_dynamics['x']
        current_y = self.ego_dynamics['y']
        current_phi = self.ego_dynamics['phi']
        steer, a_x = trans_action
        state = np.array([[current_v_x, current_v_y, current_r, current_x, current_y, current_phi]], dtype=np.float32)
        action = np.array([[steer, a_x]], dtype=np.float32)

        state = torch.Tensor(state)
        action = torch.Tensor(action)

        next_ego_state, next_ego_params = self.dynamics.prediction(state, action, 10)  # params: 侧偏角，附着系数

        next_ego_state, next_ego_params = next_ego_state.numpy()[0],  next_ego_params.numpy()[0]
        next_ego_state[0] = next_ego_state[0] if next_ego_state[0] >= 0 else 0.  # v_x >= 0
        next_ego_state[-1] = deal_with_phi(next_ego_state[-1])
        return next_ego_state, next_ego_params

    def _get_obs(self):  # concatenate ego, path_info and other veh
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']

        vehs_vector = self._construct_veh_vector_short()
        ego_vector = self._construct_ego_vector_short()
        path_info = self.ref_path.find_closest_point4edge(np.array([ego_x], dtype=np.float32),
                                                          np.array([ego_y], dtype=np.float32)).numpy()[0]
        self.per_tracking_info_dim = 3

        vector = np.concatenate((ego_vector, path_info, vehs_vector), axis=0)

        return vector

    def _construct_ego_vector_short(self):
        ego_v_x = self.ego_dynamics['v_x']
        ego_v_y = self.ego_dynamics['v_y']
        ego_r = self.ego_dynamics['r']
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_phi = self.ego_dynamics['phi']
        ego_feature = [ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi]
        self.ego_info_dim = 6
        return np.array(ego_feature, dtype=np.float32)

    def _construct_veh_vector_short(self):
        ego_x = self.ego_dynamics['x']
        ego_y = self.ego_dynamics['y']
        ego_v = self.ego_dynamics['v_x']
        path_info = self.ref_path.find_closest_point4edge(np.array([ego_x], dtype=np.float32),
                                                          np.array([ego_y], dtype=np.float32)).numpy()[0]
        closest_x_path = path_info[0]
        closest_y_path = path_info[1]
        phi_path = path_info[2]
        # print(f'ego_x = {ego_x}, ego_y = {ego_y}, phi_path = {phi_path}')
        vehs_vector = []

        def filter_interested_participants(vs, turning_direction):
            """
            从所有车辆中选取感兴趣的车辆：按照距离自车近远距离
            """
            detect_range = 30.
            bikes = []
            persons = []
            vehicles = []

            # slice or fill to some number
            def slice_or_fill(sorted_list, fill_value, num):
                if len(sorted_list) >= num:
                    return sorted_list[:num], 0
                else:
                    fill_number = num - len(sorted_list)
                    while len(sorted_list) < num:
                        sorted_list.append(fill_value)
                    return sorted_list, fill_number

            for v in vs:
                if v['type'] in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    v.update(partici_type=0.0)
                    bikes.append(v)

                elif v['type'] == 'DEFAULT_PEDTYPE':
                    v.update(partici_type=1.0)
                    persons.append(v)

                else:
                    v.update(partici_type=2.0)
                    vehicles.append(v)

            # 1. fetch and sort bicycles in range
            bikes = list(filter(lambda v:
                                np.square(ego_y - v['y']) + np.square(ego_x - v['x']) <= np.square(detect_range) and
                                abs(deal_with_phi(phi_path - v['phi'])) <= 135,
                                bikes))
            bikes = sorted(bikes, key=lambda v: np.square(ego_y - v['y']) + np.square(ego_x - v['x']))

            bikes_pad_x = closest_x_path - detect_range * np.cos(phi_path / 180 * pi)  # Note: PI_DSAC put pad ahead
            bikes_pad_y = closest_y_path - detect_range * np.sin(phi_path / 180 * pi)
            bikes_pad = dict(type="bicycle_1", x=bikes_pad_x, y=bikes_pad_y, v=ego_v,
                             phi=phi_path, w=0.48, l=2, route='padding', partici_type=0.0)

            tmp_bikes, fill_num_bikes = slice_or_fill(bikes, bikes_pad, ML_BIKE_NUM)

            # 2. fetch and sort persons in range
            persons = list(filter(lambda v:
                                  np.square(ego_y - v['y']) + np.square(ego_x - v['x']) <= np.square(detect_range) and
                                  abs(deal_with_phi(phi_path - v['phi'])) <= 135,
                                  persons))
            persons = sorted(persons, key=lambda v: np.square(ego_y - v['y']) + np.square(ego_x - v['x']))

            persons_pad_x = closest_x_path - detect_range * np.cos(phi_path / 180 * pi)
            persons_pad_y = closest_y_path - detect_range * np.sin(phi_path / 180 * pi)
            persons_pad = dict(type='DEFAULT_PEDTYPE', x=persons_pad_x, y=persons_pad_y, v=ego_v,
                               phi=phi_path, w=0.525, l=0.75, road="pad", partici_type=1.0)

            tmp_persons, fill_num_persons = slice_or_fill(persons, persons_pad, ML_PERSON_NUM)

            # 3. fetch and sort vehicles in range
            vehicles = list(filter(lambda v:
                                   np.square(ego_y - v['y']) + np.square(ego_x - v['x']) <= np.square(detect_range) and
                                   abs(deal_with_phi(phi_path - v['phi'])) <= 135,
                                   vehicles))
            vehicles = sorted(vehicles, key=lambda v: np.square(ego_y - v['y']) + np.square(ego_x - v['x']))

            vehicles_pad_x = closest_x_path - detect_range * np.cos(phi_path / 180 * pi)
            vehicles_pad_y = closest_y_path - detect_range * np.sin(phi_path / 180 * pi)
            vehicles_pad = dict(type="car_1", x=vehicles_pad_x, y=vehicles_pad_y, v=ego_v,
                                phi=phi_path, w=2.5, l=5, route='pad', partici_type=2.0)

            tmp_vehicles, fill_num_vehicles = slice_or_fill(vehicles, vehicles_pad, ML_VEH_NUM)

            tmp = dict(bikes=tmp_bikes,
                       persons=tmp_persons,
                       vehicles=tmp_vehicles)
            fill_num = dict(bikes=fill_num_bikes,
                            persons=fill_num_persons,
                            vehicles=fill_num_vehicles)
            return tmp, fill_num

        list_of_interested_veh_dict = []
        self.interested_vehs, self.fill_num = filter_interested_participants(self.all_vehicles, self.turning_direction)
        # pprint.pprint(self.interested_vehs)
        for part in list(self.interested_vehs.values()):
            list_of_interested_veh_dict.extend(part)

        for veh in list_of_interested_veh_dict:
            veh_x, veh_y, veh_v, veh_phi, veh_partici_type = veh['x'], veh['y'], veh['v'], veh['phi'], veh['partici_type']
            vehs_vector.extend([veh_x, veh_y, veh_v, veh_phi, veh_partici_type])

        return np.array(vehs_vector, dtype=np.float32)

    def recover_orig_position_fn(self, transformed_x, transformed_y, x, y, d):  # x, y, d are used to transform
        # maybe useless
        # coordination
        transformed_x, transformed_y, _ = rotate_coordination(transformed_x, transformed_y, 0, -d)
        orig_x, orig_y = shift_coordination(transformed_x, transformed_y, -x, -y)
        return orig_x, orig_y

    def _reset_init_state(self):
        # 初始化自车状态
        max_index = len(self.ref_path.adjacent_path[0])
        random_index = int(np.random.triangular(0, 1, 1, 1) * (max_index - 360))  # higher prob near junctions
        if np.random.random() <= self.prob_init_adj:  # initialize ego in the adjacent path with prob
            x, y, phi = self.ref_path.indexs2adj_path(random_index)
        else:
            x, y, phi = self.ref_path.indexs2points(random_index)

        # v = 7 + 6 * np.random.random()
        v = self.exp_v * np.random.random()
        routeID = 'self'
        return dict(ego=dict(v_x=v,
                             v_y=0 + (2 * np.random.random() - 1) * RANDOM_V_Y,
                             r=0 + (2 * np.random.random() - 1) * RANDOM_R,
                             x=x.numpy() + (2 * np.random.random() - 1) * RANDOM_X,
                             y=y.numpy() + (2 * np.random.random() - 1) * RANDOM_Y,
                             phi=phi.numpy() + (2 * np.random.random() - 1) * RANDOM_PHI,
                             l=self.ego_l,
                             w=self.ego_w,
                             routeID=routeID,
                             ))

    def compute_reward(self, obs, action):
        obses, actions = obs[np.newaxis, :], action[np.newaxis, :]

        # extract infos for each kind of participants
        start = 0; end = start + self.ego_info_dim + self.per_tracking_info_dim * (self.num_future_data + 1)
        obses_ego = obses[:, start:end]
        start = end; end = start + self.per_bike_info_dim * self.bike_num
        obses_bike = obses[:, start:end]
        start = end; end = start + self.per_person_info_dim * self.person_num
        obses_person = obses[:, start:end]
        start = end; end = start + self.per_veh_info_dim * self.veh_num
        obses_veh = obses[:, start:end]

        reward, _, _, _, _, _, _, reward_dict = self.env_model.compute_rewards(obses_ego, obses_bike, obses_person, obses_veh, actions)
        for k, v in reward_dict.items():
            reward_dict[k] = v.numpy()[0]
        return reward.numpy()[0], reward_dict

    def _struct_edge_area(self):
        # 构建edge
        self.edge_areas = []
        self.edge_lines = []

        def edge2roadline(edgeID):
            lane_list_whole = map_info[edgeID]['lane_list']
            lane_list = [ll for ll in lane_list_whole if map_info[edgeID][ll]['type'] != 'x']
            road_line_list = []
            road_area_list = []

            def centerline2points(start_point, end_point, width=3.75):
                # 通过车道中心线构建两侧车道线的起始点
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
                lane_shape = map_info[edgeID][l]['shape']
                start_point = lane_shape[0]
                end_point = lane_shape[1]  # 车道中心点的起始点
                width = map_info[edgeID][l]['lane_width']  # 车道宽度
                roadpoint1, roadpoint2, roadarea = centerline2points(start_point, end_point, width)
                road_line_list.append(points2line(roadpoint1))  # 两条车道线起始点的xy放在一起 -- > (x, x), (y, y)
                road_line_list.append(points2line(roadpoint2))

                road_area_list.append(points2line(roadarea))
            return road_line_list, road_area_list

        road_line_list, road_area_list = edge2roadline(self.current_edge)

        self.edge_areas.extend(road_area_list)
        self.edge_lines.extend(road_line_list)

    def render(self, mode='human', save_dir=None):
        light_line_width = 3
        dotted_line_style = '--'
        solid_line_style = '-'
        extension = 40

        def is_in_plot_area(x, y, ego_x, ego_y, tolerance=50):
            _distance2 = np.sqrt(np.square(x - ego_x) + np.square(y - ego_y))
            if _distance2 <= tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(ax, x, y, a, l, w, color, linestyle='-'):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

        def plot_phi_line(type, x, y, phi, color):
            if type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                line_length = 2
            elif type == 'DEFAULT_PEDTYPE':
                line_length = 1
            else:
                line_length = 5
            x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                             y + line_length * sin(phi * pi / 180.)
            plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

        def ploter(render_mode="human", save_dir=None):
            plt.cla()
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            fig = plt.gcf()
            # # ax = plt.gca()
            # fig.set_figheight(7.8)
            # fig.set_figwidth(16.2)
            # plt.axis("off")
            ax.axis("equal")

            # plot road lines
            for rl in self.edge_lines:
                plt.plot(rl[0], rl[1], 'k')

            for ra in self.edge_areas:
                plt.fill(*ra, 'lemonchiffon')

            # plot vehicles
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']

            for veh in self.all_vehicles:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                if veh_type in ['bicycle_1', 'bicycle_2', 'bicycle_3']:
                    veh_color = 'navy'
                elif veh_type == 'DEFAULT_PEDTYPE':
                    veh_color = 'purple'
                else:
                    veh_color = 'black'
                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, veh_color)
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, veh_color)

            # plot_interested vehs
            for veh in self.interested_vehs['vehicles']:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                # print("车辆信息", veh)
                # veh_type = 'car_1'
                task2color = {'left': 'b', 'straight': '#FF8000', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = 'straight'
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')
                    # print(f'veh_l = {veh_l}, veh_w = {veh_w}')

            # plot_interested bicycle
            for veh in self.interested_vehs['bikes']:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {'left': 'b', 'straight': '#FF8000', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = 'straight'
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

            # plot_interested person
            for veh in self.interested_vehs['persons']:
                veh_x = veh['x']
                veh_y = veh['y']
                veh_phi = veh['phi']
                veh_l = veh['l']
                veh_w = veh['w']
                veh_type = veh['type']
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {'left': 'b', 'straight': '#FF8000', 'right': 'm'}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, 'black')
                    task = 'straight'
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=':')

            # plot ego vehicle
            ego_v_x = self.ego_dynamics['v_x']
            ego_v_y = self.ego_dynamics['v_y']
            ego_r = self.ego_dynamics['r']
            ego_x = self.ego_dynamics['x']
            ego_y = self.ego_dynamics['y']
            ego_phi = self.ego_dynamics['phi']
            ego_l = self.ego_dynamics['l']
            ego_w = self.ego_dynamics['w']
            ego_alpha_f = self.ego_dynamics['alpha_f']
            ego_alpha_r = self.ego_dynamics['alpha_r']
            alpha_f_bound = self.ego_dynamics['alpha_f_bound']
            alpha_r_bound = self.ego_dynamics['alpha_r_bound']
            r_bound = self.ego_dynamics['r_bound']

            plot_phi_line('self_car', ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ax, ego_x, ego_y, ego_phi, ego_l, ego_w, 'red')

            # plot future data (static path)
            tracking_info = self.obs[self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                        self.num_future_data + 1)]
            future_path = tracking_info[self.per_tracking_info_dim:]
            for i in range(self.num_future_data):
                delta_x, delta_y, delta_phi = future_path[i * self.per_tracking_info_dim:
                                                          (i + 1) * self.per_tracking_info_dim]
                path_x, path_y, path_phi = ego_x + delta_x, ego_y + delta_y, ego_phi - delta_phi
                plt.plot(path_x, path_y, 'g.')
                plot_phi_line('self_car', path_x, path_y, path_phi, 'g')

            delta_, _, _ = tracking_info[:3]
            ax.plot(self.ref_path.path[0], self.ref_path.path[1], color='g')
            path_info = self.ref_path.find_closest_point4edge(np.array([ego_x], np.float32),
                                                              np.array([ego_y], np.float32))

            path_x, path_y, path_phi = path_info[0][0].numpy(), path_info[0][1].numpy(), path_info[0][2].numpy()
            plt.plot(path_x, path_y, 'g.')

            """
            delta_x, delta_y, delta_phi = (
                ego_x - path_x,
                ego_y - path_y,
                ego_phi - path_phi,
            )

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
                    reward_color_list.append('lavenderblush')
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

            text_color_list = ['ivory'] * 4 + ["lightcyan"] * 3 \
                              + ['ivory'] * 2 + ["lightcyan"] * 1 \
                              + ['ivory'] * 2 + ["lightcyan"] * 1 + ['ivory'] * 7 \
                              + ["salmon"] * 2 + ["palegreen"]

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
                cellText=cell_text, cellColours=cell_color,
                colWidths=[0.15] * 40,
            )
            """

            if render_mode == 'human':
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.show()
                plt.pause(0.001)
                return None
            elif render_mode == 'save':
                """
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.show()
                plt.savefig(save_dir, dpi=500)
                plt.pause(0.001)
                """
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                fig.canvas.draw()
                img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),
                                          fig.canvas.tostring_rgb())
                img.save(save_dir)
                return None
            elif render_mode == 'record':
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.savefig(save_dir, dpi=500)
                return None

        return ploter(render_mode=mode, save_dir=save_dir)

    def set_traj(self, trajectory):
        """set the real trajectory to reconstruct observation"""
        self.ref_path = trajectory


def tt():
    edge = 'EE6'
    env = HorizonMultiLaneEnv(training_task=edge, num_future_data=0, env_edge=edge, adj_ref_mode='random')
    obs = env.reset()

    for i in range(2000):
        action = np.array([0.0, 0.5], dtype=np.float32)
        s, r, d, info = env.step(action)
        env.render()
        if env._is_achieve_goal():
            exit()


def t_end2end():
    env = HorizonMultiLaneEnv(training_task='EE2', num_future_data=0)
    obs = env.reset()
    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            # action=2*np.random.random(2)-1
            if obs[4] < -18:
                action = np.array([0, 1], dtype=np.float32)
            elif obs[3] <= -18:
                action = np.array([0, 0], dtype=np.float32)
            else:
                action = np.array([0.2, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # extract infos for each kind of participants
            start = 0; end = start + env.ego_info_dim + env.per_tracking_info_dim * (env.num_future_data + 1)
            obses_ego = obses[:, start:end]
            start = end; end = start + env.per_bike_info_dim * env.bike_num
            obses_bike = obses[:, start:end]
            start = end; end = start + env.per_person_info_dim * env.person_num
            obses_person = obses[:, start:end]
            start = end; end = start + env.per_veh_info_dim * env.veh_num
            obses_veh = obses[:, start:end]

            obses_bike = np.reshape(obses_bike, [-1, env.per_bike_info_dim])
            obses_person = np.reshape(obses_person, [-1, env.per_person_info_dim])
            obses_veh = np.reshape(obses_veh, [-1, env.per_veh_info_dim])

            # env_model.reset(np.tile(obses_ego, (2, 1)), np.tile(obses_bike, (2, 1)), np.tile(obses_person, (2, 1)),
            #                 np.tile(obses_veh, (2, 1)), [env.ref_path.ref_index, random.randint(0, 2)])
            # env_model.mode = 'training'
            # for _ in range(10):
            #     obses_ego, obses_bike, obses_person, obses_veh, rewards, punish_term_for_training, \
            #         real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real = env_model.rollout_out(np.tile(actions, (2, 1)))
            # print(obses_ego.shape, obses_bike.shape, obses_person.shape, obses_veh.shape)
            # print(obses_bike[:, -1].numpy(), obses_person[:, -1].numpy(), obses_veh[:, -1].numpy())
            env.render()
            if done:
                break
        done = 0
        obs = env.reset()
        env.render()


if __name__ == '__main__':
    tt()
