#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import warnings
from math import pi

import bezier
import matplotlib.pyplot as plt
import numpy as np
import torch
# from torch import logical_and
from matplotlib.path import Path

from environment.env_horizon.misc_ic import (
    L,
    W,
    L_VEHICLE,
    W_VEHICLE,
    L_BIKE,
    W_BIKE,
    EPSILON_V,
    VEHICLE_MODE_LIST,
    BIKE_MODE_LIST,
    PERSON_MODE_LIST,
    VEH_NUM,
    BIKE_NUM,
    PERSON_NUM,
    map_info,
    MLANE_TASKS,
    ROUTE_JUNCTION,
    task2staticpath,
    turning_v_info,
    turning_b_info,
)

# LANE_WIDTH = 3.75

torch.set_num_threads(1)
# Is this same to
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)
warnings.filterwarnings("ignore")


class VehicleDynamics(object):
    def __init__(self):
        # self.vehicle_params = dict(C_f=-128915.5,  # front wheel cornering stiffness [N/rad]
        #                            C_r=-85943.6,  # rear wheel cornering stiffness [N/rad]
        #                            a=1.06,  # distance from CG to front axle [m]
        #                            b=1.85,  # distance from CG to rear axle [m]
        #                            mass=1412.,  # mass [kg]
        #                            I_z=1536.7,  # Polar moment of inertia at CG [kg*m^2]
        #                            miu=1.0,  # tire-road friction coefficient
        #                            g=9.81,  # acceleration of gravity [m/s^2]
        #                            )
        self.vehicle_params = dict(
            C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
            C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
            a=1.19,  # distance from CG to front axle [m]
            b=1.46,  # distance from CG to rear axle [m]
            mass=1520.0,  # mass [kg]
            I_z=2642.0,  # Polar moment of inertia at CG [kg*m^2]
            miu=0.8,  # tire-road friction coefficient
            g=9.81,  # acceleration of gravity [m/s^2]
        )
        a = self.vehicle_params["a"]
        b = self.vehicle_params["b"]
        mass = self.vehicle_params["mass"]
        g = self.vehicle_params["g"]

        F_zf = b * mass * g / (a + b)
        F_zr = a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf, F_zr=F_zr))

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        """
        torch method
        Parameters
        ----------
        states
        actions
        tau

        Returns
        -------

        """
        v_x, v_y, r, x, y, phi = (
            states[:, 0],
            states[:, 1],
            states[:, 2],
            states[:, 3],
            states[:, 4],
            states[:, 5],
        )
        phi = phi * np.pi / 180.0
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = torch.tensor(self.vehicle_params["C_f"], dtype=torch.float32)
        C_r = torch.tensor(self.vehicle_params["C_r"], dtype=torch.float32)
        a = torch.tensor(self.vehicle_params["a"], dtype=torch.float32)
        b = torch.tensor(self.vehicle_params["b"], dtype=torch.float32)
        mass = torch.tensor(self.vehicle_params["mass"], dtype=torch.float32)
        I_z = torch.tensor(self.vehicle_params["I_z"], dtype=torch.float32)
        miu = torch.tensor(self.vehicle_params["miu"], dtype=torch.float32)
        g = torch.tensor(self.vehicle_params["g"], dtype=torch.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = torch.where(a_x < 0, mass * a_x / 2, torch.zeros_like(a_x))
        F_xr = torch.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = torch.sqrt(torch.square(miu * F_zf) - torch.square(F_xf)) / F_zf
        miu_r = torch.sqrt(torch.square(miu * F_zr) - torch.square(F_xr)) / F_zr
        alpha_f = torch.atan((v_y + a * r) / (v_x + 1e-8)) - steer
        alpha_r = torch.atan((v_y - b * r) / (v_x + 1e-8))

        next_state = [
            v_x + tau * (a_x + v_y * r),
            (
                mass * v_y * v_x
                + tau * (a * C_f - b * C_r) * r
                - tau * C_f * steer * v_x
                - tau * mass * torch.square(v_x) * r
            )
            / (mass * v_x - tau * (C_f + C_r)),
            (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x)
            / (tau * (torch.square(a) * C_f + torch.square(b) * C_r) - I_z * v_x),
            x + tau * (v_x * torch.cos(phi) - v_y * torch.sin(phi)),
            y + tau * (v_x * torch.sin(phi) + v_y * torch.cos(phi)),
            (phi + tau * r) * 180 / np.pi,
        ]

        return torch.stack(next_state, 1), torch.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        """
        torch method
        Parameters
        ----------
        x_1
        u_1
        frequency

        Returns
        -------

        """
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params


def lane2edge_re(lane_ID):
    rex_num = re.compile("\_\d*$")
    edge_ID = rex_num.sub("", lane_ID)
    return edge_ID


class _EnvironmentModel(object):  # all tensors
    def __init__(self, training_task4, num_future_data=0, mode="training", expected_v=None, **kwargs):
        self.joe, self.edge, self.from_edge, self.to_edge = training_task4

        self.mode = mode
        self.vehicle_dynamics = VehicleDynamics()  # 自车的动力学模型
        self.base_frequency = 10.0
        self.obses = None
        self.ego_params = None
        self.actions = None

        self.junction_areas = False

        self.ref_path = ReferencePath(training_task4=training_task4, expected_v=expected_v)
        self.ref_indexes = None
        self.num_future_data = num_future_data
        self.exp_v = expected_v
        assert expected_v is not None
        self.reward_info = None

        if self.joe == "junction":  # joe --> junction or edge
            self.turning_direction = ROUTE_JUNCTION[self.from_edge][1]
        elif self.joe == "edge":
            self.turning_direction = "straight"
        else:
            raise ValueError("Have not set task properly")

        self.veh_num = VEH_NUM[self.turning_direction]
        self.bike_num = BIKE_NUM[self.turning_direction]
        self.person_num = PERSON_NUM[self.turning_direction]
        self.ego_info_dim = 6
        self.per_veh_info_dim = 5
        self.per_bike_info_dim = 5
        self.per_person_info_dim = 5
        self.per_tracking_info_dim = 3
        self.obses_ego = None
        self.path_infos = None
        self.obses_bike = None
        self.obses_person = None
        self.obses_veh = None

        if self.joe == "junction":
            self._struct_junction_area()

    def _struct_junction_area(self):
        # 构建十字路口中心区域
        route_list = task2staticpath[self.from_edge]
        ref_route = route_list[self.ref_path.ref_index]
        e, l = ref_route[-2]
        junctionID = ROUTE_JUNCTION[e][0]
        junction_points = map_info[junctionID]["shape"]
        self.junction_areas = junction_points

    def reset(self, obses_ego, obses_bike, obses_person, obses_veh, ref_indexes=None):  # input are all tensors
        self.obses_ego = obses_ego.clone()
        self.path_infos = obses_ego[:, self.ego_info_dim :].clone()
        self.obses_bike = obses_bike.reshape(-1, self.per_bike_info_dim * self.bike_num)
        self.obses_person = obses_person.reshape(-1, self.per_person_info_dim * self.person_num)
        self.obses_veh = obses_veh.reshape(-1, self.per_veh_info_dim * self.veh_num)
        self.ref_indexes = ref_indexes
        self.actions = None
        self.reward_info = None

    def add_traj(self, obses_ego, obses_bike, obses_person, obses_veh, path_index):
        # maybe useless
        self.obses_ego = obses_ego

        self.obses_bike = obses_bike.reshape(-1, self.per_bike_info_dim * self.bike_num)
        self.obses_person = obses_person.reshape(-1, self.per_person_info_dim * self.person_num)
        self.obses_veh = obses_veh.reshape(-1, self.per_veh_info_dim * self.veh_num)
        self.ref_path.set_path(path_index)
        self._struct_junction_area()

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        if isinstance(actions, np.ndarray):
            actions = torch.Tensor(actions)
        self.actions = self._action_transformation_for_end2end(actions)

        (
            rewards,
            punish_term_for_training,
            real_punish_term,
            veh2veh4real,
            veh2road4real,
            veh2bike4real,
            veh2person4real,
            reward_dict,
        ) = self.compute_rewards(
            self.obses_ego,
            self.obses_bike,
            self.obses_person,
            self.obses_veh,
            self.actions,
        )

        (self.obses_ego, self.obses_bike, self.obses_person, self.obses_veh,) = self.compute_next_obses(
            self.obses_ego,
            self.obses_bike,
            self.obses_person,
            self.obses_veh,
            self.actions,
        )  # output original information

        obses_bike = self.obses_bike.reshape(-1, self.per_bike_info_dim)
        obses_person = self.obses_person.reshape(-1, self.per_person_info_dim)
        obses_veh = self.obses_veh.reshape(-1, self.per_veh_info_dim)

        return (
            self.obses_ego,
            obses_bike,
            obses_person,
            obses_veh,
            rewards,
            punish_term_for_training,
            real_punish_term,
            veh2veh4real,
            veh2road4real,
            veh2bike4real,
            veh2person4real,
            reward_dict,
        )

    def _action_transformation_for_end2end(self, actions):  # [-1, 1]
        actions = torch.clamp(actions, -1.05, 1.05)
        steer_norm, a_xs_norm = actions[:, 0], actions[:, 1]
        steer_scale, a_xs_scale = 0.4 * steer_norm, 2.25 * a_xs_norm - 0.75
        return torch.stack([steer_scale, a_xs_scale], 1)

    def compute_rewards(self, obses_ego, obses_bike, obses_person, obses_veh, actions):
        raise NotImplementedError("compute reward method is not implemented")

    def compute_next_obses(self, obses_ego, obses_bike, obses_person, obses_veh, actions):
        # obses = self.convert_vehs_to_abso(obses)
        if isinstance(obses_ego, np.ndarray):
            obses_ego = torch.tensor(obses_ego)
        if isinstance(obses_bike, np.ndarray):
            obses_bike = torch.tensor(obses_bike)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_veh, np.ndarray):
            obses_veh = torch.tensor(obses_veh)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions)

        ego_infos, tracking_infos = (
            obses_ego[:, : self.ego_info_dim],
            obses_ego[:, self.ego_info_dim :],
        )
        bike_infos = obses_bike.detach()
        person_infos = obses_person.detach()
        veh_infos = obses_veh.detach()

        next_ego_infos = self.ego_predict(ego_infos, actions)
        # different for training and selecting
        if self.mode != "training":
            next_tracking_infos = self.ref_path.tracking_error_vector(
                next_ego_infos[:, 3],
                next_ego_infos[:, 4],
                next_ego_infos[:, 5],
                next_ego_infos[:, 0],
                self.num_future_data,
            )
        else:
            if self.joe == "junction":  # joe --> junction or edge
                next_tracking_infos = torch.zeros(
                    len(next_ego_infos),
                    (self.num_future_data + 1) * self.per_tracking_info_dim,
                )

                # print(self.ref_indexes)
                # print(torch.Tensor(self.ref_indexes))
                ref_indexes = torch.unsqueeze(torch.as_tensor(self.ref_indexes, dtype=torch.int), dim=1)
                for ref_idx, path in enumerate(self.ref_path.path_list):
                    self.ref_path.path = path
                    tracking_info_4_this_ref_idx = self.ref_path.tracking_error_vector(
                        next_ego_infos[:, 3],
                        next_ego_infos[:, 4],
                        next_ego_infos[:, 5],
                        next_ego_infos[:, 0],
                        self.num_future_data,
                    )
                    next_tracking_infos = torch.where(
                        ref_indexes == ref_idx,
                        tracking_info_4_this_ref_idx,
                        next_tracking_infos,
                    )
            elif self.joe == "edge":
                next_tracking_infos = self.path_infos.detach()
            else:
                raise ValueError("Have not set joe properly")

        next_bike_infos = self.bike_predict(bike_infos)
        if not person_infos.shape[1]:  # no pedestrian is considered
            next_person_infos = person_infos
        else:
            next_person_infos = self.person_predict(person_infos)
        next_veh_infos = self.veh_predict(veh_infos)

        next_obses_ego = torch.cat([next_ego_infos, next_tracking_infos], 1)
        # next_obses = self.convert_vehs_to_rela(next_obses)
        return next_obses_ego, next_bike_infos, next_person_infos, next_veh_infos

    # def convert_vehs_to_rela(self, obs_abso):
    #     ego_infos, tracking_infos, veh_infos = obs_abso[:, :self.ego_info_dim], \
    #                                            obs_abso[:, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                      self.num_future_data + 1)], \
    #                                            obs_abso[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                        self.num_future_data + 1):]
    #     ego_x, ego_y = ego_infos[:, 3], ego_infos[:, 4]
    #     ego = tf.tile(torch.stack([ego_x, ego_y, torch.zeros_like(ego_x), torch.zeros_like(ego_x)], 1),
    #                   (1, int(tf.shape(veh_infos)[1]/self.per_veh_info_dim)))
    #     vehs_rela = veh_infos - ego
    #     out = tf.concat([ego_infos, tracking_infos, vehs_rela], 1)
    #     return out

    # def convert_vehs_to_abso(self, obs_rela):
    #     ego_infos, tracking_infos, veh_rela = obs_rela[:, :self.ego_info_dim], \
    #                                            obs_rela[:, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1)], \
    #                                            obs_rela[:, self.ego_info_dim + self.per_tracking_info_dim * (
    #                                                    self.num_future_data + 1):]
    #     ego_x, ego_y = ego_infos[:, 3], ego_infos[:, 4]
    #     ego = tf.tile(torch.stack([ego_x, ego_y, torch.zeros_like(ego_x), torch.zeros_like(ego_x)], 1),
    #                   (1, int(tf.shape(veh_rela)[1] / self.per_veh_info_dim)))
    #     vehs_abso = veh_rela + ego
    #     out = tf.concat([ego_infos, tracking_infos, vehs_abso], 1)
    #     return out

    def tracking_error_predict_for_edge(self, ego_infos, tracking_infos, next_ego_infos):
        assert self.num_future_data == 0, "ERROR: num_future_data does not equal to 0!"
        next_tracking_infos = torch.zeros(len(next_ego_infos), self.per_tracking_info_dim)
        next_tracking_infos[:, 2] = tracking_infos[:, 2] + next_ego_infos[:, 0] - ego_infos[:, 0]
        heading_ego2path = deal_with_phi_diff(tracking_infos[:, 1] + next_ego_infos[:, 5] - ego_infos[:, 5])
        next_tracking_infos[:, 1] = heading_ego2path
        next_tracking_infos[:, 0] = (tracking_infos[:, 0]
                                     + torch.mul(next_ego_infos[:, 0], torch.sin(heading_ego2path * pi / 180)) / self.base_frequency)
        return next_tracking_infos

    def ego_predict(self, ego_infos, actions):
        ego_next_infos, _ = self.vehicle_dynamics.prediction(ego_infos[:, :6], actions, self.base_frequency)
        v_xs, v_ys, rs, xs, ys, phis = (
            ego_next_infos[:, 0],
            ego_next_infos[:, 1],
            ego_next_infos[:, 2],
            ego_next_infos[:, 3],
            ego_next_infos[:, 4],
            ego_next_infos[:, 5],
        )
        v_xs = torch.clamp(v_xs, 0.0, 35.0)
        ego_next_infos = torch.stack([v_xs, v_ys, rs, xs, ys, phis], 1)
        return ego_next_infos

    def bike_predict(self, bike_infos):
        bike_mode_list = BIKE_MODE_LIST[self.turning_direction]
        predictions_to_be_concat = []
        for bikes_index in range(len(bike_mode_list)):
            predictions_to_be_concat.append(
                self.predict_for_bike_mode(
                    bike_infos[
                        :,
                        bikes_index * self.per_bike_info_dim : (bikes_index + 1) * self.per_bike_info_dim,
                    ],
                    bike_mode_list[bikes_index],
                )
            )
        pred = torch.cat(predictions_to_be_concat, 1).detach()
        return pred

    def person_predict(self, person_infos):
        person_mode_list = PERSON_MODE_LIST[self.turning_direction]
        pred = []

        for persons_index in range(len(person_mode_list)):
            persons = person_infos[
                :,
                persons_index * self.per_person_info_dim : (persons_index + 1) * self.per_person_info_dim,
            ]

            person_xs, person_ys, person_vs, person_phis, person_index = (
                persons[:, 0],
                persons[:, 1],
                persons[:, 2],
                persons[:, 3],
                persons[:, 4],
            )
            person_phis_rad = person_phis * np.pi / 180.0

            person_xs_delta = person_vs / self.base_frequency * torch.cos(person_phis_rad)
            person_ys_delta = person_vs / self.base_frequency * torch.sin(person_phis_rad)

            next_person_xs, next_person_ys, next_person_vs, next_person_phis_rad = (
                person_xs + person_xs_delta,
                person_ys + person_ys_delta,
                person_vs,
                person_phis_rad,
            )
            next_person_phis_rad = torch.where(
                next_person_phis_rad > np.pi,
                next_person_phis_rad - 2 * np.pi,
                next_person_phis_rad,
            )
            next_person_phis_rad = torch.where(
                next_person_phis_rad <= -np.pi,
                next_person_phis_rad + 2 * np.pi,
                next_person_phis_rad,
            )
            next_person_phis = next_person_phis_rad * 180 / np.pi
            next_person_index = person_index
            pred.append(
                torch.stack(
                    [
                        next_person_xs,
                        next_person_ys,
                        next_person_vs,
                        next_person_phis,
                        next_person_index,
                    ],
                    1,
                )
            )

        pred = torch.cat(pred, 1)
        return pred

    def veh_predict(self, veh_infos):
        veh_mode_list = VEHICLE_MODE_LIST[self.turning_direction]
        # print(veh_mode_list)
        predictions_to_be_concat = []

        for vehs_index in range(len(veh_mode_list)):
            predictions_to_be_concat.append(
                self.predict_for_veh_mode(
                    veh_infos[
                        :,
                        vehs_index * self.per_veh_info_dim : (vehs_index + 1) * self.per_veh_info_dim,
                    ],
                    veh_mode_list[vehs_index],
                )
            )
        pred = torch.cat(predictions_to_be_concat, 1).detach()
        return pred

    def predict_for_bike_mode(self, bikes, mode):
        bike_xs, bike_ys, bike_vs, bike_phis, bike_index = (
            bikes[:, 0],
            bikes[:, 1],
            bikes[:, 2],
            bikes[:, 3],
            bikes[:, 4],
        )
        bike_phis_rad = bike_phis * np.pi / 180.0

        zeros = torch.zeros_like(bike_xs)

        bike_xs_delta = bike_vs / self.base_frequency * torch.cos(bike_phis_rad)
        bike_ys_delta = bike_vs / self.base_frequency * torch.sin(bike_phis_rad)

        if self.joe == "junction":
            middle_cond = self.judge_in_middle(bike_xs, bike_ys)
            if mode in ["dl_b", "rd_b", "ur_b", "lu_b"]:
                route_list = task2staticpath[self.from_edge]
                ref_route = route_list[self.ref_path.ref_index]
                e, _ = ref_route[-2]
                in_lane_ID = turning_b_info[e][mode]["path"][0][0]
                out_lane_ID = turning_b_info[e][mode]["path"][0][-1]
                in_edge_ID = lane2edge_re(in_lane_ID)
                out_edge_ID = lane2edge_re(out_lane_ID)
                in_lane_shape = map_info[in_edge_ID][in_lane_ID]["shape"]
                out_lane_shape = map_info[out_edge_ID][out_lane_ID]["shape"]
                est_R = self.compute_turning_r(in_lane_shape, out_lane_shape)
                # print("bike:", est_R)
                bike_phis_rad_delta = torch.where(middle_cond, (bike_vs / (est_R + 1e-8)) / self.base_frequency, zeros)
            elif mode in ["dr_b", "ru_b", "ul_b", "ld_b"]:
                route_list = task2staticpath[self.from_edge]
                ref_route = route_list[self.ref_path.ref_index]
                e, _ = ref_route[-2]
                in_lane_ID = turning_b_info[e][mode]["path"][0][0]
                out_lane_ID = turning_b_info[e][mode]["path"][0][-1]
                in_edge_ID = lane2edge_re(in_lane_ID)
                out_edge_ID = lane2edge_re(out_lane_ID)
                in_lane_shape = map_info[in_edge_ID][in_lane_ID]["shape"]
                out_lane_shape = map_info[out_edge_ID][out_lane_ID]["shape"]
                est_R = self.compute_turning_r(in_lane_shape, out_lane_shape)
                # print("bike:", est_R)
                bike_phis_rad_delta = torch.where(
                    middle_cond,
                    torch.as_tensor(-(bike_vs / (est_R + 1e-8)) / self.base_frequency, dtype=torch.float32),
                    zeros,
                )
            else:
                bike_phis_rad_delta = zeros
        elif self.joe == "edge":
            bike_phis_rad_delta = zeros
        else:
            raise ValueError("Have not set task properly")

        next_bike_xs = bike_xs + bike_xs_delta
        next_bike_ys = bike_ys + bike_ys_delta
        next_bike_vs = bike_vs
        next_bike_phis_rad = bike_phis_rad + bike_phis_rad_delta

        next_bike_phis_rad = torch.where(
            next_bike_phis_rad > np.pi,
            next_bike_phis_rad - 2 * np.pi,
            next_bike_phis_rad,
        )
        next_bike_phis_rad = torch.where(
            next_bike_phis_rad <= -np.pi,
            next_bike_phis_rad + 2 * np.pi,
            next_bike_phis_rad,
        )
        next_bike_phis = next_bike_phis_rad * 180 / np.pi
        next_bike_index = bike_index
        return torch.stack(
            [next_bike_xs, next_bike_ys, next_bike_vs, next_bike_phis, next_bike_index],
            1,
        )

    def compute_circle_center(self, x_a, y_a, x_b, y_b, x_c, y_c, x_d, y_d):
        """
        @Author: wzs
        @Time: 20210907
        @Function: 计算路口圆弧的圆心坐标。
        @Input:
            x_a: 车辆当前所在lane的末端x坐标；
            y_a: 车辆当前所在lane的末端y坐标；
            x_b: 车辆当前所在lane的始端x坐标；
            y_b: 车辆当前所在lane的始端y坐标；
            x_c: 车辆将要转到的下一个lane的始端x坐标；
            y_c: 车辆将要转到的下一个lane的始端y坐标；
            x_d: 车辆将要转到的下一个lane的末端x坐标；
            y_d: 车辆将要转到的下一个lane的末端y坐标。

        @Output:
            x_r: 路口圆弧圆心x坐标；
            y_r: 路口圆弧圆心y坐标；
        """

        delta_x_ba = x_b - x_a  # 1
        delta_y_ba = y_b - y_a  # 2
        delta_x_dc = x_d - x_c  # 3
        delta_y_dc = y_d - y_c  # 4

        x_r = (
            delta_y_ba * delta_y_dc * (y_a - y_c) + delta_x_ba * delta_y_dc * x_a - delta_x_dc * delta_y_ba * x_c
        ) / (delta_x_ba * delta_y_dc - delta_x_dc * delta_y_ba)
        y_r = (
            delta_x_ba * delta_x_dc * (x_a - x_c) + delta_y_ba * delta_x_dc * y_a - delta_y_dc * delta_x_ba * y_c
        ) / (delta_y_ba * delta_x_dc - delta_y_dc * delta_x_ba)

        return x_r, y_r

    def compute_turning_r(self, in_lane_shape, out_lane_shape):
        """
        in_lane_shape = a1, a2
        out_lane_shape = b1, b2

        (b1-o) (b2-b1) = 0
        (a2-o) (a2-a1) = 0
        :param in_lane_shape:
        :param out_lane_shape:
        :return:
        """

        # 初始路口的ID和shape
        lane_init_shape = in_lane_shape

        # 获取下一路段的ID和shape
        lane_next_shape = out_lane_shape

        # 圆心坐标计算
        x_r, y_r = self.compute_circle_center(
            lane_init_shape[1][0],
            lane_init_shape[1][1],
            lane_init_shape[0][0],
            lane_init_shape[0][1],
            lane_next_shape[0][0],
            lane_next_shape[0][1],
            lane_next_shape[1][0],
            lane_next_shape[1][1],
        )

        # 半径R的计算
        est_r = np.sqrt(np.power(x_r - lane_init_shape[1][0], 2) + np.power(y_r - lane_init_shape[1][1], 2))
        # print('est_r', est_r)

        return est_r

    def predict_for_veh_mode(self, vehs, mode):
        """
        预测周车的轨迹， 在十字路口范围内，给一个固定的横摆角速度。其他位置直线递推。
        """
        veh_xs = vehs[:, 0]
        veh_ys = vehs[:, 1]
        veh_vs = vehs[:, 2]
        veh_phis = vehs[:, 3]
        veh_index = vehs[:, 4]

        veh_phis_rad = veh_phis * np.pi / 180.0

        zeros = torch.zeros_like(veh_xs)

        veh_xs_delta = veh_vs / self.base_frequency * torch.cos(veh_phis_rad)
        veh_ys_delta = veh_vs / self.base_frequency * torch.sin(veh_phis_rad)

        if self.joe == "junction":
            middle_cond = self.judge_in_middle(veh_xs, veh_ys)

            if mode in ["dl", "rd", "ur", "lu"]:
                route_list = task2staticpath[self.from_edge]
                ref_route = route_list[self.ref_path.ref_index]
                e, _ = ref_route[-2]
                if turning_v_info[e][mode]["path"]:
                    in_lane_ID = turning_v_info[e][mode]["path"][0][0]
                    out_lane_ID = turning_v_info[e][mode]["path"][0][-1]
                    in_edge_ID = lane2edge_re(in_lane_ID)
                    out_edge_ID = lane2edge_re(out_lane_ID)
                    in_lane_shape = map_info[in_edge_ID][in_lane_ID]["shape"]
                    out_lane_shape = map_info[out_edge_ID][out_lane_ID]["shape"]
                    est_R = self.compute_turning_r(in_lane_shape, out_lane_shape)
                    # print('veh:', est_R)
                    veh_phis_rad_delta = torch.where(
                        middle_cond,
                        (veh_vs / (est_R + 1e-8)) / self.base_frequency,
                        zeros,
                    )
                else:
                    veh_phis_rad_delta = zeros

            elif mode in ["dr", "ru", "ul", "ld"]:
                route_list = task2staticpath[self.from_edge]
                ref_route = route_list[self.ref_path.ref_index]
                e, _ = ref_route[-2]

                if turning_v_info[e][mode]["path"]:
                    in_lane_ID = turning_v_info[e][mode]["path"][0][0]
                    out_lane_ID = turning_v_info[e][mode]["path"][0][-1]
                    in_edge_ID = lane2edge_re(in_lane_ID)
                    out_edge_ID = lane2edge_re(out_lane_ID)
                    in_lane_shape = map_info[in_edge_ID][in_lane_ID]["shape"]
                    out_lane_shape = map_info[out_edge_ID][out_lane_ID]["shape"]
                    est_R = self.compute_turning_r(in_lane_shape, out_lane_shape)
                    # print('veh:', est_R)
                    veh_phis_rad_delta = torch.where(
                        middle_cond,
                        -(veh_vs / (est_R + 1e-8)) / self.base_frequency,
                        zeros,
                    )
                else:
                    veh_phis_rad_delta = zeros
                    # print('>>>>', turning_v_info[e][mode])

            else:
                veh_phis_rad_delta = zeros
        elif self.joe == "edge":
            veh_phis_rad_delta = zeros
        else:
            raise ValueError("Have not set task properly")

        next_veh_xs = veh_xs + veh_xs_delta
        next_veh_ys = veh_ys + veh_ys_delta
        next_veh_vs = veh_vs
        next_veh_phis_rad = veh_phis_rad + veh_phis_rad_delta

        next_veh_phis_rad = torch.where(next_veh_phis_rad > np.pi, next_veh_phis_rad - 2 * np.pi, next_veh_phis_rad)
        next_veh_phis_rad = torch.where(
            next_veh_phis_rad <= -np.pi,
            next_veh_phis_rad + 2 * np.pi,
            next_veh_phis_rad,
        )
        next_veh_phis = next_veh_phis_rad * 180 / np.pi
        next_veh_index = veh_index
        return torch.stack([next_veh_xs, next_veh_ys, next_veh_vs, next_veh_phis, next_veh_index], 1)

    def judge_in_middle(self, veh_x, veh_y):
        # maybe useless
        veh_stack = torch.stack([veh_x, veh_y], 1)
        path_area = Path(self.junction_areas)

        out_ = path_area.contains_points(veh_stack)

        return torch.as_tensor(out_, dtype=bool)

    def render(self, mode="human"):
        pass
        """
        if mode == 'human':
            # plot basic map
            square_length = CROSSROAD_SIZE
            extension = 40
            lane_width = LANE_WIDTH
            dotted_line_style = '--'
            solid_line_style = '-'

            plt.cla()
            plt.title("Crossroad")
            ax = plt.axes(xlim=(-square_length / 2 - extension, square_length / 2 + extension),
                          ylim=(-square_length / 2 - extension, square_length / 2 + extension))
            plt.axis("equal")
            plt.axis('off')

            # ax.add_patch(plt.Rectangle((-square_length / 2, -square_length / 2),
            #                            square_length, square_length, edgecolor='black', facecolor='none'))
            ax.add_patch(plt.Rectangle((-square_length / 2 - extension, -square_length / 2 - extension),
                                       square_length + 2 * extension, square_length + 2 * extension, edgecolor='black',
                                       facecolor='none'))

            # ----------horizon--------------
            plt.plot([-square_length / 2 - extension, -square_length / 2], [0, 0], color='black')
            plt.plot([square_length / 2 + extension, square_length / 2], [0, 0], color='black')

            #
            for i in range(1, LANE_NUMBER+1):
                linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
                plt.plot([-square_length / 2 - extension, -square_length / 2], [i*lane_width, i*lane_width],
                         linestyle=linestyle, color='black')
                plt.plot([square_length / 2 + extension, square_length / 2], [i*lane_width, i*lane_width],
                         linestyle=linestyle, color='black')
                plt.plot([-square_length / 2 - extension, -square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black')
                plt.plot([square_length / 2 + extension, square_length / 2], [-i * lane_width, -i * lane_width],
                         linestyle=linestyle, color='black')

            # ----------vertical----------------
            plt.plot([0, 0], [-square_length / 2 - extension, -square_length / 2], color='black')
            plt.plot([0, 0], [square_length / 2 + extension, square_length / 2], color='black')

            #
            for i in range(1, LANE_NUMBER+1):
                linestyle = dotted_line_style if i < LANE_NUMBER else solid_line_style
                plt.plot([i*lane_width, i*lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black')
                plt.plot([i*lane_width, i*lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black')
                plt.plot([-i * lane_width, -i * lane_width], [-square_length / 2 - extension, -square_length / 2],
                         linestyle=linestyle, color='black')
                plt.plot([-i * lane_width, -i * lane_width], [square_length / 2 + extension, square_length / 2],
                         linestyle=linestyle, color='black')

            # ----------stop line--------------
            plt.plot([0, LANE_NUMBER * lane_width], [-square_length / 2, -square_length / 2], color='black')
            plt.plot([-LANE_NUMBER * lane_width, 0], [square_length / 2, square_length / 2], color='black')
            plt.plot([-square_length / 2, -square_length / 2], [0, -LANE_NUMBER * lane_width], color='black')
            plt.plot([square_length / 2, square_length / 2], [LANE_NUMBER * lane_width, 0], color='black')

            # ----------Oblique--------------
            plt.plot([LANE_NUMBER * lane_width, square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                     color='black')
            plt.plot([LANE_NUMBER * lane_width, square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                     color='black')
            plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [-square_length / 2, -LANE_NUMBER * lane_width],
                     color='black')
            plt.plot([-LANE_NUMBER * lane_width, -square_length / 2], [square_length / 2, LANE_NUMBER * lane_width],
                     color='black')

            def is_in_plot_area(x, y, tolerance=5):
                if -square_length / 2 - extension + tolerance < x < square_length / 2 + extension - tolerance and \
                        -square_length / 2 - extension + tolerance < y < square_length / 2 + extension - tolerance:
                    return True
                else:
                    return False

            def draw_rotate_rec(x, y, a, l, w, color):
                RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
                RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
                LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
                LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
                ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color)
                ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color)
                ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color)
                ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color)

            def plot_phi_line(x, y, phi, color):
                line_length = 3
                x_forw, y_forw = x + line_length * cos(phi * pi / 180.), \
                                 y + line_length * sin(phi * pi / 180.)
                plt.plot([x, x_forw], [y, y_forw], color=color, linewidth=0.5)

            # abso_obs = self.convert_vehs_to_abso(self.obses)
            obses = self.obses.numpy()
            ego_info, tracing_info, vehs_info = obses[0, :self.ego_info_dim], \
                                                obses[0, self.ego_info_dim:self.ego_info_dim + self.per_tracking_info_dim * (
                                                                                          self.num_future_data + 1)], \
                                                obses[0, self.ego_info_dim + self.per_tracking_info_dim * (
                                                            self.num_future_data + 1):]
            # plot cars
            for veh_index in range(int(len(vehs_info) / self.per_veh_info_dim)):
                veh = vehs_info[self.per_veh_info_dim * veh_index:self.per_veh_info_dim * (veh_index + 1)]
                veh_x, veh_y, veh_v, veh_phi = veh

                if is_in_plot_area(veh_x, veh_y):
                    plot_phi_line(veh_x, veh_y, veh_phi, 'black')
                    draw_rotate_rec(veh_x, veh_y, veh_phi, L, W, 'black')

            # plot own car
            delta_y, delta_phi = tracing_info[0], tracing_info[1]
            ego_v_x, ego_v_y, ego_r, ego_x, ego_y, ego_phi = ego_info

            plot_phi_line(ego_x, ego_y, ego_phi, 'red')
            draw_rotate_rec(ego_x, ego_y, ego_phi, L, W, 'red')

            # plot text
            text_x, text_y_start = -110, 60
            ge = iter(range(0, 1000, 4))
            plt.text(text_x, text_y_start - next(ge), 'ego_x: {:.2f}m'.format(ego_x))
            plt.text(text_x, text_y_start - next(ge), 'ego_y: {:.2f}m'.format(ego_y))
            plt.text(text_x, text_y_start - next(ge), 'delta_y: {:.2f}m'.format(delta_y))
            plt.text(text_x, text_y_start - next(ge), r'ego_phi: ${:.2f}\degree$'.format(ego_phi))
            plt.text(text_x, text_y_start - next(ge), r'delta_phi: ${:.2f}\degree$'.format(delta_phi))

            plt.text(text_x, text_y_start - next(ge), 'v_x: {:.2f}m/s'.format(ego_v_x))
            plt.text(text_x, text_y_start - next(ge), 'exp_v: {:.2f}m/s'.format(self.exp_v))
            plt.text(text_x, text_y_start - next(ge), 'v_y: {:.2f}m/s'.format(ego_v_y))
            plt.text(text_x, text_y_start - next(ge), 'yaw_rate: {:.2f}rad/s'.format(ego_r))

            if self.actions is not None:
                steer, a_x = self.actions[0, 0], self.actions[0, 1]
                plt.text(text_x, text_y_start - next(ge),
                         r'steer: {:.2f}rad (${:.2f}\degree$)'.format(steer, steer * 180 / np.pi))
                plt.text(text_x, text_y_start - next(ge), 'a_x: {:.2f}m/s^2'.format(a_x))

            text_x, text_y_start = 70, 60
            ge = iter(range(0, 1000, 4))

            # reward info
            if self.reward_info is not None:
                for key, val in self.reward_info.items():
                    plt.text(text_x, text_y_start - next(ge), '{}: {:.4f}'.format(key, val))

            plt.show()
            plt.pause(0.1)
            """


class MultiLaneEnvModel(_EnvironmentModel):
    def compute_rewards(self, obses_ego, obses_bike, obses_person, obses_veh, actions):
        # obses = self.convert_vehs_to_abso(obses)
        if isinstance(obses_ego, np.ndarray):
            obses_ego = torch.tensor(obses_ego)
        if isinstance(obses_bike, np.ndarray):
            obses_bike = torch.tensor(obses_bike)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_veh, np.ndarray):
            obses_veh = torch.tensor(obses_veh)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions)

        # recompute tracking info from original information
        v_x = obses_ego[:, 0]
        x = obses_ego[:, 3]
        y = obses_ego[:, 4]
        phi = obses_ego[:, 5]
        closest_x_path = obses_ego[:, 6]
        closest_y_path = obses_ego[:, 7]
        phi_path = obses_ego[:, 8]  # [-180, 180]

        dist2path = dist_ego2path(x, y, closest_x_path, closest_y_path, phi_path)
        phi2path = deal_with_phi_diff(phi - phi_path)
        vel2path = v_x - self.exp_v
        sin4phi2path = torch.sin(phi2path * pi / 180.0)

        bike_infos = obses_bike.detach()
        person_infos = obses_person.detach()
        veh_infos = obses_veh.detach()

        steers, a_xs = actions[:, 0], actions[:, 1]
        # rewards related to action
        punish_steer = -torch.square(steers)
        punish_a_x = -torch.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -torch.square(obses_ego[:, 2])

        # rewards related to tracking error
        devi_v = -(torch.square((v_x - self.exp_v) / self.exp_v)
                   + torch.div(1.0, 1.0 + torch.exp(-10.0 * (torch.abs(v_x - self.exp_v + 1) - 1.0)))
                   + torch.div(1.0, 1.0 + torch.exp(-10.0 * (v_x - self.exp_v))))
        devi_y = -torch.square(dist2path)
        devi_phi = -torch.square(phi2path * pi / 180.0)

        # rewards related to veh2veh collision
        safe_gap = 0.1
        ego_v_x = obses_ego[:, 0]
        ego_x = obses_ego[:, 3]
        ego_y = obses_ego[:, 4]

        ego_lws = (L - W) / 2.0
        ego_front_points = (
            obses_ego[:, 3] + ego_lws * torch.cos(obses_ego[:, 5] * np.pi / 180.0),
            obses_ego[:, 4] + ego_lws * torch.sin(obses_ego[:, 5] * np.pi / 180.0),
        )
        ego_rear_points = (
            obses_ego[:, 3] - ego_lws * torch.cos(obses_ego[:, 5] * np.pi / 180.0),
            obses_ego[:, 4] - ego_lws * torch.sin(obses_ego[:, 5] * np.pi / 180.0),
        )

        veh2veh4real = torch.zeros_like(veh_infos[:, 0])
        veh2veh4training = torch.zeros_like(veh_infos[:, 0])
        for veh_index in range(int(veh_infos.shape[1] / self.per_veh_info_dim)):
            min_long_gap = (L + L_VEHICLE) / 2 + safe_gap
            min_lat_gap = (W + W_VEHICLE) / 2 + safe_gap
            vehs = veh_infos[:, veh_index * self.per_veh_info_dim: (veh_index + 1) * self.per_veh_info_dim]
            veh_x = vehs[:, 0]
            veh_y = vehs[:, 1]
            veh_v_x = vehs[:, 2]
            dif_x_ego2veh = veh_x - ego_x
            dif_y_ego2veh = veh_y - ego_y
            rel_x_along_path_ego2veh = \
                torch.mul(dif_x_ego2veh, torch.cos(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2veh, torch.sin(phi_path * pi / 180.0))
            rel_y_along_path_ego2veh = \
                - torch.mul(dif_x_ego2veh, torch.sin(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2veh, torch.cos(phi_path * pi / 180.0))

            flag_long = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_lat_gap - torch.abs(rel_y_along_path_ego2veh))))
            veh2veh4training += 1.0 * torch.where(
                rel_x_along_path_ego2veh >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2veh - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2veh - min_long_gap, veh_v_x + EPSILON_V))),
            )
            veh2veh4real += 1.0 * torch.where(
                rel_x_along_path_ego2veh >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2veh - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2veh - min_long_gap, veh_v_x + EPSILON_V))),
            )

            flag_lat = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_long_gap - torch.abs(rel_x_along_path_ego2veh))))
            veh2veh4training += 1.0 * torch.where(
                (rel_y_along_path_ego2veh >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2veh - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2veh4training += 1.0 * torch.where(
                (rel_y_along_path_ego2veh < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2veh + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2veh4real += 1.0 * torch.where(
                (rel_y_along_path_ego2veh >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2veh - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2veh4real += 1.0 * torch.where(
                (rel_y_along_path_ego2veh < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2veh + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )

        veh2bike4real = torch.zeros_like(veh_infos[:, 0])
        veh2bike4training = torch.zeros_like(veh_infos[:, 0])
        for bike_index in range(int(bike_infos.shape[1] / self.per_bike_info_dim)):
            min_long_gap = (L + L_BIKE) / 2 + safe_gap
            min_lat_gap = (W + W_BIKE) / 2 + safe_gap
            bikes = bike_infos[:, bike_index * self.per_bike_info_dim: (bike_index + 1) * self.per_bike_info_dim]
            bike_x = bikes[:, 0]
            bike_y = bikes[:, 1]
            bike_v_x = bikes[:, 2]
            dif_x_ego2bike = bike_x - ego_x
            dif_y_ego2bike = bike_y - ego_y
            rel_x_along_path_ego2bike = \
                torch.mul(dif_x_ego2bike, torch.cos(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2bike, torch.sin(phi_path * pi / 180.0))
            rel_y_along_path_ego2bike = \
                - torch.mul(dif_x_ego2bike, torch.sin(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2bike, torch.cos(phi_path * pi / 180.0))

            flag_long = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_lat_gap - torch.abs(rel_y_along_path_ego2bike))))
            veh2bike4training += 1.0 * torch.where(
                rel_x_along_path_ego2bike >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2bike - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2bike - min_long_gap, bike_v_x + EPSILON_V))),
            )
            veh2bike4real += 1.0 * torch.where(
                rel_x_along_path_ego2bike >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2bike - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2bike - min_long_gap, bike_v_x + EPSILON_V))),
            )

            flag_lat = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_long_gap - torch.abs(rel_x_along_path_ego2bike))))
            veh2bike4training += 1.0 * torch.where(
                (rel_y_along_path_ego2bike >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2bike - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2bike4training += 1.0 * torch.where(
                (rel_y_along_path_ego2bike < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2bike + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2bike4real += 1.0 * torch.where(
                (rel_y_along_path_ego2bike >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2bike - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2bike4real += 1.0 * torch.where(
                (rel_y_along_path_ego2bike < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2bike + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )

        veh2person4real = torch.zeros_like(veh_infos[:, 0])
        veh2person4training = torch.zeros_like(veh_infos[:, 0])
        for person_index in range(int(person_infos.shape[1] / self.per_person_info_dim)):
            min_long_gap = (L + 0) / 2 + safe_gap
            min_lat_gap = (W + 0) / 2 + safe_gap
            persons = person_infos[:, person_index * self.per_person_info_dim: (person_index + 1) * self.per_person_info_dim]
            person_x = persons[:, 0]
            person_y = persons[:, 1]
            person_v_x = persons[:, 2]
            dif_x_ego2person = person_x - ego_x
            dif_y_ego2person = person_y - ego_y
            rel_x_along_path_ego2person = \
                torch.mul(dif_x_ego2person, torch.cos(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2person, torch.sin(phi_path * pi / 180.0))
            rel_y_along_path_ego2person = \
                - torch.mul(dif_x_ego2person, torch.sin(phi_path * pi / 180.0)) \
                + torch.mul(dif_y_ego2person, torch.cos(phi_path * pi / 180.0))

            flag_long = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_lat_gap - torch.abs(rel_y_along_path_ego2person))))
            veh2person4training += 1.0 * torch.where(
                rel_x_along_path_ego2person >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2person - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2person - min_long_gap, person_v_x + EPSILON_V))),
            )
            veh2person4real += 1.0 * torch.where(
                rel_x_along_path_ego2person >= 0,
                1.0 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(rel_x_along_path_ego2person - min_long_gap, ego_v_x + EPSILON_V))),
                0.6 * torch.mul(flag_long, 1.0 - torch.tanh(
                    1.0 * torch.div(-rel_x_along_path_ego2person - min_long_gap, person_v_x + EPSILON_V))),
            )

            flag_lat = torch.div(1.0, 1.0 + torch.exp(-10.0 * (min_long_gap - torch.abs(rel_x_along_path_ego2person))))
            veh2person4training += 1.0 * torch.where(
                (rel_y_along_path_ego2person >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2person - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2person4training += 1.0 * torch.where(
                (rel_y_along_path_ego2person < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2person + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2person4real += 1.0 * torch.where(
                (rel_y_along_path_ego2person >= 0) & (sin4phi2path >= 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2person - min_lat_gap, torch.mul(ego_v_x, sin4phi2path) + EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )
            veh2person4real += 1.0 * torch.where(
                (rel_y_along_path_ego2person < 0) & (sin4phi2path < 0),
                1.0 * torch.mul(flag_lat, 1.0 - torch.tanh(
                    1.5 * torch.div(rel_y_along_path_ego2person + min_lat_gap, torch.mul(ego_v_x, sin4phi2path) - EPSILON_V))),
                torch.zeros_like(veh_infos[:, 0]),
            )

        veh2road4real = torch.zeros_like(veh_infos[:, 0])
        veh2road4training = torch.zeros_like(veh_infos[:, 0])  # constraint on static path
        """
        for ego_point in [ego_front_points, ego_rear_points]:
            veh2road4training += torch.where((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                             (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180)
                                             < -LANE_WIDTH / 2 + 1.0,
                                             torch.square((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                                          (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180) -
                                                          (-LANE_WIDTH / 2 + 1.0)),
                                             torch.zeros_like(veh_infos[:, 0])) * 0.01
            veh2road4training += torch.where((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                             (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180)
                                             > LANE_WIDTH / 2 - 1.0,
                                             torch.square((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                                          (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180) -
                                                          (LANE_WIDTH / 2 - 1.0)),
                                             torch.zeros_like(veh_infos[:, 0])) * 0.01
            veh2road4real += torch.where((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                         (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180)
                                         < -LANE_WIDTH / 2 + 1.0,
                                         torch.square((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                                      (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180) -
                                                      (-LANE_WIDTH / 2 + 1.0)),
                                         torch.zeros_like(veh_infos[:, 0])) * 0.01
            veh2road4real += torch.where((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                         (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180)
                                         > LANE_WIDTH / 2 - 1.0,
                                         torch.square((ego_point[0] - closest_x_path) * torch.cos((phi_path + 90) * pi / 180) +
                                                      (ego_point[1] - closest_y_path) * torch.sin((phi_path + 90) * pi / 180) -
                                                      (LANE_WIDTH / 2 - 1.0)),
                                         torch.zeros_like(veh_infos[:, 0])) * 0.01
        """

        rewards = (
                0.1 * devi_v + 2.5 * devi_y + 30.0 * devi_phi +
                2.0 * punish_yaw_rate + 5.0 * punish_steer + 0.5 * punish_a_x
        )
        punish_term_for_training = veh2veh4training + veh2road4training + veh2bike4training + veh2person4training
        real_punish_term = veh2veh4real + veh2road4real + veh2bike4real + veh2person4real

        reward_dict = dict(
            punish_steer=punish_steer,
            punish_a_x=punish_a_x,
            punish_yaw_rate=punish_yaw_rate,
            devi_v=devi_v,
            devi_y=devi_y,
            devi_phi=devi_phi,
            scaled_devi_v=0.1 * devi_v,
            scaled_devi_y=2.5 * devi_y,
            scaled_devi_phi=30.0 * devi_phi,
            scaled_punish_yaw_rate=2.0 * punish_yaw_rate,
            scaled_punish_steer=5.0 * punish_steer,
            scaled_punish_a_x=0.5 * punish_a_x,
            veh2veh4training=veh2veh4training,
            veh2road4training=veh2road4training,
            veh2bike4training=veh2bike4training,
            veh2person4training=veh2person4training,
            veh2veh4real=veh2veh4real,
            veh2road4real=veh2road4real,
            veh2bike2real=veh2bike4real,
            veh2person2real=veh2person4real,
        )

        return (
            rewards,
            punish_term_for_training,
            real_punish_term,
            veh2veh4real,
            veh2road4real,
            veh2bike4real,
            veh2person4real,
            reward_dict,
        )


class CrossingEnvModel(_EnvironmentModel):
    def compute_rewards(self, obses_ego, obses_bike, obses_person, obses_veh, actions):
        # obses = self.convert_vehs_to_abso(obses)
        if isinstance(obses_ego, np.ndarray):
            obses_ego = torch.tensor(obses_ego)
        if isinstance(obses_bike, np.ndarray):
            obses_bike = torch.tensor(obses_bike)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_person, np.ndarray):
            obses_person = torch.tensor(obses_person)
        if isinstance(obses_veh, np.ndarray):
            obses_veh = torch.tensor(obses_veh)
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions)

        ego_infos = obses_ego[:, : self.ego_info_dim]
        tracking_infos = torch.zeros_like(obses_ego[:, self.ego_info_dim:])
        tracking_infos = obses_ego[:, self.ego_info_dim:]

        bike_infos = obses_bike.detach()
        person_infos = obses_person.detach()
        veh_infos = obses_veh.detach()

        steers, a_xs = actions[:, 0], actions[:, 1]
        # rewards related to action
        punish_steer = -torch.square(steers)
        punish_a_x = -torch.square(a_xs)

        # rewards related to ego stability
        punish_yaw_rate = -torch.square(ego_infos[:, 2])

        # rewards related to tracking error
        devi_y = -torch.square(tracking_infos[:, 0])
        devi_phi = -torch.square(tracking_infos[:, 1] * np.pi / 180.0)
        devi_v = -torch.square(tracking_infos[:, 2])

        # rewards related to veh2veh collision
        ego_lws = (L - W) / 2.0
        ego_front_points = (
            ego_infos[:, 3] + ego_lws * torch.cos(ego_infos[:, 5] * np.pi / 180.0),
            ego_infos[:, 4] + ego_lws * torch.sin(ego_infos[:, 5] * np.pi / 180.0),
        )
        ego_rear_points = (
            ego_infos[:, 3] - ego_lws * torch.cos(ego_infos[:, 5] * np.pi / 180.0),
            ego_infos[:, 4] - ego_lws * torch.sin(ego_infos[:, 5] * np.pi / 180.0),
        )
        veh2veh4real = torch.zeros_like(veh_infos[:, 0])
        veh2veh4training = torch.zeros_like(veh_infos[:, 0])

        for veh_index in range(int(veh_infos.shape[1] / self.per_veh_info_dim)):
            vehs = veh_infos[
                   :,
                   veh_index * self.per_veh_info_dim: (veh_index + 1) * self.per_veh_info_dim,
                   ]
            veh_lws = (L - W) / 2.0
            veh_front_points = (
                vehs[:, 0] + veh_lws * torch.cos(vehs[:, 3] * np.pi / 180.0),
                vehs[:, 1] + veh_lws * torch.sin(vehs[:, 3] * np.pi / 180.0),
            )
            veh_rear_points = (
                vehs[:, 0] - veh_lws * torch.cos(vehs[:, 3] * np.pi / 180.0),
                vehs[:, 1] - veh_lws * torch.sin(vehs[:, 3] * np.pi / 180.0),
            )
            for ego_point in [ego_front_points, ego_rear_points]:
                for veh_point in [veh_front_points, veh_rear_points]:
                    veh2veh_dist = torch.sqrt(
                        torch.square(ego_point[0] - veh_point[0]) + torch.square(ego_point[1] - veh_point[1])
                    )
                    veh2veh4training += torch.where(
                        veh2veh_dist - 4.5 < 0,
                        torch.square(veh2veh_dist - 4.5),
                        torch.zeros_like(veh_infos[:, 0]),
                    )
                    veh2veh4real += torch.where(
                        veh2veh_dist - 2.5 < 0,
                        torch.square(veh2veh_dist - 2.5),
                        torch.zeros_like(veh_infos[:, 0]),
                    )

        veh2bike4real = torch.zeros_like(veh_infos[:, 0])
        veh2bike4training = torch.zeros_like(veh_infos[:, 0])
        for bike_index in range(int(bike_infos.shape[1] / self.per_bike_info_dim)):
            bikes = bike_infos[:, bike_index * self.per_bike_info_dim: (bike_index + 1) * self.per_bike_info_dim]
            bike_lws = (L_BIKE - W_BIKE) / 2.0
            bike_front_points = bikes[:, 0] + bike_lws * torch.cos(bikes[:, 3] * np.pi / 180.0), bikes[
                                                                                                 :, 1
                                                                                                 ] + bike_lws * torch.sin(
                bikes[:, 3] * np.pi / 180.0)
            bike_rear_points = bikes[:, 0] - bike_lws * torch.cos(bikes[:, 3] * np.pi / 180.0), bikes[
                                                                                                :, 1
                                                                                                ] - bike_lws * torch.sin(
                bikes[:, 3] * np.pi / 180.0)

            for ego_point in [ego_front_points, ego_rear_points]:
                for bike_point in [bike_front_points, bike_rear_points]:
                    veh2bike_dist = torch.sqrt(
                        torch.square(ego_point[0] - bike_point[0]) + torch.square(ego_point[1] - bike_point[1])
                    )
                    veh2bike4training += torch.where(
                        veh2bike_dist - 4.5 < 0,
                        torch.square(veh2bike_dist - 4.5),
                        torch.zeros_like(veh_infos[:, 0]),
                    )
                    veh2bike4real += torch.where(
                        veh2bike_dist - 2.5 < 0,
                        torch.square(veh2bike_dist - 2.5),
                        torch.zeros_like(veh_infos[:, 0]),
                    )

        veh2person4real = torch.zeros_like(veh_infos[:, 0])
        veh2person4training = torch.zeros_like(veh_infos[:, 0])
        for person_index in range(int(person_infos.shape[1] / self.per_person_info_dim)):
            persons = person_infos[
                      :,
                      person_index * self.per_person_info_dim: (person_index + 1) * self.per_person_info_dim,
                      ]
            person_point = persons[:, 0], persons[:, 1]
            for ego_point in [ego_front_points, ego_rear_points]:
                veh2person_dist = torch.sqrt(
                    torch.square(ego_point[0] - person_point[0]) + torch.square(ego_point[1] - person_point[1])
                )
                veh2person4training += torch.where(
                    veh2person_dist - 4.5 < 0,
                    torch.square(veh2person_dist - 4.5),
                    torch.zeros_like(veh_infos[:, 0]),
                )
                veh2person4real += torch.where(
                    veh2person_dist - 2.5 < 0,
                    torch.square(veh2person_dist - 2.5),
                    torch.zeros_like(veh_infos[:, 0]),
                )

        veh2road4real = torch.zeros_like(veh_infos[:, 0])
        veh2road4training = torch.zeros_like(veh_infos[:, 0])  # TODO: 添加道路约束惩罚
        # if self.task == 'left':
        #     for ego_point in [ego_front_points, ego_rear_points]:
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, ego_point[0] < 1),
        #                              torch.square(ego_point[0]-1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, LANE_WIDTH-ego_point[0] < 1),
        #                              torch.square(LANE_WIDTH-ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[0] < 0, LANE_WIDTH*LANE_NUMBER - ego_point[1] < 1),
        #                              torch.square(LANE_WIDTH*LANE_NUMBER - ego_point[1] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[0] < -CROSSROAD_SIZE/2, ego_point[1] - 0 < 1),
        #                              torch.square(ego_point[1] - 0 - 1), torch.zeros_like(veh_infos[:, 0]))
        #
        #         veh2road4real += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, ego_point[0] < 1),
        #                              torch.square(ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, LANE_WIDTH - ego_point[0] < 1),
        #                              torch.square(LANE_WIDTH - ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(logical_and(ego_point[0] < -CROSSROAD_SIZE/2, LANE_WIDTH*LANE_NUMBER - ego_point[1] < 1),
        #                              torch.square(LANE_WIDTH*LANE_NUMBER - ego_point[1] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(logical_and(ego_point[0] < -CROSSROAD_SIZE/2, ego_point[1] - 0 < 1),
        #                              torch.square(ego_point[1] - 0 - 1), torch.zeros_like(veh_infos[:, 0]))
        # elif self.task == 'straight':
        #     for ego_point in [ego_front_points, ego_rear_points]:
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, ego_point[0] - LANE_WIDTH < 1),
        #                              torch.square(ego_point[0] - LANE_WIDTH -1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, 2*LANE_WIDTH-ego_point[0] < 1),
        #                              torch.square(2*LANE_WIDTH-ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[1] > CROSSROAD_SIZE/2, LANE_WIDTH*LANE_NUMBER - ego_point[0] < 1),
        #                              torch.square(LANE_WIDTH*LANE_NUMBER - ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[1] > CROSSROAD_SIZE/2, ego_point[0] - 0 < 1),
        #                              torch.square(ego_point[0] - 0 - 1), torch.zeros_like(veh_infos[:, 0]))
        #
        #         veh2road4real += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0]-LANE_WIDTH < 1),
        #                                       torch.square(ego_point[0]-LANE_WIDTH - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(
        #             logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, 2 * LANE_WIDTH - ego_point[0] < 1),
        #             torch.square(2 * LANE_WIDTH - ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(
        #             logical_and(ego_point[1] > CROSSROAD_SIZE / 2, LANE_WIDTH * LANE_NUMBER - ego_point[0] < 1),
        #             torch.square(LANE_WIDTH * LANE_NUMBER - ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(logical_and(ego_point[1] > CROSSROAD_SIZE / 2, ego_point[0] - 0 < 1),
        #                                       torch.square(ego_point[0] - 0 - 1), torch.zeros_like(veh_infos[:, 0]))
        # else:
        #     assert self.task == 'right'
        #     for ego_point in [ego_front_points, ego_rear_points]:
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, ego_point[0] - 2*LANE_WIDTH < 1),
        #                              torch.square(ego_point[0] - 2*LANE_WIDTH-1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[1] < -CROSSROAD_SIZE/2, LANE_NUMBER*LANE_WIDTH-ego_point[0] < 1),
        #                              torch.square(LANE_NUMBER*LANE_WIDTH-ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[0] > CROSSROAD_SIZE/2, 0 - ego_point[1] < 1),
        #                              torch.square(0 - ego_point[1] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4training += torch.where(logical_and(ego_point[0] > CROSSROAD_SIZE/2, ego_point[1] - (-LANE_WIDTH*LANE_NUMBER) < 1),
        #                              torch.square(ego_point[1] - (-LANE_WIDTH*LANE_NUMBER) - 1), torch.zeros_like(veh_infos[:, 0]))
        #
        #         veh2road4real += torch.where(
        #             logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, ego_point[0] - 2 * LANE_WIDTH < 1),
        #             torch.square(ego_point[0] - 2 * LANE_WIDTH - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(
        #             logical_and(ego_point[1] < -CROSSROAD_SIZE / 2, LANE_NUMBER * LANE_WIDTH - ego_point[0] < 1),
        #             torch.square(LANE_NUMBER * LANE_WIDTH - ego_point[0] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(logical_and(ego_point[0] > CROSSROAD_SIZE / 2, 0 - ego_point[1] < 1),
        #                                       torch.square(0 - ego_point[1] - 1), torch.zeros_like(veh_infos[:, 0]))
        #         veh2road4real += torch.where(
        #             logical_and(ego_point[0] > CROSSROAD_SIZE / 2, ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) < 1),
        #             torch.square(ego_point[1] - (-LANE_WIDTH * LANE_NUMBER) - 1), torch.zeros_like(veh_infos[:, 0]))

        rewards = (
                0.01 * devi_v + 0.8 * devi_y + 30 * devi_phi + 0.02 * punish_yaw_rate + 5 * punish_steer + 0.01 * punish_a_x
        )
        punish_term_for_training = veh2veh4training + veh2road4training + veh2bike4training + veh2person4training
        real_punish_term = veh2veh4real + veh2road4real + veh2bike4real + veh2person4real

        reward_dict = dict(
            punish_steer=punish_steer,
            punish_a_x=punish_a_x,
            punish_yaw_rate=punish_yaw_rate,
            devi_v=devi_v,
            devi_y=devi_y,
            devi_phi=devi_phi,
            scaled_punish_steer=5 * punish_steer,
            scaled_punish_a_x=0.01 * punish_a_x,
            scaled_punish_yaw_rate=0.02 * punish_yaw_rate,
            scaled_devi_v=0.01 * devi_v,
            scaled_devi_y=0.8 * devi_y,
            scaled_devi_phi=30 * devi_phi,
            veh2veh4training=veh2veh4training,
            veh2road4training=veh2road4training,
            veh2bike4training=veh2bike4training,
            veh2person4training=veh2person4training,
            veh2veh4real=veh2veh4real,
            veh2road4real=veh2road4real,
            veh2bike2real=veh2bike4real,
            veh2person2real=veh2person4real,
        )

        return (
            rewards,
            punish_term_for_training,
            real_punish_term,
            veh2veh4real,
            veh2road4real,
            veh2bike4real,
            veh2person4real,
            reward_dict,
        )


def environment_model(training_task4, num_future_data=0, mode="training", expected_v=None, **kwargs):
    if training_task4[0] == "edge":
        return MultiLaneEnvModel(training_task4, num_future_data, mode, expected_v, **kwargs)
    elif training_task4[0] == "junction":
        return CrossingEnvModel(training_task4, num_future_data, mode, expected_v, **kwargs)
    else:
        raise ValueError("Not set joe properly")


def dist_ego2path(xs, ys, x_path, y_path, heading_path):
    dist = torch.div(
        torch.abs(torch.mul(torch.tan(heading_path * pi / 180), xs - x_path) - (ys - y_path)),
        torch.sqrt(torch.square(torch.tan(heading_path * pi / 180)) + 1.0),
    )
    return dist


def deal_with_phi_diff(phi_diff):
    phi_diff = torch.where(phi_diff > 180.0, phi_diff - 360.0, phi_diff)
    phi_diff = torch.where(phi_diff < -180.0, phi_diff + 360.0, phi_diff)
    return phi_diff


class ReferencePath(object):
    def __init__(self, training_task4, ref_index=None, expected_v=None, **kwargs):
        self.training_task = training_task4  # tuple ("edge_or_junction", edge, from_edge, to_edge)

        self.joe, self.edge, self.from_edge, self.to_edge = training_task4
        self.exp_v = expected_v
        assert self.exp_v is not None

        self.path_list = []
        self.path_len_list = []
        self.control_points = []
        self.path_joint_index = []

        self._construct_ref_path(self.joe, self.edge, self.from_edge, self.to_edge)
        self.ref_index = np.random.choice(len(self.path_list)) if ref_index is None else ref_index
        self.path = self.path_list[self.ref_index]
        self.adjacent_path = None
        if self.joe == "junction":
            self.joint_idx = self.path_joint_index[self.ref_index]

    @property
    def target_points(self):
        if self.joe == "junction":
            xr, yr, ar = self.path
            return xr[-900], yr[-900]
        else:
            xr, yr, ar = self.path
            return xr[-1], yr[-1], ar[-1]

    def set_path(self, path_index=None):
        self.ref_index = path_index
        self.path = self.path_list[self.ref_index]
        if self.joe == "junction":
            self.joint_idx = self.path_joint_index[self.ref_index]

    def _construct_ref_path(self, type_, edge=None, from_edge=None, end_edge=None):
        """生成静态参考轨迹"""
        meter_pointnum_ratio = 30
        IO_LENGTH = 60

        def deal_with_lane_in_edge(edge_ID, lane_ID):
            lane_shape = map_info[edge_ID][lane_ID]["shape"]
            assert len(lane_shape) == 2, "lane in edge length longer than 2"
            start_point, end_point = lane_shape
            lane_length = np.sqrt(np.square(start_point[0] - end_point[0]) + np.square(start_point[1] - end_point[1]))
            node = np.asfortranarray(
                [[start_point[0], end_point[0]], [start_point[1], end_point[1]]],
                dtype=np.float32,
            )
            curve = bezier.Curve(node, degree=1)
            s_vals = np.linspace(0, 1.0, int(lane_length * meter_pointnum_ratio))
            trj_data = curve.evaluate_multi(s_vals)
            trj_data = trj_data.astype(np.float32)
            return trj_data

        def deal_with_lane_in_junction(junction_ID, from_lane, to_lane):
            control_point1 = map_info[junction_ID]["{}_{}".format(from_lane, to_lane)]["control_point1"]
            control_point2 = map_info[junction_ID]["{}_{}".format(from_lane, to_lane)]["control_point2"]
            control_point3 = map_info[junction_ID]["{}_{}".format(from_lane, to_lane)]["control_point3"]
            control_point4 = map_info[junction_ID]["{}_{}".format(from_lane, to_lane)]["control_point4"]

            lane_length = (
                np.sqrt(
                    np.square(control_point1[0] - control_point4[0]) + np.square(control_point1[1] - control_point4[1])
                )
                * 1.11
            )  # pi * 1.414 * 0.25 \approx 1.11
            node = np.asfortranarray(
                [
                    [
                        control_point1[0],
                        control_point2[0],
                        control_point3[0],
                        control_point4[0],
                    ],
                    [
                        control_point1[1],
                        control_point2[1],
                        control_point3[1],
                        control_point4[1],
                    ],
                ],
                dtype=np.float32,
            )
            curve = bezier.Curve(node, degree=3)
            s_vals = np.linspace(0, 1.0, int(lane_length * meter_pointnum_ratio))
            trj_data = curve.evaluate_multi(s_vals)
            trj_data = trj_data.astype(np.float32)
            return trj_data

        def cut_start_point(start_data):
            data_length = start_data.shape[1]
            real_length = min(data_length, IO_LENGTH * meter_pointnum_ratio)
            return start_data[:, -real_length:]

        def cut_end_point(end_data):
            data_length = end_data.shape[1]
            real_length = min(data_length, IO_LENGTH * meter_pointnum_ratio)
            return end_data[:, :real_length]

        if type_ == "edge":
            assert edge is not None, "when type == 'edge', edge must not be None"
            path_list_xy = []
            lane_list = MLANE_TASKS[edge]["lane_list"]
            for l in lane_list:
                path_list_xy.append(deal_with_lane_in_edge(edge, l))  # 根据起点和终点，对道路进行离散求解
        elif type_ == "junction":
            sub_routes = task2staticpath[from_edge]
            path_list_xy = []

            for sr in sub_routes:
                data_list = []
                for ind, (cur_e, cur_l) in enumerate(sr[:-1]):
                    succ_e, succ_l = sr[ind + 1]
                    lane_path = deal_with_lane_in_edge(cur_e, cur_l)
                    joint_path = deal_with_lane_in_junction(ROUTE_JUNCTION[cur_e][0], cur_l, succ_l)

                    data_list.append(lane_path)
                    data_list.append(joint_path)

                cur_e, cur_l = sr[-1]
                lane_path = deal_with_lane_in_edge(cur_e, cur_l)
                data_list.append(lane_path)

                # cut start and end
                data_list[0] = cut_start_point(data_list[0])
                data_list[-1] = cut_end_point(data_list[-1])

                data_len = [kk.shape[1] for kk in data_list]

                S = 0
                data_len_acc = []
                for l in data_len:
                    S += l
                    data_len_acc.append(S)

                self.path_joint_index.append(data_len_acc)
                data_cat = np.concatenate(data_list, 1)
                path_list_xy.append(data_cat)

            #     start_lanes = map_info[from_edge]["lane_list"]
            #
            # pathList, stack = [], [(l, from_edge, 0, []) for l in start_lanes]  # (lane to add, edge,subpathlength, subpath)
            #
            # while stack:
            #     cur_l, cur_e, cur_sub_num, path_sub = stack.pop(0)
            #     succ_edges = map_info[cur_e][cur_l]['to_edge']
            #     succ_lanes = map_info[cur_e][cur_l]['to_lane']
            #     for e, l in zip(succ_edges, succ_lanes):
            #         if (not e) or (not l) or (cur_sub_num >= 1) or (e[0] != 'E') or e not in EGO_ROUTE:
            #             pathList.append(path_sub + [deal_with_lane_in_edge(cur_e, cur_l)])
            #         else:
            #             lane_path = deal_with_lane_in_edge(cur_e, cur_l)
            #             joint_path = deal_with_lane_in_junction(ROUTE_JUNCTION[cur_e][0], cur_l, l)
            #             stack.append((l, e, cur_sub_num + 1, path_sub + [lane_path, joint_path]))
            # path_list_xy = [np.concatenate(path, 1) for path in pathList if len(path) == 3]
        assert len(path_list_xy) > 0

        for trj_data_xy in path_list_xy:
            xs_1, ys_1 = trj_data_xy[0][:-1], trj_data_xy[1][:-1]
            xs_2, ys_2 = trj_data_xy[0][1:], trj_data_xy[1][1:]
            phis_1 = np.arctan2(ys_2 - ys_1, xs_2 - xs_1) * 180 / pi  # 计算道路朝向角
            planed_trj = xs_1, ys_1, phis_1
            self.path_list.append(planed_trj)
            self.path_len_list.append((len(trj_data_xy[0]), len(xs_1)))

    def find_closest_point(self, xs, ys, ratio=10):
        if isinstance(xs, np.ndarray):
            xs = torch.tensor(xs)
        if isinstance(ys, np.ndarray):
            ys = torch.tensor(ys)
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)
        reduced_len = len(reduced_idx)
        reduced_path_x, reduced_path_y = (
            self.path[0][reduced_idx],
            self.path[1][reduced_idx],
        )
        reduced_path_x = torch.tensor(reduced_path_x)
        reduced_path_y = torch.tensor(reduced_path_y)
        xs_tile = xs.reshape(-1, 1).repeat(1, reduced_len)
        ys_tile = ys.reshape(-1, 1).repeat(1, reduced_len)

        pathx_tile = reduced_path_x.reshape(1, -1).repeat(len(xs), 1)
        pathy_tile = reduced_path_y.reshape(1, -1).repeat(len(xs), 1)

        dist_array = torch.square(xs_tile - pathx_tile) + torch.square(ys_tile - pathy_tile)

        indexs = torch.argmin(dist_array, 1) * ratio
        return indexs, self.indexs2points(indexs)

    def find_closest_point4junction(self, xs, ys, ratio=10):
        path_len = len(self.path[0])
        reduced_idx = np.arange(0, path_len, ratio)

        reduced_path_x, reduced_path_y = (
            self.path[0][reduced_idx],
            self.path[1][reduced_idx],
        )

        dist2_array = (reduced_path_x - xs) * (reduced_path_x - xs)  + (reduced_path_y - ys) * (reduced_path_y - ys)
        index = np.argmin(dist2_array) * ratio
        point = self.path[0][index], self.path[1][index], self.path[2][index]
        return index, point

    def find_closest_point4edge(self, xs, ys, gap=6):
        if isinstance(xs, np.ndarray):
            xs = torch.tensor(xs)
        if isinstance(ys, np.ndarray):
            ys = torch.tensor(ys)

        path_len = len(self.path[0])
        path_x = torch.tensor(self.path[0])
        path_y = torch.tensor(self.path[1])

        first = 0
        last = path_len - 1
        while last - first > gap:
            mid = (last + first) // 2
            if torch.square(xs - path_x[last]) + torch.square(ys - path_y[last]) >= torch.square(
                xs - path_x[first]
            ) + torch.square(ys - path_y[first]):
                last = mid
            else:
                first = mid

        first_x, first_y = path_x[first], path_y[first]
        last_x, last_y = path_x[last], path_y[last]
        delta_x, delta_y = last_x - first_x, last_y - first_y

        if delta_x >= 0:
            phi_path = torch.arctan(delta_y / (delta_x + 1e-6))
        else:
            phi_path = torch.arctan(delta_y / (delta_x - 1e-6)) + pi
        if phi_path >= pi:
            phi_path = phi_path - 2 * pi
        elif phi_path < -pi:
            phi_path = phi_path + 2 * pi
        phi_path = torch.tensor([phi_path * 180 / pi])

        # 一般式直线方程：Ax + By + C = 0
        A = last_y - first_y
        B = -(last_x - first_x)
        C = -first_x * (last_y - first_y) + first_y * (last_x - first_x)

        # 垂足
        closest_x_path = (B * B * xs - A * B * ys - A * C) / (A * A + B * B)
        closest_y_path = (-A * B * xs + A * A * ys - B * C) / (A * A + B * B)
        # print(f'x_path = {closest_x_path}, y_path = {closest_y_path}, phi_path = {phi_path}')

        return torch.stack([closest_x_path, closest_y_path, phi_path], dim=1)

    def future_n_data(self, current_indexs, n):
        future_data_list = []
        # current_indexs = tf.cast(current_indexs, tf.int32)
        for _ in range(n):
            current_indexs += 80
            current_indexs = torch.where(
                current_indexs >= len(self.path[0]) - 2,
                len(self.path[0]) - 2,
                current_indexs,
            )
            future_data_list.append(self.indexs2points(current_indexs))
        return future_data_list

    def indexs2points(self, indexs):
        """初始化自车位置，使用index获得x,y,phi
        input numpy.array
        ouput torch.Tensor
        """
        indexs = torch.tensor(indexs)
        indexs = torch.where(torch.tensor(indexs >= 0), indexs, torch.tensor(0))
        indexs = torch.where(torch.tensor(indexs < len(self.path[0])), indexs, torch.tensor(len(self.path[0]) - 1))
        indexs = torch.tensor(indexs, dtype=torch.int64)
        points = (
            torch.gather(torch.Tensor(self.path[0]), 0, indexs),
            torch.gather(torch.Tensor(self.path[1]), 0, indexs),
            torch.gather(torch.Tensor(self.path[2]), 0, indexs),
        )

        return points[0], points[1], points[2]

    def indexs2adj_path(self, indexs):
        """初始化自车位置，使用index获得x,y,phi
        input numpy.array
        ouput torch.Tensor
        """
        indexs = torch.tensor(indexs)
        indexs = torch.where(torch.tensor(indexs >= 0), indexs, torch.tensor(0))
        indexs = torch.where(torch.tensor(indexs < len(self.adjacent_path[0])), indexs, torch.tensor(len(self.path[0]) - 1))
        indexs = torch.tensor(indexs, dtype=torch.int64)
        points = (
            torch.gather(torch.Tensor(self.adjacent_path[0]), 0, indexs),
            torch.gather(torch.Tensor(self.adjacent_path[1]), 0, indexs),
            torch.gather(torch.Tensor(self.adjacent_path[2]), 0, indexs),
        )

        return points[0], points[1], points[2]

    def tracking_error_vector(self, ego_xs, ego_ys, ego_phis, ego_vs, n):
        """
        input numpy.array
        output tensor
        Parameters
        ----------
        ego_xs
        ego_ys
        ego_phis
        ego_vs
        n

        Returns
        -------

        """

        if isinstance(ego_xs, np.ndarray):
            ego_xs = torch.tensor(ego_xs)
        if isinstance(ego_ys, np.ndarray):
            ego_ys = torch.tensor(ego_ys)
        if isinstance(ego_phis, np.ndarray):
            ego_phis = torch.tensor(ego_phis)
        if isinstance(ego_vs, np.ndarray):
            ego_vs = torch.Tensor(ego_vs)

        def two2one(ref_xs, ref_ys):
            # if self.task == 'left':
            #     delta_ = torch.sqrt(torch.square(ego_xs - (-CROSSROAD_SIZE/2)) + torch.square(ego_ys - (-CROSSROAD_SIZE/2))) - \
            #              torch.sqrt(torch.square(ref_xs - (-CROSSROAD_SIZE/2)) + torch.square(ref_ys - (-CROSSROAD_SIZE/2)))
            #     delta_ = torch.where(ego_ys < -CROSSROAD_SIZE/2, ego_xs - ref_xs, delta_)
            #     delta_ = torch.where(ego_xs < -CROSSROAD_SIZE/2, ego_ys - ref_ys, delta_)
            #     return -delta_
            # elif self.task == 'straight':
            #     delta_ = ego_xs - ref_xs
            #     return -delta_
            # else:
            #     assert self.task == 'right'
            #     delta_ = -(torch.sqrt(torch.square(ego_xs - CROSSROAD_SIZE/2) + torch.square(ego_ys - (-CROSSROAD_SIZE/2))) -
            #                torch.sqrt(torch.square(ref_xs - CROSSROAD_SIZE/2) + torch.square(ref_ys - (-CROSSROAD_SIZE/2))))
            #     delta_ = torch.where(ego_ys < -CROSSROAD_SIZE/2, ego_xs - ref_xs, delta_)
            #     delta_ = torch.where(ego_xs > CROSSROAD_SIZE/2, -(ego_ys - ref_ys), delta_)
            #     return -delta_
            dx = ego_xs - ref_xs
            dy = ego_ys - ref_ys
            delta_ = torch.sqrt(torch.square(dx) + torch.square(dy))
            return delta_

        indexs, current_points = self.find_closest_point(ego_xs, ego_ys)

        n_future_data = self.future_n_data(indexs, n)

        tracking_error = torch.stack(
            [
                two2one(current_points[0], current_points[1]),
                deal_with_phi_diff(ego_phis - current_points[2]),
                ego_vs - self.exp_v,
            ],
            1,
        )

        final = tracking_error
        if n > 0:
            future_points = torch.cat(
                [
                    torch.stack(
                        [
                            ref_point[0] - ego_xs,
                            ref_point[1] - ego_ys,
                            deal_with_phi_diff(ego_phis - ref_point[2]),
                        ],
                        1,
                    )
                    for ref_point in n_future_data
                ],
                1,
            )
            final = torch.cat([final, future_points], 1)

        return final

    def plot_path(self, x, y):
        plt.axis("equal")
        plt.plot(self.path_list[0][0], self.path_list[0][1], "b")
        plt.plot(self.path_list[1][0], self.path_list[1][1], "r")
        plt.plot(self.path_list[2][0], self.path_list[2][1], "g")

        index, closest_point = self.find_closest_point(np.array([x], np.float32), np.array([y], np.float32))
        plt.plot(x, y, "b*")
        plt.plot(closest_point[0], closest_point[1], "ro")
        plt.show()

    def plot_path2(self):
        plt.axis("equal")
        for i, path in enumerate(self.path_list):
            plt.plot(path[0], path[1], "o")

            idx = self.path_joint_index[i]

            for j in idx:
                plt.plot(path[0][j - 2], path[1][j - 2], "o")

        plt.show()

    def solve_dense(self):
        for i, path in enumerate(self.path_list):
            idx = self.path_joint_index[i]

            p1 = path[0][: idx[0]], path[1][: idx[0]]

            p2 = path[0][idx[0] : idx[1]], path[1][idx[0] : idx[1]]

            p3 = path[0][idx[1] :], path[1][idx[1] :]

            print(cal_line_dense(*p1))
            print(cal_line_dense(*p2))
            print(cal_line_dense(*p3))
            print(idx, path[0].shape)
            print("====")


def cal_line_dense(lx, ly):

    ls = 0
    x0, y0 = 0, 0
    for i, (x, y) in enumerate(zip(lx, ly)):
        if i == 0:
            x0, y0 = x, y
            continue
        ds = np.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
        ls += ds
        x0, y0 = x, y

    return lx.shape[0] / ls, lx.shape[0], ls


def t_ref_path():
    task4 = ("junction", None, "EE6", "ES8")
    path = ReferencePath(task4)

    path.solve_dense()

    path.plot_path2()

    # path = ReferencePath('junction', from_edge='EE6', to_edge='ES8')
    # path.plot_path2()


def t_future_n_data():
    path = ReferencePath("straight")
    plt.axis("equal")
    current_i = 600
    plt.plot(path.path[0], path.path[1])
    future_data_list = path.future_n_data(current_i, 5)
    plt.plot(path.indexs2points(current_i)[0], path.indexs2points(current_i)[1], "go")
    for point in future_data_list:
        plt.plot(point[0], point[1], "r*")
    plt.show()


def t_tracking_error_vector():
    path = ReferencePath("straight")
    xs = np.array([1.875, 1.875, -10, -20], np.float32)
    ys = np.array([-20, 0, -10, -1], np.float32)
    phis = np.array([90, 135, 135, 180], np.float32)
    vs = np.array([10, 12, 10, 10], np.float32)

    tracking_error_vector = path.tracking_error_vector(xs, ys, phis, vs, 10)
    # print(tracking_error_vector)


def t_model():
    pass
    # from crossing import CrossroadEnd2endMixPiFix
    # env = CrossroadEnd2endMixPiFix('left', 0)
    # model = EnvironmentModel('left', 0)
    # obs_list = []
    # obs = env.reset()
    # done = 0
    # # while not done:
    # for i in range(10):
    #     obs_list.append(obs)
    #     action = np.array([0, -1], dtype=np.float32)
    #     obs, reward, done, info = env.step(action)
    #     env.render()
    # obses = np.stack(obs_list, 0)
    # model.reset(obses, 'left')
    # # print(obses.shape)
    # for rollout_step in range(100):
    #     actions = torch.Tensor([[0.5, 0]]).repeat(len(obses), 1)
    #     obses, rewards, punish1, punish2, _, _ = model.rollout_out(actions)
    #     # print(rewards.numpy()[0], punish1.numpy()[0])
    #     model.render()


def t_tf_function():
    class Test2:
        def __init__(self):
            self.c = 2

        def step1(self, a):
            print("trace")
            self.c = a

        def step2(self):
            return self.c

    test2 = Test2()

    # @tf.function#(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
    def f(a):
        test2.step1(a)
        return test2.step2()

    print(f(2), type(test2.c))
    print(f(2), test2.c)

    print(f(torch.Tensor([2])), type(test2.c))
    print(f(torch.Tensor([3])), test2.c)

    # print(f(2), test2.c)
    # print(f(3), test2.c)
    # print(f(2), test2.c)
    # print(f())
    # print(f())
    #
    # test2.c.assign_add(12)
    # print(test2.c)
    # print(f())

    # b= test2.create_test1(1)
    # print(test2.b,b, test2.b.a)
    # b=test2.create_test1(2)
    # print(test2.b,b,test2.b.a)
    # b=test2.create_test1(1)
    # print(test2.b,b,test2.b.a)
    # test2.create_test1(1)
    # test2.pc()
    # test2.create_test1(1)
    # test2.pc()


def t_tffunc(inttt):
    print(22)
    if inttt == "1":
        a = 2
    elif inttt == "2":
        a = 233
    else:
        a = 22
    return a


def t_ref():
    import numpy as np
    import matplotlib.pyplot as plt

    # ref = ReferencePath('left')
    # path1, path2, path3 = ref.path_list
    # path1, path2, path3 = [ite[1200:-1200] for ite in path1],\
    #                       [ite[1200:-1200] for ite in path2], \
    #                       [ite[1200:-1200] for ite in path3]
    # x1, y1, phi1 = path1
    # x2, y2, phi2 = path2
    # x3, y3, phi3 = path3
    # p1, p2, p3 = np.arctan2(y1-(-CROSSROAD_SIZE/2), x1 - (-CROSSROAD_SIZE/2)), \
    #              np.arctan2(y2 - (-CROSSROAD_SIZE / 2), x2 - (-CROSSROAD_SIZE / 2)), \
    #              np.arctan2(y3 - (-CROSSROAD_SIZE / 2), x3 - (-CROSSROAD_SIZE / 2))
    # d1, d2, d3 = np.sqrt(np.square(x1-(-CROSSROAD_SIZE/2))+np.square(y1-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x2-(-CROSSROAD_SIZE/2))+np.square(y2-(-CROSSROAD_SIZE/2))),\
    #              np.sqrt(np.square(x3-(-CROSSROAD_SIZE/2))+np.square(y3-(-CROSSROAD_SIZE/2)))
    #
    # plt.plot(p1, d1, 'r')
    # plt.plot(p2, d2, 'g')
    # plt.plot(p3, d3, 'b')
    # z1 = np.polyfit(p1, d1, 3, rcond=None, full=False, w=None, cov=False)
    # p1_fit = np.poly1d(z1)
    # plt.plot(p1, p1_fit(p1), 'r*')
    #
    # z2 = np.polyfit(p2, d2, 3, rcond=None, full=False, w=None, cov=False)
    # p2_fit = np.poly1d(z2)
    # plt.plot(p2, p2_fit(p2), 'g*')
    #
    # z3 = np.polyfit(p3, d3, 3, rcond=None, full=False, w=None, cov=False)
    # p3_fit = np.poly1d(z3)
    # plt.plot(p3, p3_fit(p3), 'b*')

    ref = ReferencePath("straight")
    path1, path2, path3 = ref.path_list
    path1, path2, path3 = (
        [ite[1200:-1200] for ite in path1],
        [ite[1200:-1200] for ite in path2],
        [ite[1200:-1200] for ite in path3],
    )
    x1, y1, phi1 = path1
    x2, y2, phi2 = path2
    x3, y3, phi3 = path3

    plt.plot(y1, x1, "r")
    plt.plot(y2, x2, "g")
    plt.plot(y3, x3, "b")
    z1 = np.polyfit(y1, x1, 3, rcond=None, full=False, w=None, cov=False)
    print(type(list(z1)))
    p1_fit = np.poly1d(z1)
    print(z1, p1_fit)
    plt.plot(y1, p1_fit(y1), "r*")
    plt.show()


def t_get_lane_ID():
    from_edge = "ES9"
    route_list = task2staticpath[from_edge]
    ref_route = route_list[0]
    e, _ = ref_route[-2]
    print(e)


def t_get_shape():
    in_lane_ID = "ES8_1"
    in_edge_ID = lane2edge_re(in_lane_ID)
    in_lane_shape = map_info[in_edge_ID][in_lane_ID]["shape"]
    print(in_lane_shape)


if __name__ == "__main__":
    t_ref_path()
