#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/6/10
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: preprocessor.py
# =====================================

from math import pi
import numpy as np
import torch
from environment.env_horizon.dynamics_and_models import dist_ego2path, deal_with_phi_diff


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.torch_mean = torch.zeros(shape, dtype=torch.float32).clone().detach()
        self.torch_var = torch.ones(shape, dtype=torch.float32).clone().detach()

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )
        self.torch_mean = torch.tensor(self.mean, dtype=torch.float32).clone().detach()
        self.torch_var = torch.tensor(self.var, dtype=torch.float32).clone().detach()

    def set_params(self, mean, var, count):
        self.mean = mean
        self.var = var
        self.count = count
        self.torch_mean = torch.tensor(self.mean, dtype=torch.float32).clone().detach()
        self.torch_var = torch.tensor(self.var, dtype=torch.float32).clone().detach()

    def get_params(self):
        return self.mean, self.var, self.count


class Preprocessor(object):
    def __init__(
        self,
        ob_shape,
        obs_ptype="normalize",
        rew_ptype="normalize",
        rew_scale=None,
        rew_shift=None,
        args=None,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        **kwargs,
    ):
        self.obs_ptype = obs_ptype
        self.ob_rms = RunningMeanStd(shape=ob_shape) if self.obs_ptype == "normalize" else None
        self.rew_ptype = rew_ptype
        self.ret_rms = RunningMeanStd(shape=()) if self.rew_ptype == "normalize" else None
        self.obs_scale = None
        self.rew_scale = rew_scale if self.rew_ptype == "scale" else None
        self.rew_shift = rew_shift if self.rew_ptype == "scale" else None

        self.clipob = clipob
        self.cliprew = cliprew

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_agent = None
        self.args = args
        self.obs_ego_scale = torch.FloatTensor(np.array(self.args.obs_ego_scale +
                                                        self.args.tracking_error_scale +
                                                        self.args.future_data_scale * self.args.env_kwargs_num_future_data))
        self.obs_other_scale = torch.FloatTensor(np.array(self.args.obs_other_scale))
        self.exp_v = args.env_kwargs_expected_v
        if "num_agent" in kwargs.keys():
            self.ret = np.zeros(kwargs["num_agent"])
            self.num_agent = kwargs["num_agent"]
        else:
            self.ret = 0

    def process_rew(self, rew, done):
        if self.rew_ptype == "normalize":
            if self.num_agent is not None:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(self.ret)
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = np.where(done == 1, np.zeros(self.ret), self.ret)
            else:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(np.array([self.ret]))
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = 0 if done else self.ret
            return rew
        elif self.rew_ptype == "scale":
            return (rew + self.rew_shift) * self.rew_scale  # 0.1 * rew
        else:
            return rew

    def numpy_process_obs_PI(self, obses_ego, obses_bike, obses_person, obses_veh):  # 相对坐标转化，该坐标下自车位置为(0, 0, 0)

        obses_ego = torch.as_tensor(obses_ego[np.newaxis, :], dtype=torch.float32)
        obses_bike = torch.as_tensor(obses_bike, dtype=torch.float32)
        obses_person = torch.as_tensor(obses_person, dtype=torch.float32)
        obses_veh = torch.as_tensor(obses_veh, dtype=torch.float32)

        (
            processed_obses_ego_reduced,
            processed_obses_bike,
            processed_obses_person,
            processed_obses_veh,
        ) = self.torch_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
        processed_obses_ego_reduced = processed_obses_ego_reduced.squeeze()

        return processed_obses_ego_reduced, processed_obses_bike, processed_obses_person, processed_obses_veh

    def torch_process_obses_PI(self, obses_ego, obses_bike, obses_person, obses_veh):
        # print(obses_ego.shape, obses_bike.shape, obses_person.shape, obses_veh.shape)
        # print(obses_ego)
        # print(obses_veh)
        # print('==========================================')
        if self.obs_ptype == "scale":
            obs_ego_clone = obses_ego.clone()

            if self.args.joe == "junction":

                """
                相对自车做旋转
                """
                x = obs_ego_clone[:, 3]
                y = obs_ego_clone[:, 4]
                phi = obs_ego_clone[:, 5]
                pad = torch.zeros_like(x)

                ego_num = x.shape[0]  # batch size
                bike_num = obses_bike.shape[0] // ego_num  # 除法取商
                person_num = obses_person.shape[0] // ego_num
                veh_num = obses_veh.shape[0] // ego_num
                base_shift = torch.stack([x, y, pad, phi, pad], dim=1)
                ego_shift = torch.stack([pad, pad, pad, x, y, phi, pad, pad, pad], dim=1)

                bike_shift = torch.repeat_interleave(
                    base_shift, torch.ones(ego_num, dtype=torch.long) * bike_num, dim=0
                )
                person_shift = torch.repeat_interleave(
                    base_shift, torch.ones(ego_num, dtype=torch.long) * person_num, dim=0
                )
                veh_shift = torch.repeat_interleave(base_shift, torch.ones(ego_num, dtype=torch.long) * veh_num, dim=0)

                processed_obses_ego = self.obs_ego_scale * (obs_ego_clone - ego_shift)

                dif_ego2bike = obses_bike - bike_shift
                dif_x_ego2bike = dif_ego2bike[:, 0]
                dif_y_ego2bike = dif_ego2bike[:, 1]
                phi_ego4bike = torch.deg2rad(bike_shift[:, 3])  # to rad
                # torch.deg2rad()
                rel_x_along_path_ego2bike = torch.mul(dif_x_ego2bike, torch.cos(phi_ego4bike)) + torch.mul(
                    dif_y_ego2bike, torch.sin(phi_ego4bike)
                )
                rel_y_along_path_ego2bike = -torch.mul(dif_x_ego2bike, torch.sin(phi_ego4bike)) + torch.mul(
                    dif_y_ego2bike, torch.cos(phi_ego4bike)
                )
                dif_ego2bike[:, 0] = rel_x_along_path_ego2bike
                dif_ego2bike[:, 1] = rel_y_along_path_ego2bike
                processed_obses_bike = self.obs_other_scale * dif_ego2bike

                dif_ego2person = obses_person - person_shift
                dif_x_ego2person = dif_ego2person[:, 0]
                dif_y_ego2person = dif_ego2person[:, 1]
                phi_ego4person = torch.deg2rad(person_shift[:, 3])

                rel_x_along_path_ego2person = torch.mul(dif_x_ego2person, torch.cos(phi_ego4person)) + torch.mul(
                    dif_y_ego2person, torch.sin(phi_ego4person)
                )
                rel_y_along_path_ego2person = -torch.mul(dif_x_ego2person, torch.sin(phi_ego4person)) + torch.mul(
                    dif_y_ego2person, torch.cos(phi_ego4person)
                )
                dif_ego2person[:, 0] = rel_x_along_path_ego2person
                dif_ego2person[:, 1] = rel_y_along_path_ego2person
                processed_obses_person = self.obs_other_scale * dif_ego2person

                dif_ego2veh = obses_veh - veh_shift
                dif_x_ego2veh = dif_ego2veh[:, 0]
                dif_y_ego2veh = dif_ego2veh[:, 1]
                phi_ego4veh = torch.deg2rad(veh_shift[:, 3])

                rel_x_along_path_ego2veh = torch.mul(dif_x_ego2veh, torch.cos(phi_ego4veh)) + torch.mul(
                    dif_y_ego2veh, torch.sin(phi_ego4veh)
                )
                rel_y_along_path_ego2veh = -torch.mul(dif_x_ego2veh, torch.sin(phi_ego4veh)) + torch.mul(
                    dif_y_ego2veh, torch.cos(phi_ego4veh)
                )
                dif_ego2veh[:, 0] = rel_x_along_path_ego2veh
                dif_ego2veh[:, 1] = rel_y_along_path_ego2veh
                processed_obses_veh = self.obs_other_scale * dif_ego2veh

                processed_obses_ego_reduced = processed_obses_ego[:, [0, 1, 2, 6, 7, 8]]  # TODO hard coding
                return processed_obses_ego_reduced, processed_obses_bike, processed_obses_person, processed_obses_veh
            elif self.args.joe == "edge":  # decompose position along the path & relative speed

                v_x = obs_ego_clone[:, 0]
                x = obs_ego_clone[:, 3]
                y = obs_ego_clone[:, 4]
                phi = obs_ego_clone[:, 5]
                closest_x_path = obs_ego_clone[:, 6]
                closest_y_path = obs_ego_clone[:, 7]
                phi_path = obs_ego_clone[:, 8]  # [-180, 180]
                pad = torch.zeros_like(x)

                ego_num = x.shape[0]  # batch size
                bike_num = obses_bike.shape[0] // ego_num  # 除法取商
                person_num = obses_person.shape[0] // ego_num
                veh_num = obses_veh.shape[0] // ego_num

                base_shift = torch.stack([x, y, v_x, phi_path, pad], dim=1)
                bike_shift = torch.repeat_interleave(base_shift,
                                                     torch.ones(ego_num, dtype=torch.long) * bike_num, dim=0)
                person_shift = torch.repeat_interleave(base_shift,
                                                       torch.ones(ego_num, dtype=torch.long) * person_num, dim=0)
                veh_shift = torch.repeat_interleave(base_shift,
                                                    torch.ones(ego_num, dtype=torch.long) * veh_num, dim=0)

                obs_ego_clone[:, 6] = dist_ego2path(x, y, closest_x_path, closest_y_path, phi_path)
                obs_ego_clone[:, 7] = deal_with_phi_diff(phi - phi_path)
                obs_ego_clone[:, 8] = v_x - self.exp_v
                processed_obses_ego = self.obs_ego_scale * obs_ego_clone  # [vx, vy, r, dist_ego2p, phi_ego2p, v_ego2p]

                # process bike
                dif_ego2bike = obses_bike - bike_shift
                dif_x_ego2bike = dif_ego2bike[:, 0]
                dif_y_ego2bike = dif_ego2bike[:, 1]
                phi_path4bike = bike_shift[:, 3]

                rel_x_along_path_ego2bike = \
                    torch.mul(dif_x_ego2bike, torch.cos(phi_path4bike * pi / 180.0)) \
                    + torch.mul(dif_y_ego2bike, torch.sin(phi_path4bike * pi / 180.0))
                rel_y_along_path_ego2bike = \
                    - torch.mul(dif_x_ego2bike, torch.sin(phi_path4bike * pi / 180.0)) \
                    + torch.mul(dif_y_ego2bike, torch.cos(phi_path4bike * pi / 180.0))
                dif_ego2bike[:, 0] = rel_x_along_path_ego2bike
                dif_ego2bike[:, 1] = rel_y_along_path_ego2bike
                processed_obses_bike = self.obs_other_scale * dif_ego2bike

                # process person
                dif_ego2person = obses_person - person_shift
                dif_x_ego2person = dif_ego2person[:, 0]
                dif_y_ego2person = dif_ego2person[:, 1]
                phi_path4person = person_shift[:, 3]

                rel_x_along_path_ego2person = \
                    torch.mul(dif_x_ego2person, torch.cos(phi_path4person * pi / 180.0)) \
                    + torch.mul(dif_y_ego2person, torch.sin(phi_path4person * pi / 180.0))
                rel_y_along_path_ego2person = \
                    - torch.mul(dif_x_ego2person, torch.sin(phi_path4person * pi / 180.0)) \
                    + torch.mul(dif_y_ego2person, torch.cos(phi_path4person * pi / 180.0))
                dif_ego2person[:, 0] = rel_x_along_path_ego2person
                dif_ego2person[:, 1] = rel_y_along_path_ego2person
                processed_obses_person = self.obs_other_scale * dif_ego2person

                # process vehicle
                dif_ego2veh = obses_veh - veh_shift
                dif_x_ego2veh = dif_ego2veh[:, 0]
                dif_y_ego2veh = dif_ego2veh[:, 1]
                phi_path4veh = veh_shift[:, 3]

                rel_x_along_path_ego2veh = \
                    torch.mul(dif_x_ego2veh, torch.cos(phi_path4veh * pi / 180.0)) \
                    + torch.mul(dif_y_ego2veh, torch.sin(phi_path4veh * pi / 180.0))
                rel_y_along_path_ego2veh = \
                    - torch.mul(dif_x_ego2veh, torch.sin(phi_path4veh * pi / 180.0)) \
                    + torch.mul(dif_y_ego2veh, torch.cos(phi_path4veh * pi / 180.0))
                dif_ego2veh[:, 0] = rel_x_along_path_ego2veh
                dif_ego2veh[:, 1] = rel_y_along_path_ego2veh
                processed_obses_veh = self.obs_other_scale * dif_ego2veh

                processed_obses_ego_reduced = processed_obses_ego[:, [0, 1, 2, 6, 7, 8]]  # TODO hard coding
                return processed_obses_ego_reduced, processed_obses_bike, processed_obses_person, processed_obses_veh
            else:
                raise ValueError("Have not set joe properly")

        else:
            print("no scale")
            raise ValueError

    def np_process_rewards(self, rewards):
        if self.rew_ptype == "normalize":
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == "scale":
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def torch_process_rewards(self, rewards):
        if self.rew_ptype == "normalize":
            rewards = torch.clamp(
                rewards / torch.sqrt(self.ret_rms.torch_var + torch.tensor(self.epsilon)),
                min=-self.cliprew,
                max=self.cliprew,
            )
            return rewards
        elif self.rew_ptype == "scale":
            return (rewards + torch.tensor(self.rew_shift, dtype=torch.float32)) * torch.tensor(
                self.rew_scale, dtype=torch.float32
            )
        else:
            return torch.tensor(rewards, dtype=torch.float32)

    def set_params(self, params):
        if self.ob_rms:
            self.ob_rms.set_params(*params["ob_rms"])
        if self.ret_rms:
            self.ret_rms.set_params(*params["ret_rms"])

    def get_params(self):
        tmp = {}
        if self.ob_rms:
            tmp.update({"ob_rms": self.ob_rms.get_params()})
        if self.ret_rms:
            tmp.update({"ret_rms": self.ret_rms.get_params()})

        return tmp

    def save_params(self, save_dir):
        np.save(save_dir + "/ppc_params.npy", self.get_params())

    def load_params(self, load_dir):
        params = np.load(load_dir + "/ppc_params.npy", allow_pickle=True)
        params = params.item()
        self.set_params(params)


class PreprocessorARCHIVE(object):
    def __init__(
        self,
        ob_shape,
        obs_ptype="normalize",
        rew_ptype="normalize",
        rew_scale=None,
        rew_shift=None,
        args=None,
        clipob=10.0,
        cliprew=10.0,
        gamma=0.99,
        epsilon=1e-8,
        **kwargs,
    ):
        self.obs_ptype = obs_ptype
        self.ob_rms = RunningMeanStd(shape=ob_shape) if self.obs_ptype == "normalize" else None
        self.rew_ptype = rew_ptype
        self.ret_rms = RunningMeanStd(shape=()) if self.rew_ptype == "normalize" else None
        self.obs_scale = None
        self.rew_scale = rew_scale if self.rew_ptype == "scale" else None
        self.rew_shift = rew_shift if self.rew_ptype == "scale" else None

        self.clipob = clipob
        self.cliprew = cliprew

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_agent = None
        self.args = args
        self.obs_ego_scale = torch.FloatTensor(
            np.array(
                [0.2, 1.0, 2.0, 1 / 30.0, 1 / 30, 1 / 180.0]
                + [1.0, 1 / 15.0, 0.2]
                + [1.0, 1.0, 1 / 15.0] * self.args.env_kwargs_num_future_data
            )
        )
        self.obs_other_scale = torch.FloatTensor(np.array([1 / 30.0, 1 / 30.0, 0.2, 1 / 180.0, 1.0]))
        if "num_agent" in kwargs.keys():
            self.ret = np.zeros(kwargs["num_agent"])
            self.num_agent = kwargs["num_agent"]
        else:
            self.ret = 0

    def process_rew(self, rew, done):
        if self.rew_ptype == "normalize":
            if self.num_agent is not None:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(self.ret)
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = np.where(done == 1, np.zeros(self.ret), self.ret)
            else:
                self.ret = self.ret * self.gamma + rew
                self.ret_rms.update(np.array([self.ret]))
                rew = np.clip(rew / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
                self.ret = 0 if done else self.ret
            return rew
        elif self.rew_ptype == "scale":
            return (rew + self.rew_shift) * self.rew_scale
        else:
            return rew

    def process_obs(self, obs):
        if self.obs_ptype == "normalize":

            if self.num_agent is not None:
                self.ob_rms.update(obs)
                obs = np.clip(
                    (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob
                )
                return obs
            else:
                self.ob_rms.update(np.array([obs]))
                obs = np.clip(
                    (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob
                )
                return obs

        elif self.obs_ptype == "scale":
            return obs * self.obs_scale
        else:
            return obs

    def numpy_process_obs_PI(self, obses_ego, obses_bike, obses_person, obses_veh):
        if self.obs_ptype == "scale":
            processed_obses_ego = self.obs_ego_scale * torch.tensor(obses_ego, dtype=torch.float32)
            processed_obses_bike = self.obs_other_scale * torch.tensor(obses_bike, dtype=torch.float32)
            processed_obses_person = self.obs_other_scale * torch.tensor(obses_person, dtype=torch.float32)
            processed_obses_veh = self.obs_other_scale * torch.tensor(obses_veh, dtype=torch.float32)
            return processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh

    def torch_process_obses_PI(self, obses_ego, obses_bike, obses_person, obses_veh):
        if self.obs_ptype == "scale":
            processed_obses_ego = self.obs_ego_scale * obses_ego
            processed_obses_bike = self.obs_other_scale * obses_bike
            processed_obses_person = self.obs_other_scale * obses_person
            processed_obses_veh = self.obs_other_scale * obses_veh
            return processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh
        else:
            print("no scale")
            raise ValueError

    def np_process_obses(self, obses):
        if self.obs_ptype == "normalize":
            obses = np.clip(
                (obses - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob
            )
            return obses
        elif self.obs_ptype == "scale":
            return obses * self.obs_scale
        else:
            return obses

    def np_process_rewards(self, rewards):
        if self.rew_ptype == "normalize":
            rewards = np.clip(rewards / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return rewards
        elif self.rew_ptype == "scale":
            return (rewards + self.rew_shift) * self.rew_scale
        else:
            return rewards

    def torch_process_obses(self, obses):
        if self.obs_ptype == "normalize":
            obses = torch.clamp(
                (obses - self.ob_rms.torch_mean) / torch.sqrt(self.ob_rms.torch_var + torch.tensor(self.epsilon)),
                min=-self.clipob,
                max=self.clipob,
            )
            return obses
        elif self.obs_ptype == "scale":
            return obses * torch.tensor(self.obs_scale, dtype=torch.float32)

        else:
            return torch.tensor(obses, dtype=torch.float32)

    def torch_process_rewards(self, rewards):
        if self.rew_ptype == "normalize":
            rewards = torch.clamp(
                rewards / torch.sqrt(self.ret_rms.torch_var + torch.tensor(self.epsilon)),
                min=-self.cliprew,
                max=self.cliprew,
            )
            return rewards
        elif self.rew_ptype == "scale":
            return (rewards + torch.tensor(self.rew_shift, dtype=torch.float32)) * torch.tensor(
                self.rew_scale, dtype=torch.float32
            )
        else:
            return torch.tensor(rewards, dtype=torch.float32)

    def set_params(self, params):
        if self.ob_rms:
            self.ob_rms.set_params(*params["ob_rms"])
        if self.ret_rms:
            self.ret_rms.set_params(*params["ret_rms"])

    def get_params(self):
        tmp = {}
        if self.ob_rms:
            tmp.update({"ob_rms": self.ob_rms.get_params()})
        if self.ret_rms:
            tmp.update({"ret_rms": self.ret_rms.get_params()})

        return tmp

    def save_params(self, save_dir):
        np.save(save_dir + "/ppc_params.npy", self.get_params())

    def load_params(self, load_dir):
        params = np.load(load_dir + "/ppc_params.npy", allow_pickle=True)
        params = params.item()
        self.set_params(params)


def test_convert():
    a = np.array([1, 3, 2], dtype=np.float32)
    b = torch.tensor(a, dtype=torch.float32)
    print(b.type())


def test_torch_assign():
    shape = (3,)
    a = torch.zeros(shape, dtype=torch.float32)
    print(f"a = {a}, type = {a.type()}")
    b = torch.tensor(np.array([1, 2, 3]), dtype=torch.float32)
    a = b.clone().detach()
    print(f"a = {a}, type = {a.type()}")


def run_repeat():
    base_shift = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
    # print(torch.ones(2, dtype=torch.long) * 3)
    bike = torch.repeat_interleave(base_shift, torch.ones(2, dtype=torch.long) * 3, dim=0)
    # print(bike)

    xs = torch.tensor([1, 2, 3, 4, 5])
    ys = torch.tensor([0, 0, 0, 0, 0])
    x_path = torch.tensor([1, 2, 3, 4, 5])
    y_path = torch.tensor([1, 2, 3, 4, 5])
    phi_path = torch.tensor([45, 45, 45, 45, 45])
    print(dist_ego2path(xs, ys, x_path, y_path, phi_path))


if __name__ == "__main__":
    run_repeat()
