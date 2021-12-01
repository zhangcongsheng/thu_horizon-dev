import datetime
import json
import argparse
import os
from math import cos, pi, sin
import shutil

import PIL
import gym
import pprint
import matplotlib.pyplot as plt
import numpy as np
import torch

import environment  # noqa: F401
from environment.env_horizon.dynamics_and_models import environment_model, ReferencePath
from environment.env_horizon.endtoend_env_utils import rotate_coordination
from environment.env_horizon.misc_ic import MODE2TASK
from hierarchical_decision.multi_path_generator import MultiPathGenerator
from utils.load_policy import LoadPolicy
from utils.misc import AttrDict, TimerStat, args2envkwargs
from preprocessor import Preprocessor

RENDER_MODE = "save"

class _HierarchicalDecision(object):
    def __init__(self, args, train_exp_dir, ite, logdir=None):

        self.args = args

        self.env_id = self.args.env_id
        self.joe = self.args.joe

        if self.joe == "junction":
            self.policy = LoadPolicy(self.args, train_exp_dir, ite)
            self.env = gym.make(self.env_id, **args2envkwargs(self.args, 0))
            self.model = environment_model(mode="selecting", **args2envkwargs(self.args, 0))
        elif self.joe == "edge":
            pass
            # self.policy = LoadPolicy("../utils/models/{}/{}".format(task, train_exp_dir), ite)
            # self.env = gym.make(env_id, **args2envkwargs(args, evaluator_id))
            # self.model = EnvironmentModel(self.task, mode="selecting")
        else:
            raise ValueError("joe should be junction or edge")

        # self.recorder = Recorder()  # TODO  to change
        self.episode_counter = -1
        self.step_counter = -1
        self.obs = None

        self.task4 = args2envkwargs(self.args, 0)["training_task4"]
        path_num = len(ReferencePath(self.task4, expected_v=args.env_kwargs_expected_v).path_list)
        self.stg = MultiPathGenerator(path_num, expected_v=args.env_kwargs_expected_v)
        self.step_timer = TimerStat()
        self.ss_timer = TimerStat()
        self.logdir = logdir
        if self.logdir is not None:
            config = dict(env_id=self.env_id, train_exp_dir=train_exp_dir, ite=ite)
            with open(self.logdir + "/config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
        # self.fig = plt.figure(figsize=(8, 8))
        plt.ion()
        self.hist_posi = []
        self.old_index = 0
        self.path_list = self.stg.generate_path(self.task4)

        # # ------------------build graph for tf.function in advance-----------------------
        # for i in range(3):
        #     obs = self.env.reset()
        #     obs = torch.as_tensor(obs[np.newaxis, :], dtype=torch.float32)
        #     self.is_safe(obs, i)
        # obs = self.env.reset()
        # obs_with_specific_shape = np.tile(obs, (3, 1))
        # self.policy.run_batch(obs_with_specific_shape)
        # self.policy.obj_value_batch(obs_with_specific_shape)
        # # ------------------build graph for tf.function in advance-----------------------
        self.reset()

    def _load_json(self, exp_dir):
        pass

    def reset(self,):
        self.obs = self.env.reset()
        # self.recorder.reset() # TODO need to be changed
        self.old_index = 0
        self.hist_posi = []
        if self.logdir is not None:
            self.episode_counter += 1
            os.makedirs(self.logdir + '/episode{}/figs'.format(self.episode_counter))

            self.episodes_path = os.path.join(
                self.logdir,
                "episode{}".format(self.episode_counter),
                "render"
            )
            os.makedirs(self.episodes_path, exist_ok=True)
            self.step_counter = -1
            # self.recorder.save(self.logdir) # TODO need to be changed
            if self.episode_counter >= 1:
                pass
                # select_and_rename_snapshots_of_an_episode(self.logdir, self.episode_counter-1, 12)
                # self.recorder.plot_and_save_ith_episode_curves(self.episode_counter-1,
                #                                                self.logdir + '/episode{}/figs'.format(self.episode_counter-1),
                #                                                isshow=False) # TODO need to be changed
        return self.obs

    def is_safe(self, obs_ego, obs_bike, obs_person, obs_veh, path_index):
        self.model.add_traj(obs_ego, obs_bike, obs_person, obs_veh, path_index)
        punish = 0.0
        for step in range(5):
            action = self.policy.run_batch(obs_ego, obs_bike, obs_person, obs_veh)
            obs_ego, obs_bike , obs_person, obs_veh, _, _, _,  veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, _ =\
            self.model.rollout_out(action)
            punish += veh2veh4real[0] + veh2road4real[0] + veh2bike4real[0] + veh2person4real[0]
        return False if punish > 0 else True

    def safe_shield(self, real_obs, path_index):
        action_safe_set = [[[0.0, -1.0]]]
        real_obs = torch.as_tensor(real_obs[np.newaxis, :], dtype=torch.float32)
        obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(real_obs)
        if not self.is_safe(obs_ego, obs_bike, obs_person, obs_veh, path_index):
            print("SAFETY SHIELD STARTED!")
            return np.array(action_safe_set[0], dtype=np.float32).squeeze(0), True
        else:
            return self.policy.run_batch(obs_ego, obs_bike, obs_person, obs_veh).numpy()[0], False

    def step(self):
        self.step_counter += 1
        with self.step_timer:
            obs_list = []
            # select optimal path
            for path in self.path_list:
                self.env.set_traj(path)
                obs_list.append(self.env._get_obs())
            all_obs = torch.as_tensor(np.stack(obs_list, axis=0), dtype=torch.float32)
            obs_ego, obs_bike, obs_person, obs_veh = self.split_obses(all_obs)

            path_values = self.policy.obj_value_batch(obs_ego, obs_bike, obs_person, obs_veh).numpy()
            old_value = path_values[self.old_index]
            new_index, new_value = int(np.argmin(path_values)), min(
                path_values
            )  # value is to approximate (- sum of reward)
            path_index = self.old_index if old_value - new_value < 0.1 else new_index  # TODO: maybe need to be tuned
            self.old_index = path_index

            self.env.set_traj(self.path_list[path_index])
            self.obs_real = obs_list[path_index]

            # obtain safe action
            with self.ss_timer:
                safe_action, is_ss = self.safe_shield(self.obs_real, path_index)
            print("ALL TIME:", self.step_timer.mean, "ss", self.ss_timer.mean)

        save_dir = os.path.join(
            self.episodes_path,
            "step{}.png".format(self.step_counter)
        )

        # self.recorder.record(self.obs_real, safe_action, self.step_timer.mean,
        #                      path_index, path_values, self.ss_timer.mean, is_ss) # TODO need to be changed
        self.obs, r, done, info = self.env.step(safe_action)
        self.render(self.path_list, path_values, path_index, RENDER_MODE, save_dir)

        return done

    def split_obses(self, all_obs):
        start = 0
        end = self.args.state_ego_dim + self.args.state_track_dim
        obs_ego = all_obs[:, start:end]
        start = end
        end = start + self.args.state_bike_dim
        obs_bike = all_obs[:, start:end]
        start = end
        end = start + self.args.state_person_dim
        obs_person = all_obs[:, start:end]
        start = end
        end = start + self.args.state_veh_dim
        obs_veh = all_obs[:, start:end]
        obs_bike = obs_bike.reshape(-1, self.args.per_bike_dim)
        obs_person = obs_person.reshape(-1, self.args.per_person_dim)
        obs_veh = obs_veh.reshape(-1, self.args.per_veh_dim)
        return obs_ego.detach(), obs_bike.detach(), obs_person.detach(), obs_veh.detach()

    def render(self, traj_list, path_values, path_index, mode="save", save_dir=None):

        # need to be add to render because
        """
        try:
            color = ['blue', 'coral', 'darkcyan']
            for i, item in enumerate(traj_list):
                if i == path_index:
                    plt.plot(item.path[0], item.path[1], color=color[i])
                else:
                    plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                indexs, points = item.find_closest_point(np.array([ego_x], np.float32), np.array([ego_y], np.float32))
                path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                plt.plot(path_x, path_y, color=color[i])
        except Exception:
            pass
        """
        raise NotImplementedError

def select_and_rename_snapshots_of_an_episode(logdir, epinum, num):
    file_list = os.listdir(logdir + '/episode{}'.format(epinum))
    file_num = len(file_list) - 1
    intervavl = file_num // (num-1)
    start = file_num % (num-1)
    print(start, file_num, intervavl)
    selected = [start//2] + [start//2+intervavl*i-1 for i in range(1, num)]
    print(selected)
    if file_num > 0:
        for i, j in enumerate(selected):
            shutil.copyfile(logdir + '/episode{}/step{:03d}.pdf'.format(epinum, j),
                            logdir + '/episode{}/figs/{:03d}.pdf'.format(epinum, i))

class CrossingIDC(_HierarchicalDecision):
    def render(self, traj_list, path_values, path_index, mode="save", save_dir=None):
        light_line_width = 3
        dotted_line_style = "--"
        solid_line_style = "-"
        extension = 40
        DETECT_RANGE = 40
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

            for rl in self.env.edge_lines:
                plt.plot(rl[0], rl[1], "k")

            for ra in self.env.edge_areas:
                plt.fill(*ra, "lemonchiffon")

            for ja in self.env.junction_areas:
                plt.fill(*ja, "lightcyan")

            # plot vehicles
            ego_x = self.env.ego_dynamics["x"]
            ego_y = self.env.ego_dynamics["y"]

            for veh in self.env.all_vehicles:
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
            for mode, num in self.env.veh_mode_dict.items():
                for i in range(num):
                    veh = self.env.interested_vehs[mode][i]
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
            for mode, num in self.env.bicycle_mode_dict.items():
                for i in range(num):
                    veh = self.env.interested_vehs[mode][i]
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
            for mode, num in self.env.person_mode_dict.items():
                for i in range(num):
                    veh = self.env.interested_vehs[mode][i]
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
            ego_v_x = self.env.ego_dynamics["v_x"]
            ego_v_y = self.env.ego_dynamics["v_y"]
            ego_r = self.env.ego_dynamics["r"]
            ego_x = self.env.ego_dynamics["x"]
            ego_y = self.env.ego_dynamics["y"]
            ego_phi = self.env.ego_dynamics["phi"]
            ego_l = self.env.ego_dynamics["l"]
            ego_w = self.env.ego_dynamics["w"]
            ego_alpha_f = self.env.ego_dynamics["alpha_f"]
            ego_alpha_r = self.env.ego_dynamics["alpha_r"]
            alpha_f_bound = self.env.ego_dynamics["alpha_f_bound"]
            alpha_r_bound = self.env.ego_dynamics["alpha_r_bound"]
            r_bound = self.env.ego_dynamics["r_bound"]

            plot_phi_line("self_car", ego_x, ego_y, ego_phi, "red")
            draw_rotate_rec(ax, ego_x, ego_y, ego_phi, ego_l, ego_w, "red")

            # plot future data (static path)
            tracking_info = self.obs[
                self.env.ego_info_dim : self.env.ego_info_dim
                + self.env.per_tracking_info_dim * (self.env.num_future_data + 1)
            ]
            future_path = tracking_info[self.env.per_tracking_info_dim :]
            for i in range(self.env.num_future_data):
                delta_x, delta_y, delta_phi = future_path[
                    i * self.env.per_tracking_info_dim : (i + 1) * self.env.per_tracking_info_dim
                ]
                path_x, path_y, path_phi = (
                    ego_x + delta_x,
                    ego_y + delta_y,
                    ego_phi - delta_phi,
                )
                plt.plot(path_x, path_y, "g.")
                plot_phi_line("self_car", path_x, path_y, path_phi, "g")

            delta_, _, _ = tracking_info[:3]
            indexs, points = self.env.ref_path.find_closest_point(
                np.array([ego_x], np.float32), np.array([ego_y], np.float32)
            )
            ax.plot(self.env.ref_path.path[0], self.env.ref_path.path[1], color="g")

            # ax.scatter(self.ref_path.path[0][400], self.ref_path.path[1][400], color = 'k', s=100)
            # ax.scatter(self.ref_path.path[0][1020], self.ref_path.path[1][1020],color = 'k', s=100)
            # print(self.ref_path.path[0][400], self.ref_path.path[1][400])
            # print(self.ref_path.path[0][1020], self.ref_path.path[1][1020])
            path_x_print = points[0][0].numpy()
            path_y_print = points[1][0].numpy()
            path_phi_print = points[2][0].numpy()
            # plt.plot(path_x, path_y, 'g.')
            delta_x = ego_x - path_x_print
            delta_y = ego_y - path_y_print
            delta_phi = ego_phi - path_phi_print

            color = ['blue', 'coral', 'darkcyan']
            try:
                for i, item in enumerate(traj_list):
                    if i == path_index:
                        plt.plot(item.path[0], item.path[1], color=color[i])
                    else:
                        plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                    indexs, points = item.find_closest_point(np.array([ego_x], np.float32),
                                                             np.array([ego_y], np.float32))
                    path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                    plt.plot(path_x, path_y, color=color[i])
            except Exception:
                pass

            if self.env.action is not None:
                steer, a_x = self.env.action[0], self.env.action[1]
                steer_string = r"steer: {:.2f}rad (${:.2f}\degree$)".format(steer, steer * 180 / np.pi)
                a_x_string = "a_x: {:.2f}m/s^2".format(a_x)
            else:
                steer_string = r"steer: [N/A]rad ($[N/A]\degree$)"
                a_x_string = "a_x: [N/A] m/s^2"

            if self.env.reward_info is not None:
                reward_string_list = []
                reward_color_list = []
                for key, val in self.env.reward_info.items():
                    reward_string_list.append("{}: {:.4f}".format(key, val))
                    reward_color_list.append("lavenderblush")
            else:
                reward_string_list = []
                reward_color_list = []

            text_data_list = [
                "ego_x: {:.2f}m".format(ego_x),
                "ego_y: {:.2f}m".format(ego_y),
                "path_x: {:.2f}m".format(path_x_print),
                "path_y: {:.2f}m".format(path_y_print),
                "delta_: {:.2f}m".format(delta_),
                "delta_x: {:.2f}m".format(delta_x),
                "delta_y: {:.2f}m".format(delta_y),
                r"ego_phi: ${:.2f}\degree$".format(ego_phi),
                r"path_phi: ${:.2f}\degree$".format(path_phi_print),
                r"delta_phi: ${:.2f}\degree$".format(delta_phi),
                "v_x: {:.2f}m/s".format(ego_v_x),
                "exp_v: {:.2f}m/s".format(self.env.exp_v),
                "delta_v: {:.2f}m/s".format(ego_v_x - self.env.exp_v),
                "v_y: {:.2f}m/s".format(ego_v_y),
                "yaw_rate: {:.2f}rad/s".format(ego_r),
                "yaw_rate bound: [{:.2f}, {:.2f}]".format(-r_bound, r_bound),
                r"$\alpha_f$: {:.2f} rad".format(ego_alpha_f),
                r"$\alpha_f$ bound: [{:.2f}, {:.2f}] ".format(-alpha_f_bound, alpha_f_bound),
                r"$\alpha_r$: {:.2f} rad".format(ego_alpha_r),
                r"$\alpha_r$ bound: [{:.2f}, {:.2f}] ".format(-alpha_r_bound, alpha_r_bound),
                steer_string,
                a_x_string,
                "done info: {}".format(self.env.done_type),
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
            plt.tight_layout()

            if render_mode == "human":
                plt.xlim(xmin=path_x_print - 50, xmax=path_x_print + 50)
                plt.ylim(ymin=path_y_print - 40, ymax=path_y_print + 40)
                plt.show()
                plt.pause(0.001)
                return None
            elif render_mode == "save":
                """
                plt.show()
                plt.savefig(save_dir, dpi=500)
                plt.pause(0.001)
                """
                plt.xlim(xmin=path_x_print - 50, xmax=path_x_print + 50)
                plt.ylim(ymin=path_y_print - 40, ymax=path_y_print + 40)
                fig.canvas.draw()
                img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                # image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                # image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                # img = PIL.Image.fromarray(image_from_plot, "RGB")
                img.save(save_dir)
                return None
            elif render_mode == "rgb_array":
                plt.xlim(xmin=path_x_print - 50, xmax=path_x_print + 50)
                plt.ylim(ymin=path_y_print - 40, ymax=path_y_print + 40)
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                return image_from_plot

        return ploter(render_mode=mode, save_dir=save_dir)


class MultiLaneIDC(_HierarchicalDecision):
    def render(self, traj_list, path_values, path_index, mode="human", save_dir=None):
        light_line_width = 3
        dotted_line_style = "--"
        solid_line_style = "-"
        extension = 40

        def is_in_plot_area(x, y, ego_x, ego_y, tolerance=50):
            _distance2 = np.sqrt(np.square(x - ego_x) + np.square(y - ego_y))
            if _distance2 <= tolerance:
                return True
            else:
                return False

        def draw_rotate_rec(ax, x, y, a, l, w, color, linestyle="-"):
            RU_x, RU_y, _ = rotate_coordination(l / 2, w / 2, 0, -a)
            RD_x, RD_y, _ = rotate_coordination(l / 2, -w / 2, 0, -a)
            LU_x, LU_y, _ = rotate_coordination(-l / 2, w / 2, 0, -a)
            LD_x, LD_y, _ = rotate_coordination(-l / 2, -w / 2, 0, -a)
            ax.plot([RU_x + x, RD_x + x], [RU_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([RU_x + x, LU_x + x], [RU_y + y, LU_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, RD_x + x], [LD_y + y, RD_y + y], color=color, linestyle=linestyle)
            ax.plot([LD_x + x, LU_x + x], [LD_y + y, LU_y + y], color=color, linestyle=linestyle)

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
            ax = plt.axes([-0.05, -0.05, 1.1, 1.1])
            fig = plt.gcf()
            # # ax = plt.gca()
            # fig.set_figheight(7.8)
            # fig.set_figwidth(16.2)
            # plt.axis("off")
            ax.axis("equal")

            # plot road lines
            for rl in self.env.edge_lines:
                plt.plot(rl[0], rl[1], "k")

            for ra in self.env.edge_areas:
                plt.fill(*ra, "lemonchiffon")

            # plot vehicles
            ego_x = self.env.ego_dynamics["x"]
            ego_y = self.env.ego_dynamics["y"]
            for veh in self.env.all_vehicles:
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
            for veh in self.env.interested_vehs["vehicles"]:
                veh_x = veh["x"]
                veh_y = veh["y"]
                veh_phi = veh["phi"]
                veh_l = veh["l"]
                veh_w = veh["w"]
                veh_type = veh["type"]
                # print("车辆信息", veh)
                # veh_type = 'car_1'
                task2color = {"left": "b", "straight": "#FF8000", "right": "m"}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                    task = "straight"
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")
                    # print(f'veh_l = {veh_l}, veh_w = {veh_w}')

            # plot_interested bicycle
            for veh in self.env.interested_vehs["bikes"]:
                veh_x = veh["x"]
                veh_y = veh["y"]
                veh_phi = veh["phi"]
                veh_l = veh["l"]
                veh_w = veh["w"]
                veh_type = veh["type"]
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {"left": "b", "straight": "#FF8000", "right": "m"}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                    task = "straight"
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")

            # plot_interested person
            for veh in self.env.interested_vehs["persons"]:
                veh_x = veh["x"]
                veh_y = veh["y"]
                veh_phi = veh["phi"]
                veh_l = veh["l"]
                veh_w = veh["w"]
                veh_type = veh["type"]
                # print("车辆信息", veh)
                # veh_type = 'bicycle_1'
                task2color = {"left": "b", "straight": "#FF8000", "right": "m"}

                if is_in_plot_area(veh_x, veh_y, ego_x, ego_y):
                    plot_phi_line(veh_type, veh_x, veh_y, veh_phi, "black")
                    task = "straight"
                    color = task2color[task]
                    draw_rotate_rec(ax, veh_x, veh_y, veh_phi, veh_l, veh_w, color, linestyle=":")

            # plot ego vehicle
            ego_v_x = self.env.ego_dynamics["v_x"]
            ego_v_y = self.env.ego_dynamics["v_y"]
            ego_r = self.env.ego_dynamics["r"]
            ego_x = self.env.ego_dynamics["x"]
            ego_y = self.env.ego_dynamics["y"]
            ego_phi = self.env.ego_dynamics["phi"]
            ego_l = self.env.ego_dynamics["l"]
            ego_w = self.env.ego_dynamics["w"]
            ego_alpha_f = self.env.ego_dynamics["alpha_f"]
            ego_alpha_r = self.env.ego_dynamics["alpha_r"]
            alpha_f_bound = self.env.ego_dynamics["alpha_f_bound"]
            alpha_r_bound = self.env.ego_dynamics["alpha_r_bound"]
            r_bound = self.env.ego_dynamics["r_bound"]

            plot_phi_line("self_car", ego_x, ego_y, ego_phi, "red")
            draw_rotate_rec(ax, ego_x, ego_y, ego_phi, ego_l, ego_w, "red")

            # plot future data (static path)
            tracking_info = self.obs[
                self.env.ego_info_dim : self.env.ego_info_dim
                + self.env.per_tracking_info_dim * (self.env.num_future_data + 1)
            ]
            future_path = tracking_info[self.env.per_tracking_info_dim :]
            for i in range(self.env.num_future_data):
                delta_x, delta_y, delta_phi = future_path[
                    i * self.env.per_tracking_info_dim : (i + 1) * self.env.per_tracking_info_dim
                ]
                path_x, path_y, path_phi = ego_x + delta_x, ego_y + delta_y, ego_phi - delta_phi
                plt.plot(path_x, path_y, "g.")
                plot_phi_line("self_car", path_x, path_y, path_phi, "g")

            delta_, _, _ = tracking_info[:3]
            ax.plot(self.env.ref_path.path[0], self.env.ref_path.path[1], color="g")
            path_info = self.env.ref_path.find_closest_point4edge(
                np.array([ego_x], np.float32), np.array([ego_y], np.float32)
            )

            color = ['blue', 'coral', 'darkcyan']
            try:
                for i, item in enumerate(traj_list):
                    if i == path_index:
                        plt.plot(item.path[0], item.path[1], color=color[i])
                    else:
                        plt.plot(item.path[0], item.path[1], color=color[i], alpha=0.3)
                    indexs, points = item.find_closest_point(np.array([ego_x], np.float32),
                                                             np.array([ego_y], np.float32))
                    path_x, path_y, path_phi = points[0][0], points[1][0], points[2][0]
                    plt.plot(path_x, path_y, color=color[i])
            except Exception:
                pass

            # text to show in figure


            if render_mode == "human":
                # plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                # plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.show()
                plt.pause(0.001)
                return None
            elif render_mode == "save":
                """
                plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.show()
                plt.savefig(save_dir, dpi=500)
                plt.pause(0.001)
                """
                # plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                # plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                fig.canvas.draw()
                img = PIL.Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                img.save(save_dir)
                return None
            elif render_mode == "record":
                # plt.xlim(xmin=path_x - 50, xmax=path_x + 50)
                # plt.ylim(ymin=path_y - 40, ymax=path_y + 40)
                plt.savefig(save_dir, dpi=500)
                return None

        return ploter(render_mode=mode, save_dir=save_dir)


def load_json(exp_dir):
    with open(os.path.join(exp_dir, "config.json"), "r") as f:
         arg_dict = json.load(f)
    parser = argparse.ArgumentParser()
    for key, val in arg_dict.items():
        parser.add_argument("--" + key, default=val)
    return parser.parse_args()

def hierarchical_decision(arg, train_exp_dir, ite, logdir=None):
    if arg.env_id == "HorizonCrossing-v0":
        return CrossingIDC(arg, train_exp_dir, ite, logdir)
    elif arg.env_id == "HorizonMultiLane-v0":
        return MultiLaneIDC(arg, train_exp_dir, ite, logdir)
    else:
        raise KeyError("Unknown env id")


def main():

    iteration = 299000
    exp_dir = "../results/HorizonCrossing-v0/experiment-2021-11-25-19-42-33"
    args = load_json(exp_dir)
    # pprint.pprint(args)
    # change mode to selecting
    args.mode = "selecting"


    trace = {
        "exp_dir": exp_dir,
        "env_id": args.env_id,
        "training_task": args.env_kwargs_training_task,
        "commit_id": args.commit_id,
        "git_branch": args.branch_name,
    }
    pprint.pprint(trace)

    # make logdir
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(os.path.abspath(exp_dir), "select", time_now)
    os.makedirs(logdir, exist_ok=True)

    idc = hierarchical_decision(args, exp_dir, iteration, logdir)
    for i in range(300):
        done = 0
        while not done:
            done = idc.step()
        idc.reset()

if __name__ == "__main__":
    main()
