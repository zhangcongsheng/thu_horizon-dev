#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import pprint
import os

import gym
import environment  # noqa: F401
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import torch
    # tf.config.experimental.set_visible_devices([], 'GPU')
    torch.set_num_threads(1)

    def __init__(self, policy_cls, env_id, args, evaluator_id=1):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id, **args2envkwargs(args, evaluator_id))
        self.policy_with_value = policy_cls(self.args)
        self.iteration = 0
        if self.args.mode == 'training' or self.args.mode == 'debug':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        
        # print(self.log_dir, '------------------->')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args, gamma=self.args.gamma)

        self.writer = SummaryWriter(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

        self.episode_counter = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, steps=None, render=True):
        reward_list = []
        reward_info_dict_list = []
        done = 0
        obs = self.env.reset()
        if render and not self.args.eval_save:
            self.env.render()
        elif self.args.eval_save:
            episode_save_dir = self.log_dir + f'/ite{self.iteration}_episode{self.episode_counter}'
            os.makedirs(episode_save_dir, exist_ok=True)
            img_dir = os.path.join(episode_save_dir, 'step{:03d}.png'.format(0))
            self.env.render(mode="save", save_dir=img_dir)

        if steps is not None:
            for _ in range(steps):
                # extract infos for each kind of participants
                start = 0;
                end = self.args.state_ego_dim + self.args.state_track_dim
                obs_ego = obs[start:end]
                start = end;
                end = start + self.args.state_bike_dim
                obs_bike = obs[start:end]
                start = end;
                end = start + self.args.state_person_dim
                obs_person = obs[start:end]
                start = end;
                end = start + self.args.state_veh_dim
                obs_veh = obs[start:end]
                obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
                obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
                obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])

                processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
                    = self.preprocessor.numpy_process_obs_PI(obs_ego, obs_bike, obs_person, obs_veh)
                processed_obs_other = torch.cat([processed_obs_bike, processed_obs_person, processed_obs_veh], dim=0)

                PI_obs_other = torch.sum(self.policy_with_value.compute_PI(processed_obs_other), dim=0)
                processed_obs = torch.cat((processed_obs_ego, PI_obs_other), dim=0)

                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.detach().numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render and not self.args.eval_save:
                    self.env.render()
                elif self.args.eval_save:
                    img_dir = os.path.join(episode_save_dir, 'step{:03d}.png'.format(_ + 1))
                    self.env.render(mode="save", save_dir=img_dir)
                    
                reward_list.append(reward)
        # else:
        #     while not done:
        #         start = 0; end = self.args.state_ego_dim + self.args.state_track_dim
        #         obs_ego = obs[start:end]
        #         start = end; end = start + self.args.state_bike_dim
        #         obs_bike = obs[start:end]
        #         start = end; end = start + self.args.state_person_dim
        #         obs_person = obs[start:end]
        #         start = end; end = start + self.args.state_veh_dim
        #         obs_veh = obs[start:end]
        #         obs_bike = np.reshape(obs_bike, [-1, self.args.per_bike_dim])
        #         obs_person = np.reshape(obs_person, [-1, self.args.per_person_dim])
        #         obs_veh = np.reshape(obs_veh, [-1, self.args.per_veh_dim])
        #         processed_obs_ego, processed_obs_bike, processed_obs_person, processed_obs_veh \
        #             = self.preprocessor.torch_process_obses_PI(obs_ego, obs_bike, obs_person, obs_veh)
        #         processed_obs_other = torch.cat([processed_obs_bike, processed_obs_person, processed_obs_veh], dim=0)
        #
        #         PI_obs_other = torch.sum(self.policy_with_value.compute_PI(processed_obs_other), dim=0)
        #         processed_obs = torch.cat((processed_obs_ego, PI_obs_other), dim=0)
        #
        #         action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
        #         obs, reward, done, info = self.env.step(action.detach().numpy()[0])
        #         reward_info_dict_list.append(info['reward_info'])
        #         if render and not self.args.eval_save:
        #             self.env.render()
        #         elif self.args.eval_save:
        #             img_dir = os.path.join(episode_save_dir,'step{:03d}.png'.format(_ + 1))
        #             self.env.render(mode="save", save_dir=img_dir)
        #         reward_list.append(reward)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        for key in reward_info_dict_list[0].keys():
            info_key = list(map(lambda x: x[key], reward_info_dict_list))
            mean_key = sum(info_key) / len(info_key)
            info_dict.update({key: mean_key})
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n):
        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            self.episode_counter = _
            info_dict = self.run_an_episode(self.args.fixed_steps, self.args.eval_render)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})
        return n_info_dict

    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration):
        with self.eval_timer:
            self.iteration = iteration
            n_info_dict = self.run_n_episode(self.args.num_eval_episode)
            for key, val in n_info_dict.items():
                self.writer.add_scalar("evaluation/{}".format(key), val, global_step=self.iteration)
            for key, val in self.get_stats().items():
                self.writer.add_scalar("evaluation/{}".format(key), val, global_step=self.iteration)
            self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}'.format(self.get_stats()))
            pprint.pprint(n_info_dict)
        self.eval_times += 1


def test_trained_model(model_dir, ppc_params_dir, iteration):
    from train_script import built_AMPC_parser
    from policy import Policy4Horizon
    args = built_AMPC_parser()
    evaluator = Evaluator(Policy4Horizon, args.env_id, args)
    evaluator.load_weights(model_dir, iteration)
    evaluator.load_ppc_params(ppc_params_dir)
    return evaluator.metrics(1000, render=False, reset=False)


def test_evaluator():
    import ray
    ray.init()
    import time
    from train_script import built_parser
    from policy import Policy4Horizon
    args = built_parser('AMPC')
    # evaluator = Evaluator(Policy4Horizon, args.env_id, args)
    # evaluator.run_evaluation(3)
    evaluator = ray.remote(num_cpus=1)(Evaluator).remote(Policy4Horizon, args.env_id, args)
    evaluator.run_evaluation.remote(3)
    time.sleep(10000)


def test_where():
    a = torch.tensor([True, False, True])
    print(f'a = {a.expand(2, -1)}')
    b = torch.ones([2, 2], dtype=torch.int64)
    c = torch.ones([2, 2]).long()
    print(b.type())
    print(c.type())
    d = torch.ones([2])
    print(d[0])


if __name__ == '__main__':
    test_where()
