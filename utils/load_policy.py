import argparse
import json
import os
import gym
import numpy as np
import torch

from policy import Policy4Horizon
from preprocessor import Preprocessor


class LoadPolicy(object):
    def __init__(self, args, exp_dir, iter):
        self.args = args
        model_dir = os.path.join(exp_dir, 'models')
        
        self.policy = Policy4Horizon(self.args)
        self.policy.load_weights(model_dir, iter)

        self.preprocessor = Preprocessor(
            ob_shape=(self.args.obs_dim,),
            obs_ptype=self.args.obs_preprocess_type,
            rew_ptype=self.args.reward_preprocess_type,
            rew_scale=self.args.reward_scale,
            rew_shift=self.args.reward_shift,
            args=self.args,
            gamma=self.args.gamma
        )

    # @tf.function
    # def run(self, obs):
    #     processed_obs = self.preprocessor.np_process_obses(obs)
    #     action, _ = self.policy.compute_action(processed_obs[np.newaxis, :])
    #     return action[0]
    #
    # @tf.function
    # def obj_value(self, obs):
    #     processed_obs = self.preprocessor.np_process_obses(obs)
    #     value = self.policy.compute_obj_v(processed_obs[np.newaxis, :])
    #     return value

    def run_batch(self, obses_ego, obses_bike, obses_person, obses_veh):
        processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh = self.preprocessor.torch_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
        processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person,
                                          processed_obses_veh, grad=False)
        actions, _ = self.policy.compute_action(processed_obses)
        return actions.detach()

    def obj_value_batch(self, obses_ego, obses_bike, obses_person, obses_veh):
        processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh = self.preprocessor.torch_process_obses_PI(
            obses_ego, obses_bike, obses_person, obses_veh)
        processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person,
                                          processed_obses_veh, grad=False)
        values = self.policy.compute_obj_v(processed_obses)
        return values.detach()

    def get_states(self, processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad):
        PI_obses_bike = self.policy.compute_PI(processed_obses_bike)
        PI_obses_person = self.policy.compute_PI(processed_obses_person)
        PI_obses_veh = self.policy.compute_PI(processed_obses_veh)
        PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum = [], [], []

        for i in range(len(processed_obses_ego)):
            PI_obses_bike_sum.append(torch.sum(PI_obses_bike[i * self.args.max_bike_num: (i+1) * self.args.max_bike_num, :],
                                                        keepdim=True, dim=0))
            PI_obses_person_sum.append(torch.sum(PI_obses_person[i * self.args.max_person_num: (i+1) * self.args.max_person_num, :],
                                                          keepdim=True, dim=0))
            PI_obses_veh_sum.append(torch.sum(PI_obses_veh[i * self.args.max_veh_num: (i+1) * self.args.max_veh_num, :],
                                                       keepdim=True, dim=0))
        PI_obses_bike_sum = torch.cat(PI_obses_bike_sum, dim=0)
        PI_obses_person_sum = torch.cat(PI_obses_person_sum, dim=0)
        PI_obses_veh_sum = torch.cat(PI_obses_veh_sum, dim=0)
        if not grad:
            PI_obses_bike_sum = PI_obses_bike_sum.detach()
            PI_obses_person_sum = PI_obses_person_sum.detach()
            PI_obses_veh_sum = PI_obses_veh_sum.detach()
        if self.args.per_bike_dim == self.args.per_person_dim == self.args.per_veh_dim:
            PI_obses_other_sum = PI_obses_bike_sum + PI_obses_person_sum + PI_obses_veh_sum
        else:
            PI_obses_other_sum = torch.cat([PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum],dim=1)
        processed_obses = torch.cat((processed_obses_ego, PI_obses_other_sum), dim=1)
        # print('4', processed_obses.shape)
        return processed_obses
