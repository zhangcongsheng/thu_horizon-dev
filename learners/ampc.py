#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import numpy as np
import torch
import environment  # noqa: F401

# from environment.env_mix_single.dynamics_and_models import EnvironmentModel
from environment.env_horizon.dynamics_and_models import environment_model
from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs, judge_is_nan

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AMPCLearner(object):
    import torch
    # tf.config.experimental.set_visible_devices([], 'GPU') #屏蔽GPU，直接注释掉，pytorch默认CPU
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    torch.set_num_threads(1)

    def __init__(self, policy_cls, args, learner_id):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = environment_model(**args2envkwargs(args, learner_id))
        # logger.warning(self.model)
        self.preprocessor = Preprocessor((self.args.obs_dim, ), self.args.obs_preprocess_type, self.args.reward_preprocess_type,
                                         self.args.reward_scale, self.args.reward_shift, args=self.args,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_states(self, processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad):
        # print('1', processed_obses_ego.shape, processed_obses_bike.shape,
        #       processed_obses_person.shape, processed_obses_veh.shape)
        PI_obses_bike = self.policy_with_value.compute_PI(processed_obses_bike)
        PI_obses_person = self.policy_with_value.compute_PI(processed_obses_person)
        PI_obses_veh = self.policy_with_value.compute_PI(processed_obses_veh)
        # print('2', PI_obses_bike.shape, PI_obses_person.shape, PI_obses_veh.shape)
        PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum = [], [], []

        for i in range(len(processed_obses_ego)):
            PI_obses_bike_sum.append(self.torch.sum(PI_obses_bike[i * self.args.max_bike_num: (i+1) * self.args.max_bike_num, :],
                                                        keepdim=True, dim=0))
            PI_obses_person_sum.append(self.torch.sum(PI_obses_person[i * self.args.max_person_num: (i+1) * self.args.max_person_num, :],
                                                          keepdim=True, dim=0))
            PI_obses_veh_sum.append(self.torch.sum(PI_obses_veh[i * self.args.max_veh_num: (i+1) * self.args.max_veh_num, :],
                                                       keepdim=True, dim=0))
        PI_obses_bike_sum = self.torch.cat(PI_obses_bike_sum, dim=0)
        PI_obses_person_sum = self.torch.cat(PI_obses_person_sum, dim=0)
        PI_obses_veh_sum = self.torch.cat(PI_obses_veh_sum, dim=0)
        # print('3', PI_obses_bike_sum.shape, PI_obses_person_sum.shape, PI_obses_veh_sum.shape)
        if not grad:
            PI_obses_bike_sum = PI_obses_bike_sum.detach()
            PI_obses_person_sum = PI_obses_person_sum.detach()
            PI_obses_veh_sum = PI_obses_veh_sum.detach()
        if self.args.per_bike_dim == self.args.per_person_dim == self.args.per_veh_dim:
            PI_obses_other_sum = PI_obses_bike_sum + PI_obses_person_sum + PI_obses_veh_sum
        else:
            PI_obses_other_sum = self.torch.cat([PI_obses_bike_sum, PI_obses_person_sum, PI_obses_veh_sum],dim=1)
        processed_obses = self.torch.cat((processed_obses_ego, PI_obses_other_sum), dim=1)
        # print('4', processed_obses.shape)
        return processed_obses

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs_ego': batch_data[0].astype(np.float32),
                           'batch_obs_bike': batch_data[1].astype(np.float32),
                           'batch_obs_person': batch_data[2].astype(np.float32),
                           'batch_obs_veh': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def punish_factor_schedule(self, ite):
        init_pf = self.args.init_punish_factor
        interval = self.args.pf_enlarge_interval
        amplifier = self.args.pf_amplifier
        pf = init_pf * self.torch.pow(amplifier, ite//interval)
        return pf

    def model_rollout_for_update(self, start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh, ite, mb_ref_index):
        start_obses_ego = start_obses_ego.repeat(self.M, 1)
        start_obses_bike = start_obses_bike.repeat(self.M, 1)
        start_obses_person = start_obses_person.repeat(self.M, 1)
        start_obses_veh = start_obses_veh.repeat(self.M, 1)  # self.tf.tile(start_obses_veh, [self.M, 1])

        self.model.reset(start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh, mb_ref_index)

        rewards_sum = self.torch.zeros((start_obses_ego.shape[0],))
        punish_terms_for_training_sum = self.torch.zeros((start_obses_ego.shape[0],))
        real_punish_terms_sum = self.torch.zeros((start_obses_ego.shape[0],))
        veh2veh4real_sum = self.torch.zeros((start_obses_ego.shape[0],))
        veh2road4real_sum = self.torch.zeros((start_obses_ego.shape[0],))
        veh2bike4real_sum = self.torch.zeros((start_obses_ego.shape[0],))
        veh2person4real_sum = self.torch.zeros((start_obses_ego.shape[0],))

        pf = self.punish_factor_schedule(ite)
        obses_ego, obses_bike, obses_person, obses_veh = start_obses_ego, start_obses_bike, start_obses_person, start_obses_veh
        processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh \
            = self.preprocessor.torch_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
        # no supplement vehicle currently
        processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh, grad=True)
        obj_v_pred = self.policy_with_value.compute_obj_v(processed_obses)

        for s in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses_ego, processed_obses_bike, processed_obses_person, processed_obses_veh \
                = self.preprocessor.torch_process_obses_PI(obses_ego, obses_bike, obses_person, obses_veh)
            processed_obses = self.get_states(processed_obses_ego, processed_obses_bike, processed_obses_person,
                                              processed_obses_veh, grad=False)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses_ego, obses_bike, obses_person, obses_veh, rewards, punish_terms_for_training, real_punish_term, \
                veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, reward_dict = self.model.rollout_out(actions)
            # print(f'step = {s}, obs_ego = {obses_ego[1, :]}')
            rewards_sum += self.preprocessor.torch_process_rewards(rewards)
            punish_terms_for_training_sum += self.args.reward_scale * punish_terms_for_training
            real_punish_terms_sum += self.args.reward_scale * real_punish_term
            veh2veh4real_sum += self.args.reward_scale * veh2veh4real
            veh2road4real_sum += self.args.reward_scale * veh2road4real
            veh2bike4real_sum += self.args.reward_scale * veh2bike4real
            veh2person4real_sum += self.args.reward_scale * veh2person4real
        # judge_is_nan((obj_v_pred,))
        # judge_is_nan((rewards_sum,))
        # obj v loss
        obj_v_loss = self.torch.mean(self.torch.square(obj_v_pred - (-rewards_sum).detach()))
        # con_v_loss = self.torch.mean(self.torch.square(con_v_pred - self.tf.stop_gradient(real_punish_terms_sum)))
        # judge_is_nan((obj_v_loss,))
        # pg loss
        obj_loss = -self.torch.mean(rewards_sum)
        # judge_is_nan([obj_loss, ])
        punish_term_for_training = self.torch.mean(punish_terms_for_training_sum)
        punish_loss = pf.detach() * punish_term_for_training
        pg_loss = obj_loss + punish_loss

        real_punish_term = self.torch.mean(real_punish_terms_sum)
        veh2veh4real = self.torch.mean(veh2veh4real_sum)
        veh2road4real = self.torch.mean(veh2road4real_sum)
        veh2bike4real = self.torch.mean(veh2bike4real_sum)
        veh2person4real = self.torch.mean(veh2person4real_sum)

        return obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf

    def forward_and_backward(self, mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, ite, mb_ref_index):
        obj_v_loss, obj_loss, punish_term_for_training, punish_loss, pg_loss, \
        real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf\
            = self.model_rollout_for_update(mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, ite, mb_ref_index)
        self.policy_with_value.policy.zero_grad()
        self.policy_with_value.obj_v.zero_grad()
        self.policy_with_value.PI_net.zero_grad()

        pg_loss.backward()
        obj_v_loss.backward()
        return obj_v_loss, obj_loss, \
               punish_term_for_training, punish_loss, pg_loss,\
               real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf

    # def export_graph(self, writer):  # 导出TF计算图，直接鲨了
    #     mb_obs = self.batch_data['batch_obs']
    #     self.tf.summary.trace_on(graph=True, profiler=False)
    #     self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
    #                               self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
    #     with writer.as_default():
    #         self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs_ego = self.torch.FloatTensor(self.batch_data['batch_obs_ego'])
        mb_obs_bike = self.torch.FloatTensor(self.batch_data['batch_obs_bike'])
        mb_obs_person = self.torch.FloatTensor(self.batch_data['batch_obs_person'])
        mb_obs_veh = self.torch.FloatTensor(self.batch_data['batch_obs_veh'])
        iteration = self.torch.tensor(iteration, dtype=self.torch.int32)
        mb_ref_index = self.torch.tensor(self.batch_data['batch_ref_index'], dtype=self.torch.int32)

        with self.grad_timer:
            obj_v_loss, obj_loss, \
            punish_term_for_training, punish_loss, pg_loss, \
            real_punish_term, veh2veh4real, veh2road4real, veh2bike4real, veh2person4real, pf =\
                self.forward_and_backward(mb_obs_ego, mb_obs_bike, mb_obs_person, mb_obs_veh, iteration, mb_ref_index)
            # judge_is_nan((obj_loss,))
            pg_grad_norm = self.torch.nn.utils.clip_grad_norm_(self.policy_with_value.policy.parameters(), self.args.gradient_clip_norm)
            obj_v_grad_norm = self.torch.nn.utils.clip_grad_norm_(self.policy_with_value.obj_v.parameters(), self.args.gradient_clip_norm)
            PI_net_grad_norm = self.torch.nn.utils.clip_grad_norm_(self.policy_with_value.PI_net.parameters(), self.args.gradient_clip_norm)
            pg_grad = [p.grad.numpy() for p in self.policy_with_value.policy.parameters()]
            obj_v_grad = [p.grad.numpy() for p in self.policy_with_value.obj_v.parameters()]
            PI_net_grad = [p.grad.numpy() for p in self.policy_with_value.PI_net.parameters()]

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            obj_loss=obj_loss.detach().numpy(),
            punish_term_for_training=punish_term_for_training.detach().numpy(),
            real_punish_term=real_punish_term.detach().numpy(),
            veh2veh4real=veh2veh4real.detach().numpy(),
            veh2road4real=veh2road4real.detach().numpy(),
            veh2bike4real=veh2bike4real.detach().numpy(),
            veh2person4real=veh2person4real.detach().numpy(),
            punish_loss=punish_loss.detach().numpy(),
            pg_loss=pg_loss.detach().numpy(),
            obj_v_loss=obj_v_loss.detach().numpy(),
            # con_v_loss=con_v_loss.detach().numpy(),
            punish_factor=pf.numpy(),
            pg_grads_norm=pg_grad_norm.numpy(),
            obj_v_grad_norm=obj_v_grad_norm.numpy(),
            PI_net_grad_norm=PI_net_grad_norm.numpy()
            # con_v_grad_norm=con_v_grad_norm.numpy()
        ))

        grads = obj_v_grad + pg_grad + PI_net_grad
        return grads #？


if __name__ == '__main__':
    pass
