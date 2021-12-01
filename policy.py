#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from model import MLPNet
import environment  # noqa: F401

NAME2MODELCLS = dict([('MLP', MLPNet), ])


class Policy4Horizon(nn.Module):
    # tf.config.experimental.set_visible_devices([], 'GPU')
    # tf.config.threading.set_inter_op_parallelism_threads(1)
    # tf.config.threading.set_intra_op_parallelism_threads(1)

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls, PI_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                                          NAME2MODELCLS[self.args.policy_model_cls], \
                                                          NAME2MODELCLS[self.args.PI_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)  # 输出均值和方差（没有用方差）
        policy_init_lr, policy_decay_steps, policy_end_lr = self.args.policy_lr_schedule
        policy_decay_steps = float(policy_decay_steps)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_init_lr)  # name='adam_opt'
        self.policy_lr_schedule = LambdaLR(self.policy_optimizer,
                                           lr_lambda=lambda epoch: (1. - policy_end_lr / policy_init_lr) *
                                                                   (1. - min(epoch, policy_decay_steps) / policy_decay_steps) +
                                                                   policy_end_lr / policy_init_lr)  # Note: self.policy_lr_schedule.step()

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v',
                                     output_activation='softplus')  # Note: softplus
        value_init_lr, value_decay_steps, value_end_lr = self.args.value_lr_schedule
        value_decay_steps = float(value_decay_steps)
        self.obj_value_optimizer = torch.optim.Adam(self.obj_v.parameters(), lr=value_init_lr)  # name='objv_adam_opt'
        self.obj_value_lr_schedule = LambdaLR(self.obj_value_optimizer,
                                              lr_lambda=lambda epoch: (1. - value_end_lr / value_init_lr) *
                                                                      (1. - min(epoch, value_decay_steps) / value_decay_steps) +
                                                                      value_end_lr / value_init_lr)  # Note: self.obj_value_lr_schedule.step()

        # self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v',
        #                              output_activation='softplus')
        # self.con_value_optimizer = torch.optim.Adam(self.con_v.parameters(), lr=value_init_lr)  # name='conv_adam_opt'
        # self.con_value_lr_schedule = LambdaLR(self.con_value_optimizer,
        #                                       lr_lambda=lambda epoch: (1 - value_end_lr / value_init_lr) *
        #                                                               (1 - min(epoch, value_decay_steps) / value_decay_steps) +
        #                                                               value_end_lr / value_init_lr)  # Note: self.con_value_lr_schedule.step()

        # add PI_net
        PI_in_dim, PI_out_dim = self.args.PI_in_dim, self.args.PI_out_dim
        n_hiddens, n_units, hidden_activation = self.args.PI_num_hidden_layers, self.args.PI_num_hidden_units, \
                                                self.args.PI_hidden_activation
        self.PI_net = PI_model_cls(PI_in_dim, n_hiddens, n_units, hidden_activation, PI_out_dim, name='PI_net',
                                   output_activation=self.args.PI_out_activation)
        PI_init_lr, PI_decay_steps, PI_end_lr = self.args.PI_lr_schedule
        PI_decay_steps = float(PI_decay_steps)
        self.PI_optimizer = torch.optim.Adam(self.PI_net.parameters(), lr=PI_init_lr)  # name='PI_adam_opt'
        self.PI_lr_schedule = LambdaLR(self.PI_optimizer,
                                       lr_lambda=lambda epoch: (1. - PI_end_lr / PI_init_lr) *
                                                               (1. - min(epoch, PI_decay_steps) / PI_decay_steps) +
                                                               PI_end_lr / PI_init_lr)  # Note: self.PI_lr_schedule.step()

        self.models = (self.obj_v, self.policy, self.PI_net)
        self.optimizers = (self.obj_value_optimizer, self.policy_optimizer, self.PI_optimizer)
        self.lr_schedules = (self.obj_value_lr_schedule, self.policy_lr_schedule, self.PI_lr_schedule)

    def save_weights(self, save_dir, iteration):
        save_dir = save_dir + '/'

        torch.save(self.obj_v.state_dict(), save_dir + self.obj_v.name + '_ite' + str(iteration))
        torch.save(self.obj_value_optimizer.state_dict(), save_dir + self.obj_v.name + '_adam_opt_ite' + str(iteration))
        torch.save(self.obj_value_lr_schedule.state_dict(), save_dir + self.obj_v.name + '_lr_ite' + str(iteration))

        torch.save(self.policy.state_dict(), save_dir + self.policy.name + '_ite' + str(iteration))
        torch.save(self.policy_optimizer.state_dict(), save_dir + self.policy.name + '_adam_opt_ite' + str(iteration))
        torch.save(self.policy_lr_schedule.state_dict(), save_dir + self.policy.name + '_lr_ite' + str(iteration))

        torch.save(self.PI_net.state_dict(), save_dir + self.PI_net.name + '_ite' + str(iteration))
        torch.save(self.PI_optimizer.state_dict(), save_dir + self.PI_net.name + '_adam_opt_ite' + str(iteration))
        torch.save(self.PI_lr_schedule.state_dict(), save_dir + self.PI_net.name + '_lr_ite' + str(iteration))

    def load_weights(self, load_dir, iteration):
        load_dir = load_dir + '/'

        self.obj_v.load_state_dict(torch.load(load_dir + self.obj_v.name + '_ite' + str(iteration)))
        self.obj_value_optimizer.load_state_dict(torch.load(load_dir + self.obj_v.name + '_adam_opt_ite' + str(iteration)))
        self.obj_value_lr_schedule.load_state_dict(torch.load(load_dir + self.obj_v.name + '_lr_ite' + str(iteration)))

        self.policy.load_state_dict(torch.load(load_dir + self.policy.name + '_ite' + str(iteration)))
        self.policy_optimizer.load_state_dict(torch.load(load_dir + self.policy.name + '_adam_opt_ite' + str(iteration)))
        self.policy_lr_schedule.load_state_dict(torch.load(load_dir + self.policy.name + '_lr_ite' + str(iteration)))

        self.PI_net.load_state_dict(torch.load(load_dir + self.PI_net.name + '_ite' + str(iteration)))
        self.PI_optimizer.load_state_dict(torch.load(load_dir + self.PI_net.name + '_adam_opt_ite' + str(iteration)))
        self.PI_lr_schedule.load_state_dict(torch.load(load_dir + self.PI_net.name + '_lr_ite' + str(iteration)))

    def get_weights(self):
        return [model.state_dict() for model in self.models]

    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].load_state_dict(weight)

    def apply_gradients(self, iteration, grads):
        obj_v_len = len(list(self.obj_v.parameters()))
        pg_len = len(list(self.policy.parameters()))
        obj_v_grad, policy_grad = grads[:obj_v_len], grads[obj_v_len:obj_v_len + pg_len]
        PI_grad = grads[obj_v_len + pg_len:]

        for param, grad in zip(self.obj_v.parameters(), obj_v_grad):
            param._grad = torch.from_numpy(grad)
        for param, grad in zip(self.policy.parameters(), policy_grad):
            param._grad = torch.from_numpy(grad)
        for param, grad in zip(self.PI_net.parameters(), PI_grad):
            param._grad = torch.from_numpy(grad)

        self.obj_value_optimizer.step()
        self.obj_value_lr_schedule.step(iteration)  # 更新学习率
        self.policy_optimizer.step()  # todo: consider delay update in the future
        self.policy_lr_schedule.step(iteration)
        self.PI_optimizer.step()
        self.PI_lr_schedule.step(iteration)

    def compute_mode(self, obs):
        logits = self.policy(obs)  # 最后一层已经计算过tanh
        mean, _ = torch.chunk(logits, chunks=2, dim=-1)  # output the mean
        return self.args.action_range * torch.tanh(mean) if self.args.action_range is not None else mean

    def _logits2dist(self, logits):
        mean, log_std = torch.chunk(logits, chunks=2, dim=-1)
        act_dist = torch.distributions.Normal(mean, torch.exp(log_std))
        # if self.args.action_range is not None:
        #     act_dist = (
        #         self.tfp.distributions.TransformedDistribution(
        #             distribution=act_dist,
        #             bijector=self.tfb.Chain(
        #                 [self.tfb.Affine(scale_identity_multiplier=self.args.action_range),
        #                  self.tfb.Tanh()])
        #         ))
        return act_dist

    def compute_action(self, obs):
        logits = self.policy(obs)
        if self.args.deterministic_policy:
            mean, log_std = torch.chunk(logits, chunks=2, dim=-1)  # output the mean
            return self.args.action_range * torch.tanh(mean) if self.args.action_range is not None else mean, 0.
        else:
            act_dist = self._logits2dist(logits)
            actions = act_dist.sample()
            logps = act_dist.log_prob(actions)
            return actions, logps

    def compute_obj_v(self, obs):
        return torch.squeeze(self.obj_v(obs), dim=1)

    def compute_PI(self, obs):
        return self.PI_net(obs)


def test_policy():
    from train_script import built_AMPC_parser
    args = built_AMPC_parser()
    print(args.obs_dim, args.act_dim)
    args.obs_dim = 3
    args.act_dim = 2
    policy = Policy4Horizon(args)
    print(policy.state_dict())


# def test_policy2():
#     from train_script import built_AMPC_parser
#     import gym
#     args = built_AMPC_parser()
#     env = gym.make('Pendulum-v0')
#     policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
#
#
# def test_policy_with_Qs():
#     from train_script import built_AMPC_parser
#     import gym
#     import numpy as np
#     import tensorflow as tf
#     args = built_AMPC_parser()
#     args.obs_dim = 3
#     env = gym.make('Pendulum-v0')
#     policy_with_value = PolicyWithQs(env.observation_space, env.action_space, args)
#     # print(policy_with_value.policy.trainable_weights)
#     # print(policy_with_value.Qs[0].trainable_weights)
#     obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
#
#     with tf.GradientTape() as tape:
#         acts, _ = policy_with_value.compute_action(obses)
#         Qs = policy_with_value.compute_Qs(obses, acts)[0]
#         print(Qs)
#         loss = tf.reduce_mean(Qs)
#
#     gradient = tape.gradient(loss, policy_with_value.policy.trainable_weights)
#     print(gradient)
#
#
# def test_mlp():
#     import tensorflow as tf
#     import numpy as np
#     policy = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
#                                   tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
#                                   tf.keras.layers.Dense(1, activation='elu')])
#     value = tf.keras.Sequential([tf.keras.layers.Dense(128, input_shape=(4,), activation='elu'),
#                                   tf.keras.layers.Dense(128, input_shape=(3,), activation='elu'),
#                                   tf.keras.layers.Dense(1, activation='elu')])
#     print(policy.trainable_variables)
#     print(value.trainable_variables)
#     with tf.GradientTape() as tape:
#         obses = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float32)
#         obses = tf.convert_to_tensor(obses)
#         acts = policy(obses)
#         a = tf.reduce_mean(acts)
#         print(acts)
#         Qs = value(tf.concat([obses, acts], axis=-1))
#         print(Qs)
#         loss = tf.reduce_mean(Qs)
#
#     gradient = tape.gradient(loss, policy.trainable_weights)
#     print(gradient)


def test_torch_distribution():
    mean = torch.tensor([0., 0.])
    std = torch.tensor([1., 1.])
    base_distribution = torch.distributions.Normal(mean, std)
    action = base_distribution.sample()
    print('action = ', action)


if __name__ == '__main__':
    test_policy()
