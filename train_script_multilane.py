#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: train_script.py
# =====================================

import pprint
import argparse
import datetime
import json
import logging
import os
import gym
import ray

import environment  # noqa: F401
from buffer import ReplayBuffer
from evaluator import Evaluator
from learners.ampc import AMPCLearner
from optimizer import OffPolicyAsyncOptimizer, SingleProcessOffPolicyOptimizer
from policy import Policy4Horizon
from tester import Tester
from trainer import Trainer
from worker import OffPolicyWorker
from utils.misc import TimerStat, args2envkwargs, get_git_info

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
NAME2WORKERCLS = dict([('OffPolicyWorker', OffPolicyWorker)])
NAME2LEARNERCLS = dict([('AMPC', AMPCLearner)])
NAME2BUFFERCLS = dict([('normal', ReplayBuffer), ('None', None)])
NAME2OPTIMIZERCLS = dict([('OffPolicyAsync', OffPolicyAsyncOptimizer),
                          ('SingleProcessOffPolicy', SingleProcessOffPolicyOptimizer)])
NAME2POLICIES = dict([('Policy4Horizon', Policy4Horizon)])
NAME2EVALUATORS = dict([('Evaluator', Evaluator), ('None', None)])


def built_AMPC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training')  # training testing
    mode = parser.parse_args().mode

    if mode == 'testing':
        test_dir = os.path.dirname(__file__) + '/results/HorizonMultiLane-v0/experiment-2021-11-24-18-05-02'
        params = json.loads(open(test_dir + '/config.json').read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params['log_dir'] + '/tester/test-{}'.format(time_now)
        params.update(dict(test_dir=test_dir,
                           test_iter_list=[382000],
                           test_log_dir=test_log_dir,
                           num_eval_episode=20,
                           eval_log_interval=1,
                           eval_render=True,
                           eval_save=True,  # if True, render is canceled, and video is saved
                           fixed_steps=120))
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()

    # code info
    branch_name, commit_id = get_git_info()
    parser.add_argument("--branch_name", type=str, default=branch_name)
    parser.add_argument("--commit_id", type=str, default=commit_id)

    # trainer
    parser.add_argument('--policy_type', type=str, default='Policy4Horizon')
    parser.add_argument('--worker_type', type=str, default='OffPolicyWorker')
    parser.add_argument('--evaluator_type', type=str, default='Evaluator')
    parser.add_argument('--buffer_type', type=str, default='normal')
    parser.add_argument('--optimizer_type', type=str, default='OffPolicyAsync')
    parser.add_argument('--off_policy', type=str, default=True)

    # env
    parser.add_argument('--joe', default='edge')
    parser.add_argument('--env_id', default='HorizonMultiLane-v0')
    parser.add_argument('--env_kwargs_num_future_data', type=int, default=0)
    parser.add_argument('--env_kwargs_training_task', type=str, default='EE6')
    parser.add_argument("--env_kwargs_expected_v", type=float, default=8.0)
    parser.add_argument('--adj_ref_mode', type=str, default='random')
    parser.add_argument('--obs_dim', default=None)
    parser.add_argument('--act_dim', default=None)

    parser.add_argument('--PI_in_dim', type=int, default=None)
    parser.add_argument('--PI_out_dim', type=int, default=None)
    parser.add_argument('--max_bike_num', type=int, default=2)
    parser.add_argument('--max_person_num', type=int, default=4)
    parser.add_argument('--max_veh_num', type=int, default=8)
    parser.add_argument('--state_ego_dim', type=int, default=None)
    parser.add_argument('--state_track_dim', type=int, default=None)
    parser.add_argument('--state_bike_dim', type=int, default=None)
    parser.add_argument('--per_bike_dim', type=int, default=None)
    parser.add_argument('--state_person_dim', type=int, default=None)
    parser.add_argument('--per_person_dim', type=int, default=None)
    parser.add_argument('--state_veh_dim', type=int, default=None)
    parser.add_argument('--per_veh_dim', type=int, default=None)

    # learner
    parser.add_argument('--alg_name', default='AMPC')
    parser.add_argument('--M', type=int, default=1)  # used in model_rollout_for_update in ampc.py
    parser.add_argument('--num_rollout_list_for_policy_update', type=list, default=[25])
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--gradient_clip_norm', type=float, default=10)
    parser.add_argument('--init_punish_factor', type=float, default=300.)  # 广义罚函数法的罚因子
    parser.add_argument('--pf_enlarge_interval', type=int, default=20000)
    parser.add_argument('--pf_amplifier', type=float, default=1.)

    # worker
    parser.add_argument('--batch_size', type=int, default=64, help='default=512')  # 448 = 64 * 7
    parser.add_argument('--worker_log_interval', type=int, default=5)
    parser.add_argument('--explore_sigma', type=float, default=None)

    # buffer
    parser.add_argument('--max_buffer_size', type=int, default=50000)
    parser.add_argument('--replay_starts', type=int, default=256, help='default=3000')
    parser.add_argument('--replay_batch_size', type=int, default=64, help='default=480')
    parser.add_argument('--replay_alpha', type=float, default=0.6)  # maybe useless
    parser.add_argument('--replay_beta', type=float, default=0.4)
    parser.add_argument('--buffer_log_interval', type=int, default=40000)

    # tester and evaluator
    parser.add_argument('--num_eval_episode', type=int, default=2)
    parser.add_argument('--eval_log_interval', type=int, default=1)
    parser.add_argument('--fixed_steps', type=int, default=50)
    parser.add_argument('--eval_render', type=bool, default=False)
    parser.add_argument("--eval_save", type=bool, default=False)
    # policy and model
    parser.add_argument('--value_model_cls', type=str, default='MLP')
    parser.add_argument('--policy_model_cls', type=str, default='MLP')
    parser.add_argument('--num_hidden_layers', type=int, default=2)
    parser.add_argument('--num_hidden_units', type=int, default=256)
    parser.add_argument('--hidden_activation', type=str, default='relu')
    parser.add_argument('--deterministic_policy', default=True, action='store_true')
    parser.add_argument('--policy_out_activation', type=str, default='tanh')
    parser.add_argument('--action_range', type=float, default=None)

    # model for PI_net
    parser.add_argument('--PI_model_cls', type=str, default='MLP')
    parser.add_argument('--PI_num_hidden_layers', type=int, default=2)
    parser.add_argument('--PI_num_hidden_units', type=int, default=256)
    parser.add_argument('--PI_hidden_activation', type=str, default='relu')
    parser.add_argument('--PI_out_activation', type=str, default='linear')

    # preprocessor
    parser.add_argument('--obs_preprocess_type', type=str, default='scale')
    parser.add_argument('--obs_scale', type=list, default=None)
    parser.add_argument('--obs_ego_scale', type=list, default=[0.2, 2., 18., 1 / 30., 1 / 30., 1 / 180.])
    parser.add_argument('--tracking_error_scale', type=list, default=[1., 1 / 15., 0.2])
    parser.add_argument('--future_data_scale', type=list, default=[1., 1., 1 / 15.])
    parser.add_argument('--obs_other_scale', type=list, default=[1 / 7.5, 1 / 30., 0.2, 1 / 15., 1.0])
    parser.add_argument('--reward_preprocess_type', type=str, default='scale')
    parser.add_argument('--reward_scale', type=float, default=0.1)
    parser.add_argument('--reward_shift', type=float, default=0.)

    # optimizer (PABAL)
    parser.add_argument('--max_sampled_steps', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=400001)
    parser.add_argument('--policy_lr_schedule', type=list, default=[3e-4, parser.parse_args().max_iter, 1e-5])
    parser.add_argument('--value_lr_schedule', type=list, default=[8e-4, parser.parse_args().max_iter, 1e-5])
    parser.add_argument('--PI_lr_schedule', type=list, default=[8e-4, parser.parse_args().max_iter, 1e-5])
    parser.add_argument('--num_workers', type=int, default=12, help='default=8')  # 12:12:12 for best
    parser.add_argument('--num_learners', type=int, default=12, help='default=16')
    parser.add_argument('--num_buffers', type=int, default=12, help='default=8')
    parser.add_argument('--max_weight_sync_delay', type=int, default=300)
    parser.add_argument('--grads_queue_size', type=int, default=20)
    parser.add_argument('--grads_max_reuse', type=int, default=0)  # Note: if not 0, then obj_v_grad and pg_grad will be 0
    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)

    # quantization
    parser.add_argument("--task", type=str, default='qat')  # qat: 感知量化训练
    parser.add_argument("--net_type", type=str, default='fp32')  # fp32, int8：网络参数的类型
    parser.add_argument("--train_behavioral", type=bool, default=True)  # False：直接基于训练好的浮点网络做量化训练
    parser.add_argument('--max_ite_quantization', type=int, default=2000)  # 感知量化训练的迭代步数
    parser.add_argument('--init_lr_quantization', type=float, default=1e-5)  # 感知量化训练的学习率

    # IO
    args = parser.parse_args()
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = './results/{env}/experiment-{time}'.format(env=args.env_id, time=time_now)
    parser.add_argument('--result_dir', type=str, default=results_dir)
    parser.add_argument('--log_dir', type=str, default=results_dir + '/logs')
    parser.add_argument('--model_dir', type=str, default=results_dir + '/models')
    parser.add_argument('--model_load_dir', type=str, default=None)
    parser.add_argument('--model_load_ite', type=int, default=None)
    parser.add_argument('--ppc_load_dir', type=str, default=None)

    return parser.parse_args()


def built_parser(alg_name):
    if alg_name == 'AMPC':
        args = built_AMPC_parser()
        env = gym.make(args.env_id, **args2envkwargs(args))
        obs_space, act_space = env.observation_space, env.action_space
        args.state_ego_dim = env.ego_info_dim
        args.state_track_dim = env.per_tracking_info_dim * (env.num_future_data + 1)
        args.state_bike_dim = env.per_bike_info_dim * env.bike_num
        args.state_person_dim = env.per_person_info_dim * env.person_num
        args.state_veh_dim = env.per_veh_info_dim * env.veh_num
        args.per_bike_dim = env.per_bike_info_dim
        args.per_person_dim = env.per_person_info_dim
        args.per_veh_dim = env.per_veh_info_dim
        if args.per_bike_dim == args.per_person_dim == args.per_veh_dim:
            args.PI_in_dim = env.per_veh_info_dim
        else:
            raise ValueError
        args.PI_out_dim = args.max_bike_num * env.per_bike_info_dim + args.max_person_num * env.per_person_info_dim + \
                          args.max_veh_num * env.per_veh_info_dim + 1
        args.obs_dim, args.act_dim = args.PI_out_dim + args.state_ego_dim + args.state_track_dim - 3, act_space.shape[0] # TODO hard coding
        return args


def main(alg_name):
    args = built_parser(alg_name)
    logger.info('begin training agents with parameter: ')
    pprint.pprint(vars(args))
    if args.mode == 'training':
        ray.init(object_store_memory=5120*1024*1024)
        os.makedirs(args.result_dir, exist_ok=True)
        with open(args.result_dir + '/config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(policy_cls=NAME2POLICIES[args.policy_type],
                          worker_cls=NAME2WORKERCLS[args.worker_type],
                          learner_cls=NAME2LEARNERCLS[args.alg_name],
                          buffer_cls=NAME2BUFFERCLS[args.buffer_type],
                          optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
                          evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                          args=args)
        if args.model_load_dir is not None:
            logger.info('loading model')
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        if args.ppc_load_dir is not None:  # preprocessor
            logger.info('loading ppc parameter')
            trainer.load_ppc_params(args.ppc_load_dir)
        if args.train_behavioral:
            trainer.train()
        trainer.optimizer.stop()

    elif args.mode == 'testing':
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + '/test_config.json', 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(policy_cls=NAME2POLICIES[args.policy_type],
                        evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
                        args=args)
        tester.test()


if __name__ == '__main__':
    main('AMPC')
