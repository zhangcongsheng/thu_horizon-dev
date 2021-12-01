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
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
NAME2WORKERCLS = dict([("OffPolicyWorker", OffPolicyWorker)])
NAME2LEARNERCLS = dict([("AMPC", AMPCLearner)])
NAME2BUFFERCLS = dict([("normal", ReplayBuffer), ("None", None)])
NAME2OPTIMIZERCLS = dict(
    [
        ("OffPolicyAsync", OffPolicyAsyncOptimizer),
        ("SingleProcessOffPolicy", SingleProcessOffPolicyOptimizer),
    ]
)
NAME2POLICIES = dict([("Policy4Horizon", Policy4Horizon)])
NAME2EVALUATORS = dict([("Evaluator", Evaluator), ("None", None)])


def built_AMPC_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="testing")  # training testing debug
    mode = parser.parse_args().mode
    if mode == "testing":
        test_dir = os.path.dirname(__file__) + "./results/HorizonCrossing-v0/experiment-2021-11-24-13-13-05"
        print(test_dir)
        params = json.loads(open(test_dir + "/config.json").read())
        time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        test_log_dir = params["log_dir"] + "/tester/test-{}".format(time_now)
        params.update(
            dict(
                test_dir=test_dir,
                test_iter_list=[299000],
                test_log_dir=test_log_dir,
                num_eval_episode=30,
                eval_log_interval=1,
                eval_render=True,
                eval_save=True,
                fixed_steps=120,
            )
        )
        for key, val in params.items():
            parser.add_argument("-" + key, default=val)
        return parser.parse_args()



def built_parser(alg_name):
    if alg_name == "AMPC":
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
        args.PI_out_dim = (
            args.max_bike_num * env.per_bike_info_dim
            + args.max_person_num * env.per_person_info_dim
            + args.max_veh_num * env.per_veh_info_dim
            + 1
        )
        args.obs_dim, args.act_dim = (
            args.PI_out_dim + args.state_ego_dim + args.state_track_dim - 3,
            act_space.shape[0],
        )  # TODO hard coding
        return args


def main(alg_name):
    args = built_parser(alg_name)
    logger.info("begin training agents with parameter: ")

    if args.mode == "training" or args.mode == "debug":
        if args.mode == "debug":
            args.num_workers = 1
            args.num_learners = 1
            args.num_buffers = 1
        pprint.pprint(vars(args))
        ray.init(object_store_memory=5120 * 1024 * 1024)  # ray.init(object_store_memory=128*1024*1024)
        os.makedirs(args.result_dir, exist_ok=True)
        with open(args.result_dir + "/config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        trainer = Trainer(
            policy_cls=NAME2POLICIES[args.policy_type],
            worker_cls=NAME2WORKERCLS[args.worker_type],
            learner_cls=NAME2LEARNERCLS[args.alg_name],
            buffer_cls=NAME2BUFFERCLS[args.buffer_type],
            optimizer_cls=NAME2OPTIMIZERCLS[args.optimizer_type],
            evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
            args=args,
        )
        if args.model_load_dir is not None:
            logger.info("loading model")
            trainer.load_weights(args.model_load_dir, args.model_load_ite)
        if args.ppc_load_dir is not None:
            logger.info("loading ppc parameter")
            trainer.load_ppc_params(args.ppc_load_dir)
        if args.train_behavioral:
            trainer.train()
        trainer.optimizer.stop()

    elif args.mode == "testing":
        pprint.pprint(vars(args))
        os.makedirs(args.test_log_dir)
        with open(args.test_log_dir + "/test_config.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        tester = Tester(
            policy_cls=NAME2POLICIES[args.policy_type],
            evaluator_cls=NAME2EVALUATORS[args.evaluator_type],
            args=args,
        )
        tester.test()


if __name__ == "__main__":
    main("AMPC")
