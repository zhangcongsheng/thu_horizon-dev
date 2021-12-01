import os
import random
import time

import numpy as np
import torch
from environment.env_horizon.misc_ic import worker_id2env_edge


def args2envkwargs(args, object_id=None):
    env_kwargs = {}

    for key, val in vars(args).items():
        if key.startswith("env_kwargs"):
            env_kwargs.update({key[11:]: val})

    if args.env_id == "HorizonCrossing-v0":
        # env_kwargs.update({key[11:]: val})
        assert args.env_kwargs_training_task is not None
        from_edge, to_edge = args.env_kwargs_training_task
        joe, edge = "junction", None
        env_kwargs.update(
            {
                "training_task4": (joe, edge, from_edge, to_edge),
                "training_task2": (from_edge, to_edge),
                "training_task1": edge,
            }
        )
    elif args.env_id == "HorizonMultiLane-v0":
        if object_id is None:
            print("args.id in train/test_script is missed, do not worry.")
            env_edge = "EE6"
        else:
            env_edge = worker_id2env_edge[object_id]
        env_kwargs.update({key[11:]: val})
        # assert args.env_kwargs_training_task is not None
        from_edge, to_edge = None, None
        # joe, edge = 'edge', args.env_kwargs_training_task
        joe, edge = "edge", env_edge
        env_kwargs.update(
            {
                "training_task4": (joe, edge, from_edge, to_edge),
                "training_task2": (from_edge, to_edge),
                "training_task1": edge,
                "env_edge": env_edge,
                "adj_ref_mode": args.adj_ref_mode,
            }
        )
    return env_kwargs


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def random_choice_with_index(obj_list):
    obj_len = len(obj_list)
    random_index = random.choice(list(range(obj_len)))
    random_value = obj_list[random_index]
    return random_value, random_index


def judge_is_nan(list_of_np_or_tensor):
    for m in list_of_np_or_tensor:
        if hasattr(m, "numpy"):  # on Tensor that requires grad
            m_np = m.clone().detach().numpy()
            if np.any(np.isnan(m_np)):
                # nan_index = np.nonzero(m_np != m_np)
                # print('total Nan: {}'.format(len(nan_index)))
                # print('Nan indexs')
                # print(nan_index)
                # np.save("nan_array", m.detach().numpy())
                raise ValueError
        else:
            if np.any(np.isnan(m)):
                # nan_index = np.nonzero(m != m)
                # print('Nan indexs')
                # print('total Nan: {}'.format(len(nan_index)))
                # print(nan_index)
                raise ValueError


class TimerStat:  # 记录处理数据的数目、时间、均值等信息
    def __init__(self, window_size=10):
        self._window_size = window_size
        self._samples = []
        self._units_processed = []
        self._start_time = None
        self._total_time = 0.0
        self.count = 0

    def __enter__(self):
        assert self._start_time is None, "concurrent updates not supported"
        self._start_time = time.time()

    def __exit__(self, type, value, tb):
        assert self._start_time is not None
        time_delta = time.time() - self._start_time
        self.push(time_delta)
        self._start_time = None

    def push(self, time_delta):
        self._samples.append(time_delta)
        if len(self._samples) > self._window_size:
            self._samples.pop(0)
        self.count += 1
        self._total_time += time_delta

    def push_units_processed(self, n):
        self._units_processed.append(n)
        if len(self._units_processed) > self._window_size:
            self._units_processed.pop(0)

    def has_units_processed(self):
        return len(self._units_processed) > 0

    @property
    def mean(self):
        if not self._samples:
            return 0.0
        return float(np.mean(self._samples))

    @property
    def mean_units_processed(self):
        if not self._units_processed:
            return 0.0
        return float(np.mean(self._units_processed))

    @property
    def mean_throughput(self):
        time_total = float(sum(self._samples))
        if not time_total:
            return 0.0
        return float(sum(self._units_processed)) / time_total


def get_git_info():

    commit_id = os.popen("git rev-parse HEAD").read()
    commit_id = commit_id.replace("\n", "").replace("\r", "")

    branch_name = os.popen("git rev-parse --abbrev-ref HEAD").read()
    branch_name = branch_name.replace("\n", "").replace("\r", "")
    return branch_name, commit_id


class AttrDict(dict):
    def __init__(self, init_dict):
        super(AttrDict, self).__init__()
        for key, value in init_dict.items():
            self[key] = value

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


if __name__ == "__main__":
    print(worker_id2env_edge[12])
    print(get_git_info())
