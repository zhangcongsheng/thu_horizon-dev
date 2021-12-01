#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/4
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: tester.py
# =====================================

import logging
import os
from tools.image2video import image2video_with_num

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tester(object):
    def __init__(self, policy_cls, evaluator_cls, args):
        self.args = args
        self.evaluator = evaluator_cls(policy_cls, self.args.env_id, self.args)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.evaluator.evaluate_saved_model(model_load_dir, ppc_params_load_dir, iteration)

    def test(self):
        logger.info('testing beginning')
        for ite in self.args.test_iter_list:
            logger.info('testing {}-th iter model'.format(ite))
            model_load_dir = self.args.test_dir + '/models'
            ppc_params_load_dir = self.args.test_dir + '/models'
            self.evaluate_saved_model(model_load_dir, ppc_params_load_dir, ite)
            self.evaluator.run_evaluation(ite)

        if self.args.eval_save:
            logger.info('convert images to video')
            os.chdir(self.evaluator.log_dir + f'/ite{self.evaluator.iteration}_episode0')
            for ite in self.args.test_iter_list:
                for epi in range(self.args.num_eval_episode):
                    image_forder = f"../ite{ite}_episode" + str(epi)
                    image2video_with_num(image_forder, ite, epi)

