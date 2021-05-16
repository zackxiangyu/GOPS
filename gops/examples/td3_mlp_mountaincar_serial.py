#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: gym environment, continuous action, cart pole
#  Update Date: 2020-11-10, Hao SUN: renew env para
#  Update Date: 2020-11-13, Hao SUN：add new ddpg demo
#  Update Date: 2020-12-11, Hao SUN：move buffer to trainer
#  Update Date: 2020-12-12, Hao SUN：move create_* files to create_pkg

#  General Optimal control Problem Solver (GOPS)


import argparse
import copy
import datetime
import json
import os

import numpy as np

from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_buffer import create_buffer
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_evaluator import create_evaluator
from modules.create_pkg.create_sampler import create_sampler
from modules.create_pkg.create_trainer import create_trainer
from modules.utils.utils import change_type

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_mountaincarconti', help='')
    parser.add_argument('--apprfunc', type=str, default='MLP', help='')
    parser.add_argument('--algorithm', type=str, default='DDPG', help='')
    parser.add_argument('--trainer', type=str, default='serial_trainer', help='')

    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_dim', type=int, default=None, help='')
    parser.add_argument('--action_high_limit', type=list, default=None, help='')
    parser.add_argument('--action_low_limit', type=list, default=None, help='')
    parser.add_argument('--action_type', type=str, default='conti', help='')
    parser.add_argument('--is_render', type=bool, default=False)

    # 2. Parameters for approximate function
    parser.add_argument('--value_func_name', type=str, default='', help='')
    parser.add_argument('--value_func_type', type=str, default=parser.parse_args().apprfunc, help='')
    parser.add_argument('--value_hidden_sizes', type=list, default=[256, 256])
    parser.add_argument('--value_hidden_activation', type=str, default='relu', help='')
    parser.add_argument('--value_output_activation', type=str, default='linear', help='')

    parser.add_argument('--policy_func_name', type=str, default='DetermPolicy', help='')
    parser.add_argument('--policy_func_type', type=str, default=parser.parse_args().apprfunc, help='')
    parser.add_argument('--policy_hidden_sizes', type=list, default=[256, 256])
    parser.add_argument('--policy_hidden_activation', type=str, default='relu', help='')
    parser.add_argument('--policy_output_activation', type=str, default='tanh', help='')

    # 3. Parameters for algorithm
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005, help='')
    parser.add_argument('--value_learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--delay_update', type=int, default=1, help='')
    parser.add_argument('--distribution_type', type=str, default='Dirac')

    # 4. Parameters for trainer
    # Parameters for sampler
    parser.add_argument('--sample_batch_size', type=int, default=256, help='')
    parser.add_argument('--sampler_name', type=str, default='mc_sampler')
    parser.add_argument('--noise_params', type=dict,
                        default={'mean': np.array([0], dtype=np.float32), 'std': np.array([1], dtype=np.float32)},
                        help='')
    parser.add_argument('--reward_scale', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--sample_sync_interval', type=int, default=300, help='')
    # Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=1000)
    parser.add_argument('--buffer_max_size', type=int, default=100000)
    # Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=10)
    # Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--apprfunc_save_interval', type=int, default=1000)
    parser.add_argument('--log_save_interval', type=int, default=50)  # reward?

    # get parameter dict
    args = vars(parser.parse_args())
    env = create_env(**args)
    args.obsv_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_high_limit = env.action_space.high
    args.action_low_limit = env.action_space.low

    # Step 2: create algorithm and approximate function
    alg = create_alg(**vars(args)) # create appr_model in algo **vars(args)

    # Step 3: create trainer # create buffer in trainer
    trainer = create_trainer(args.trainer,args,env,alg)

    # start training
    trainer.train()
    print("Training is Done!")

