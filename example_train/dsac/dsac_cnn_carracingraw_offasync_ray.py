#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + carracingraw + mlp + offasync
#  Update Date: 2024-07-09, zack: create example
import os
import argparse
import json
import multiprocessing
from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.tensorboard_setup import save_tb_to_csv


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_carracingraw", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="DSAC", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=True, help="Disable CUDA")
    parser.add_argument("--seed", default=12345, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument('--value_func_type', type=str, default='CNN')
    # 2.1.1 MLP, CNN, RNN
    parser.add_argument('--value_hidden_activation', type=str, default='gelu')
    parser.add_argument('--value_output_activation', type=str, default='linear')
    parser.add_argument('--value_conv_type', type=str, default='type_2')
    parser.add_argument("--value_min_log_std", type=int, default=-0.1)
    parser.add_argument("--value_max_log_std", type=int, default=5)

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    parser.add_argument('--policy_func_type', type=str, default='CNN')
    parser.add_argument('--policy_hidden_activation', type=str, default='gelu', help='')
    parser.add_argument('--policy_output_activation', type=str, default='linear', help='')
    parser.add_argument('--policy_conv_type', type=str, default='type_2')
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--alpha", type=bool, default=0.2)
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--TD_bound", type=float, default=10)
    parser.add_argument("--bound", default=True)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_async_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=500_000)
    parser.add_argument("--ini_network_dir", type=str, default=None)

    # 4.1. Parameters for off_async_trainer
    parser.add_argument("--num_algs", type=int, default=1, help="number of algs")
    parser.add_argument("--num_samplers", type=int, default=2, help="number of samplers")
    parser.add_argument("--num_buffers", type=int, default=1, help="number of buffers")
    cpu_core_num = multiprocessing.cpu_count()
    num_core_input = (
        parser.parse_known_args()[0].num_algs
        + parser.parse_known_args()[0].num_samplers
        + parser.parse_known_args()[0].num_buffers
        + 2
    )
    if num_core_input > cpu_core_num:
        raise ValueError("The number of core is {}, but you want {}!".format(cpu_core_num, num_core_input))
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10_000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=200_000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=8)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50_000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=5000)
    parser.add_argument("--wandb_mode", type=str, default="online", help="online or offline")
    parser.add_argument("--wandb_project_sup", type=str, default=None, help="Supplementary information for wandb project")

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    #start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    #plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
