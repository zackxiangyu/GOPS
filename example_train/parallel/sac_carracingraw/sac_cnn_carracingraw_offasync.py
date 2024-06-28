#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for sac + carracingraw + cnn + off_async
#  Update Date: 2024-03-26, Zack


import argparse
import json
import yaml
import copy
import multiprocessing
import os
from gops.create_pkg.create_env import create_env
from gops.nodes.launcher import launch_nodes
from gops.utils.init_args import init_args
from gops.utils.tensorboard_setup import save_tb_to_csv
from gops.utils.common_utils import change_type


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_carracingraw", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="SAC", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Disable CUDA")
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
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="CNN", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    # parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
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
        "--policy_func_type", type=str, default="CNN", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    # parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument('--policy_output_activation', type=str, default='linear', help='')
    parser.add_argument('--policy_conv_type', type=str, default='type_2')
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0001)
    parser.add_argument("--q_learning_rate", type=float, default=0.0001)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0001)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0001)

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
    trainer_type = parser.parse_known_args()[0].trainer
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
    parser.add_argument("--buffer_warm_size", type=int, default=10)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=200_000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=4)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50_000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=5000)
    parser.add_argument("--wandb_mode", type=str, default="online", help="online or offline")
    
    # 8. Parallel nodes config path
    parser.add_argument("--config_path", type=str, default='/home/dodo/zack/GOPS/example_train/parallel/sac_carracingraw/example.yaml', help="Path to config file")


    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    with open(args["config_path"], "r") as f:
        config = yaml.safe_load(f)
        f.close()
    for ns_name, ns_config in config.items():
        ns_config["all_args"] = args
    with open(args["save_folder"] + "/all_config.json", "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(config)), f, ensure_ascii=False, indent=4)
    launch_nodes(config)

    ################################################
    # Plot and save training figures
    # plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")