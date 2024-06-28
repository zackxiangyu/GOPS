#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + veh3dof + mlp + offserial
#  Update Date: 2021-03-05, Gu Ziqing: create example

import sys
sys.path.append('../../')
import argparse
import os
import yaml
import json
import copy
from gops.create_pkg.create_env import create_env
from gops.nodes.launcher import launch_nodes
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.utils.common_utils import change_type

# os.environ["OMP_NUM_THREADS"] = "4"
os.environ['WANDB_API_KEY'] = "8d3bf9907d76a72a3cce256b3903755c2d06dd51"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_veh3dofconti", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="DSAC", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=3, help="seed")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--pre_horizon", type=int, default=20, help="Prediction horizon")
    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDistri",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
    parser.add_argument("--value_min_log_std", type=int, default=-0.1)
    parser.add_argument("--value_max_log_std", type=int, default=4)

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[512, 256, 256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="gelu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=1)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-4)
    ## Special parameters
    parser.add_argument("--delay_update", type=int, default=2)
    parser.add_argument("--TD_bound", type=float, default=10)
    parser.add_argument("--bound", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--auto_alpha", type=bool, default=True)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=40000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    # 4.1. Parameters for off_serial_trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument(
            "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
        )
        # Size of collected samples before training
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=int(1e5))
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=256)
        # Period of sync central policy of each sampler
        parser.add_argument("--sampler_sync_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=32)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=2000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=200)
    parser.add_argument("--wandb_mode", type=str, default="online", help="online or offline")

    # 8. Parallel nodes config path
    parser.add_argument("--config_path", type=str, default='./example.yaml', help="Path to config file")

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    # start_tensorboard(args["save_folder"])

    ################################################

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
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")