#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dsac + humanoidconti + mlp + offserial
#  Update Date: 2021-03-05, Wenxuan Wang: create example
import os
import argparse
import json
import yaml
import copy
from gops.create_pkg.create_env import create_env
from gops.nodes.launcher import launch_nodes
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv
from gops.utils.common_utils import change_type

os.environ['WANDB_API_KEY'] = "8d3bf9907d76a72a3cce256b3903755c2d06dd51"


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="gym_carracingraw", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="DSACT", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Disable CUDA")
    parser.add_argument("--seed", default=12345, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    # parser.add_argument("--vector_env_num", type=int, default=8, help="Number of vector envs")
    # parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    parser.add_argument("--gym2gymnasium", type=bool, default=False, help="Convert Gym-style env to Gymnasium-style")
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
    # parser.add_argument("--policy_scheduler", type=json.loads, default={
    #     "name": "LinearLR",
    #     "params": {
    #         "start_factor": 1.0,
    #         "end_factor": 0.0,
    #         "total_iters": 100000,
    #         }
    # })
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
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # parser.add_argument("--num_algs", type=int, default=2, help="number of algs")
    # parser.add_argument("--num_samplers", type=int, default=2, help="number of samplers")
    # parser.add_argument("--num_buffers", type=int, default=1, help="number of buffers")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=500_000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10_000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=200000)
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
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50_000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=5000)
    parser.add_argument("--wandb_mode", type=str, default="offline", help="online or offline")
    
    # 8. Parallel nodes config path
    parser.add_argument("--config_path", type=str, default='/home/dodo/zack/GOPS/example_train/parallel/dsact_carracingraw/example.yaml', help="Path to config file")

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**{**args, "vector_env_num": None})
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
    #plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
