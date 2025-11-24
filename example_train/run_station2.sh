#!/bin/bash

# Add your own wandb API key here to use wandb.
export WANDB_API_KEY="8d3bf9907d76a72a3cce256b3903755c2d06dd51"


## Humanoid
python sac/sac_mlp_humanoidconti_async.py --max_iteration=1000000 --num_samplers=2 --wandb_project_sup='station2(3-party test)'
python parallel/sac_humanoid/sac_mlp_humanoidconti_async.py --max_iteration=1000000 --env_node_num=2 --wandb_project_sup='station2(3-party test)'
python sac/sac_mlp_humanoidconti_async.py --max_iteration=1000000 --num_samplers=4 --wandb_project_sup='station2(3-party test)'
python parallel/sac_humanoid/sac_mlp_humanoidconti_async.py --max_iteration=1000000 --env_node_num=4 --wandb_project_sup='station2(3-party test)'


## Ant
python sac/sac_mlp_ant_async.py  --max_iteration=1000000 --num_samplers=2 --wandb_project_sup='station2(3-party test)'
python parallel/sac_ant/sac_mlp_ant_async.py --max_iteration=1000000 --env_node_num=2 --wandb_project_sup='station2(3-party test)'
python sac/sac_mlp_ant_async.py  --max_iteration=1000000 --num_samplers=4 --wandb_project_sup='station2(3-party test)'
python parallel/sac_ant/sac_mlp_ant_async.py --max_iteration=1000000 --env_node_num=4 --wandb_project_sup='station2(3-party test)'


## HalfCheetah
python sac/sac_mlp_halfcheetah_offasync.py --max_iteration=1500000 --num_samplers=2 --wandb_project_sup='station2(3-party test)'
python parallel/sac_halfcheetah/sac_mlp_halfcheetah_offasync.py --max_iteration=1000000 --env_node_num=2 --wandb_project_sup='station2(3-party test)'
python sac/sac_mlp_halfcheetah_offasync.py --max_iteration=1500000 --num_samplers=4 --wandb_project_sup='station2(3-party test)'
python parallel/sac_halfcheetah/sac_mlp_halfcheetah_offasync.py --max_iteration=1000000 --env_node_num=4 --wandb_project_sup='station2(3-party test)'


## CarRacingRaw (dsac)
python dsac/dsac_cnn_carracingraw_offasync_ray.py --max_iteration=400000 --num_samplers=2 --wandb_project_sup='station2(3-party test)'
python parallel/dsac_carracingraw/dsac_cnn_carracingraw_offasync.py --max_iteration=600000 --env_node_num=2 --wandb_project_sup='station2(3-party test)'
python dsac/dsac_cnn_carracingraw_offasync_ray.py --max_iteration=400000 --num_samplers=4 --wandb_project_sup='station2(3-party test)'
python parallel/dsac_carracingraw/dsac_cnn_carracingraw_offasync.py --max_iteration=600000 --env_node_num=4 --wandb_project_sup='station2(3-party test)'
