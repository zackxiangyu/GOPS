#!/bin/bash

# Add your own wandb API key here to use wandb.
# export WANDB_API_KEY="8d3bf9907d76a72a3cce256b3903755c2d06dd51"


# python sac/sac_mlp_humanoidconti_async.py --max_iteration=600000 --num_samplers=4 --sample_interval=1 --sample_batch_size=20

# python sac/sac_mlp_humanoidconti_async.py --max_iteration=600000 --num_samplers=4 --sample_interval=20 --sample_batch_size=400
# python sac/sac_mlp_humanoidconti_vec_offserial.py --vector_env_num=2
# python sac/sac_mlp_humanoidconti_vec_offserial.py --vector_env_num=4

# python parallel/sac_ant/sac_mlp_ant_async.py
# python sac/sac_mlp_ant_async.py
# python sac/sac_mlp_ant_vec_offserial.py --vector_env_num=2
# python sac/sac_mlp_ant_vec_offserial.py --vector_env_num=4

# python parallel/sac_halfcheetah/sac_mlp_halfcheetah_offasync.py
# python sac/sac_mlp_halfcheetah_offasync.py
# python sac/sac_mlp_halfcheetah_vec_offserial.py --vector_env_num=2
# python sac/sac_mlp_ant_vec_offserial.py --vector_env_num=2
# python sac/sac_mlp_ant_vec_offserial.py --vector_env_num=4
# python sac/sac_mlp_halfcheetah_vec_offserial.py --vector_env_num=4

# python dsact/dsact_cnn_carracingraw_offserial.py --max_iteration=400000
python dsact/dsact_cnn_carracingraw_offasync_ray.py --max_iteration=400000 --num_samplers=4
python parallel/dsact_carracingraw/dsact_cnn_carracingraw_offasync.py --max_iteration=800000 --env_node_num=4

python sac/sac_mlp_humanoidconti_async.py --max_iteration=600000 --num_samplers=4
python parallel/sac_humanoid/sac_mlp_humanoidconti_async.py --max_iteration=600000 --env_node_num=4

python sac/sac_mlp_ant_async.py  --max_iteration=600000 --num_samplers=4
python parallel/sac_ant/sac_mlp_ant_async.py --max_iteration=600000 --env_node_num=4

python sac/sac_mlp_halfcheetah_offasync.py --max_iteration=1500000 --num_samplers=4
python parallel/sac_halfcheetah/sac_mlp_halfcheetah_offasync.py --max_iteration=5000000 --env_node_num=4
