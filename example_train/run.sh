#!/bin/bash

# python sac/sac_mlp_humanoidconti_async.py

python parallel/sac_ant/sac_mlp_ant_async.py
python sac/sac_mlp_ant_async.py

python parallel/sac_halfcheetah/sac_mlp_halfcheetah_offasync.py
python sac/sac_mlp_halfcheetah_offasync.py