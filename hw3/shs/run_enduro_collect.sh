#!/usr/bin/env bash

# this is intend to work with config_boyuan.py
cd ../

declare -a arr=("dqn_enduro_collect()")
# declare -a arr=("torcs_human_sal()" "torcs_human_cross_entropy()" "torcs_human_dqfd()")

for i in "${arr[@]}"
do
    python3 run_dqn_atari.py --config $i
    sleep 30
done
