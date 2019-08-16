#!/usr/bin/env bash

# this is intend to work with config_yang2.py
cd ../

declare -a arr=("torcs_human_sal()")
# declare -a arr=("torcs_human_sal()" "torcs_human_cross_entropy()" "torcs_human_dqfd()")

for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i &
    sleep 30
done
