#!/usr/bin/env bash

# this is intend to work with config_yang2.py
cd ../

declare -a arr=("enduro_machine_cross_entropy()")

for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i
    sleep 30
done
