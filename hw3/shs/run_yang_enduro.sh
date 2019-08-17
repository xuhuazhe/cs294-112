#!/usr/bin/env bash

# requirements
# pip3 install gym[atari], tensorflow-gpu

# this is intend to work with config_yang2.py
cd ../

#declare -a arr=("enduro_perfectdemo_cross_entropy()" "enduro_perfectdemo_sal()")
declare -a arr=("enduro_perfectdemo_sal()")

for i in "${arr[@]}"
do
    python3 run_dqn_atari.py --config $i &
    sleep 30
done
