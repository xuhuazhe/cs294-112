#!/usr/bin/env bash

# requirements
# pip3 install gym[atari], tensorflow-gpu

# this is intend to work with config_yang2.py
cd ../

#declare -a arr=("enduro_perfectdemo_cross_entropy()" "enduro_perfectdemo_sal()")
#declare -a arr=("enduro_perfectdemo_sal_divider5()")
#declare -a arr=("enduro_perfectdemo_sal_divider5_vizbellmanerror()")
#declare -a arr=("enduro_perfectdemo_sal_maxclip()" "enduro_perfectdemo_sal_minclip()")
#declare -a arr=("enduro_perfectdemo_sal_minclip_novaluecritic()")
#declare -a arr=("enduro_perfectdemo_sal_novaluecritic()")
#declare -a arr=("enduro_perfectdemo_sal_debug_with_1()")
declare -a arr=("enduro_perfectdemo_sal_debug_reward_0_to_1()" "enduro_perfectdemo_sal_debug_reward_0_to_1_hasvaluecritic()")


for i in "${arr[@]}"
do
    python3 run_dqn_atari.py --config $i &
    sleep 30
done
