#!/usr/bin/env bash

#declare -a arr=("torcs_V_grounding_no_weighting_demo_stage_1_amount_revive('300000','0')")
#declare -a arr=("torcs_V_grounding_no_weighting_demo_stage_1_amount_revive_piv('300000','0')")
#declare -a arr=("torcs_V_grounding_no_weighting_demo_stage_1_amount_advantage('300000','0')")
#declare -a arr=("torcs_V_grounding_no_weighting_inenv_stage_2_rerun_advantage('')")
#declare -a arr=("torcs_V_grounding_no_weighting_inenv_stage_2_rerun_Critic_Weighting('1.0')")
declare -a arr=("torcs_V_grounding_no_weighting_demo_stage_1_amount_advantage_nocritic('300000','0')")


for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i &
    sleep 30
done
