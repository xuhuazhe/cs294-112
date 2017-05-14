#!/usr/bin/env bash
declare -a arr=("redo_policy_gradient_soft_1_step" "redo_policy_gradient_soft_1_step_surrogate"
                "redo_DQfD_no_l2_softQ" "redo_exp_policy_grad_weighting" "redo_DQfD_no_l2")

## now loop through the above array
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i --ddqn=True --tag_prefix=harry_doubleQ_ &
done
