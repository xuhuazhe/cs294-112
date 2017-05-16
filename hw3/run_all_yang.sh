#!/usr/bin/env bash
declare -a arr=("redo_policy_gradient_soft_1_step"
                "redo_DQfD_no_l2_softQ" "redo_exp_policy_grad_weighting" "redo_DQfD_no_l2" "exp_advantage_diff_learning")

## now loop through the above array
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i --ddqn=False --soft_Q_alpha=1.0 --tag_prefix=newdata_ &
done
