#!/usr/bin/env bash
#declare -a arr=("redo_policy_gradient_soft_1_step" "redo_DQfD_no_l2_softQ" "redo_exp_policy_grad_weighting" "redo_DQfD_no_l2" "exp_advantage_diff_learning")
#declare -a arr=("redo_policy_gradient_soft_1_step" "redo_DQfD_no_l2_softQ" "redo_exp_policy_grad_weighting" "redo_DQfD_no_l2" "exp_advantage_diff_learning" "yang_cross_entropy" "yang_hinge_dqfd")
# configs for visualize the signs
#declare -a arr=("redo_policy_gradient_soft_1_step" "redo_exp_policy_grad_weighting"  "exp_advantage_diff_learning")
declare -a arr=("exp_advantage_diff_learning")

## now loop through the above array
for i in "${arr[@]}"
do
    #python run_dqn_atari.py --config $i --ddqn=False --soft_Q_alpha=1.0 --tag_prefix=newdata_ &
    #python run_dqn_atari.py --config $i --ddqn=False --soft_Q_alpha=0.1 --tag_prefix=harrydata_ --demo_file_path=/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro.h5 &
    python run_dqn_atari.py --config $i --tag_prefix=viz_harrydata_correctV_ --demo_file_path=/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro.h5 &
    #python run_dqn_atari.py --config $i --tag_prefix=viz_newdata_correctV_ --demo_file_path=/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-00.h5,/backup/hxu/cs294-112/hw3/link_data/dmformat_demo/enduro-dm-egs-01.h5 &
done
