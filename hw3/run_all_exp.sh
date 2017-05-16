#declare -a arr=("cross_entropy_plot()" "hinge_dqfd_plot()" "hard_Q_plot()" "soft_Q_plot()" "DQfD_no_l2_softQ_plot()" "DQfD_no_l2_plot()" "policy_gradient_soft_1_step_plot()" "exp_policy_grad_weighting_plot()" "exp_advantage_diff_learning_plot()")
#declare -a arr=("collect_demonstration()")
#declare -a arr=("DQFD_no_l2_T()" "DQfD_full_T()")
#declare -a arr=("hard_Q_in_env()" "soft_Q_in_env()")
#declare -a arr=("cross_entropy_dm_finetune_small_explore()" "cross_entropy_dm_finetune_normal_explore()")
#declare -a arr=("cross_entropy_finetune()" "hinge_standard_finetune()" "hard_Q_finetune()")
declare -a arr=("exp_advantage_diff_learning_visualize()" "exp_policy_gradient_visualize()")
## now loop through the above array
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 &
done
