#declare -a arr=("cross_entropy_T()" "hinge_dqfd_T()" "hinge_standard_T()" "hard_Q_T()" "soft_Q_T()" "DQFD_no_l2_T()" "DQfD_full_T()" "policy_gradient_soft_1_step_new_T()" "policy_gradient_soft_1_step_T()")
#declare -a arr=("collect_demonstration()")
#declare -a arr=("DQFD_no_l2_T()" "DQfD_full_T()")
#declare -a arr=("hard_Q_in_env()" "soft_Q_in_env()")
declare -a arr=("DQfD_full_finetune()")
#declare -a arr=("cross_entropy_finetune()" "hinge_standard_finetune()" "hard_Q_finetune()")
## now loop through the above array
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 &
done
