#declare -a arr=("cross_entropy_plot()" "hinge_dqfd_plot()" "hard_Q_plot()" "soft_Q_plot()" "DQfD_no_l2_softQ_plot()" "DQfD_no_l2_plot()" "policy_gradient_soft_1_step_plot()" "exp_policy_grad_weighting_plot()" "exp_advantage_diff_learning_plot()")
#declare -a arr=("policy_gradient_soft_1_step_finetune_small_explore_May16()")
#declare -a arr=("DQFD_no_l2_T()" "DQfD_full_T()")
#declare -a arr=("hard_Q_in_env()" "soft_Q_in_env()")
#declare -a arr=("cross_entropy_dm_finetune_small_explore()" "cross_entropy_dm_finetune_normal_explore()")
#declare -a arr=("cross_entropy_finetune()" "hinge_standard_finetune()" "hard_Q_finetune()")
#declare -a arr=("policy_gradient_soft_1_step_finetune_small_explore_May16" "policy_gradient_soft_1_step_finetune_normal_explore_May16" "adv_learn_finetune_small_explore_May16" "adv_learn_finetune_normal_explore_May16" "exp_policy_grad_weighting_finetune_normal_explore_May16()" "exp_policy_grad_weighting_finetune_small_explore_May16()" "DQfD_no_l2_finetune_small_explore_human_May16()" "DQfD_no_l2_finetune_normal_explore_human_May16()")
## now loop through the above arrayi
#declare -a arr=("dueling_net_double_Q_eval")
#declare -a arr=("urex_multistep")
#declare -a arr=("collect_torcs_demonstration_3e5")
#declare -a arr=("torcs_cross_entropy_demo")
#declare -a arr=("torcs_cross_entropy_demo" "torcs_hinge_dqfd_demo" "torcs_hard_Q_demo" "torcs_soft_Q_demo" "torcs_dqfd_full_demo" "torcs_V_grounding_demo" "torcs_V_grounding_no_weighting_demo" "torcs_PG_demo" "torcs_PG_no_weighting_demo")
declare -a arr=("torcs_only_V_no_weighting_demo" "torcs_Q_grounding_no_weighting_demo")
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i &
    sleep 10
    #python run_dqn_atari.py --config $i --dataset_size=300000 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_50000.0_0.3' &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --core_num=1 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_150000.0_0.7' &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --core_num=2 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_250000.0_0.7' &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --core_num=3 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_50000.0_0.3' &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --core_num=0 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_150000.0_0.3' &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --core_num=1 --demo_file_path='/data/hxu/cs294-112/hw3/link_data/bad_demo_250000.0_0.3' &
	#python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 --bad_starts=1.5e5 --tiny_explore=0.7 --core_num=1 &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 --bad_starts=0.5e5 --tiny_explore=0.7 --core_num=2 &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 --bad_starts=2.5e5 --tiny_explore=0.3 --core_num=3 &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 --bad_starts=1.5e5 --tiny_explore=0.3 --core_num=0 &
    #python run_dqn_atari.py --config $i --dataset_size=300000 --bad_portion=0 --bad_starts=0.5e5 --tiny_explore=0.3 --core_num=1 &
done
