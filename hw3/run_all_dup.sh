declare -a arr=("torcs_cross_entropy_demo_stage_1_amount(10000,5)" "torcs_dqfd_full_demo_stage_1_amount(10000,5)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(10000,5)" \
                "torcs_cross_entropy_demo_stage_1_amount(10000,6)" "torcs_dqfd_full_demo_stage_1_amount(10000,6)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(10000,6)" \
                "torcs_cross_entropy_demo_stage_1_amount(10000,7)" "torcs_dqfd_full_demo_stage_1_amount(10000,7)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(10000,7)" \
                "torcs_cross_entropy_demo_stage_1_amount(10000,8)" "torcs_dqfd_full_demo_stage_1_amount(10000,8)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(10000,8)" \
                "torcs_cross_entropy_demo_stage_1_amount(10000,9)" "torcs_dqfd_full_demo_stage_1_amount(10000,9)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(10000,9)" \
                "torcs_cross_entropy_demo_stage_1_amount(150000,5)" "torcs_dqfd_full_demo_stage_1_amount(150000,5)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(150000,5)" \
                "torcs_cross_entropy_demo_stage_1_amount(150000,6)" "torcs_dqfd_full_demo_stage_1_amount(150000,6)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(150000,6)" \
                "torcs_cross_entropy_demo_stage_1_amount(150000,7)" "torcs_dqfd_full_demo_stage_1_amount(150000,7)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(150000,7)" \
                "torcs_cross_entropy_demo_stage_1_amount(150000,8)" "torcs_dqfd_full_demo_stage_1_amount(150000,8)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(150000,8)" \
                "torcs_cross_entropy_demo_stage_1_amount(150000,9)" "torcs_dqfd_full_demo_stage_1_amount(150000,9)" "torcs_V_grounding_no_weighting_demo_stage_1_amount(150000,9)")
for i in "${arr[@]}"
do
    python run_dqn_atari.py --config $i &
    sleep 30
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
