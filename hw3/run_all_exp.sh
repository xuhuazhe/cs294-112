declare -a arr=("test_test()" "test_test()")

## now loop through the above array
for i in "${arr[@]}"
do
    python run_dqn_atari.py $i &  
done
