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
#declare -a arr=("enduro_perfectdemo_sal_debug_reward_0_to_1()" "enduro_perfectdemo_sal_debug_reward_0_to_1_hasvaluecritic()")
#declare -a arr=("enduro_perfectdemo_sal_debug_reward_0_to_10()" "enduro_perfectdemo_sal_debug_reward_0_to_0d1()")
#declare -a arr=("enduro_perfectdemo_sal_debug_reward_0_to_2()")
#declare -a arr=("enduro_perfectdemo_sal_debug_reward_0_to_1_divider30()")

:'
declare -a arr=("atari_imperfectdemo_sal('Alien', 0.3, 1)")

for i in "${arr[@]}"
do
    python3 run_dqn_atari.py --config "$i" &
    sleep 30
done
'

#config_name="atari_imperfectdemo_sal"
#config_name="atari_imperfectdemo_cross_ent"
#config_name="atari_imperfectdemo_sal_hasvalue"
#config_name="atari_imperfectdemo_sal_hasvalue_norewardchange"
config_name=$1

declare -a GAMES=("Alien" "Asterix" "Boxing" "Enduro" "Hero" "IceHockey" "Jamesbond" "PrivateEye")
#declare -a GAMES=("Alien" "Enduro" "Jamesbond" "PrivateEye")
#declare -a GAMES=("Boxing" "Hero")
#declare -a IMPERFECT_LEVEL=("0.3" "0.5" "0.7")
declare -a IMPERFECT_LEVEL=("0.0")
declare -a GPU=(1 2 3 4 5 6)

iGPU=0
for g in "${GAMES[@]}"
do
  for level in "${IMPERFECT_LEVEL[@]}"
  do
      gpu_len=${#GPU[@]}
      this_gpu=${GPU[$((iGPU%$gpu_len))]}
      cfg="$config_name('$g', $level, $this_gpu)"
      echo $cfg
      python3 run_dqn_atari.py --config "$cfg" &
      sleep 10
      iGPU=$((iGPU+1))
  done
done
