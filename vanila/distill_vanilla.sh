#!/bin/bash

LOGDIR="distill"
#TASK=("Pong" "Breakout" "Enduro" "Qbert" "MsPacman" "Seaquest" "SpaceInvaders" )

TASK=("BeamRider" "Freeway" "Riverraid")

resume_path=(
#"/root/RL_IoT_distillation/vanila/log/vanilla/PongNoFrameskip-v4/dqn/seed_0_1206_063927-PongNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/BreakoutNoFrameskip-v4/dqn/seed_0_1206_092731-BreakoutNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/EnduroNoFrameskip-v4/dqn/seed_0_1207_063407-EnduroNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/QbertNoFrameskip-v4/dqn/seed_0_1208_032214-QbertNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/MsPacmanNoFrameskip-v4/dqn/seed_0_1208_184113-MsPacmanNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/SeaquestNoFrameskip-v4/dqn/seed_0_1209_094702-SeaquestNoFrameskip_v4_vanilla/policy.pth"
#"/root/RL_IoT_distillation/vanila/log/vanilla/SpaceInvadersNoFrameskip-v4/dqn/seed_0_1210_023838-SpaceInvadersNoFrameskip_v4_vanilla/policy.pth"
"/root/RL_IoT_distillation/vanila/log/vanilla/BeamRiderNoFrameskip-v4/dqn/seed_0_1210_181838-BeamRiderNoFrameskip_v4_vanilla/policy.pth"
"/root/RL_IoT_distillation/vanila/log/vanilla/FreewayNoFrameskip-v4/dqn/seed_0_1211_101955-FreewayNoFrameskip_v4_vanilla/policy.pth"
"/root/RL_IoT_distillation/vanila/log/vanilla/RiverraidNoFrameskip-v4/dqn/seed_0_1212_034510-RiverraidNoFrameskip_v4_vanilla/policy.pth"
)

for i in $(seq 0 ${#TASK[@]});
do
  echo ${TASK[$i]} ${resume_path[$i]}
  /root/anaconda3/envs/rtdpd/bin/python -u \
  /root/RL_IoT_distillation/vanila/distillDQN.py \
  --task ${TASK[$i]}"NoFrameskip-v4" \
  --resume-path ${resume_path[$i]}""
done
echo "Experiments end"


#echo ${TASK[0]}
#for element in ${TASK[*]}
#do
#  echo $element"env"
#  #/home/zhoufu/anaconda3/envs/pd3.7.5_zf/bin/python -u \
#  /root/anaconda3/envs/rtdpd/bin/python -u \
#  /root/RL_IoT_distillation/vanila/distillDQN.py
#  --task $element"NoFrameskip-v4"\
#  #--test-num 30 \
#  #--epoch 130 \
#  #--logdir $LOGDIR
#  --resume-path "/root/RL_IoT_distillation/vanila/log/vanilla/"$element"NoFrameskip-v4/dqn/seed_0_1206_063927-PongNoFrameskip_v4_vanilla/policy.pth"
#done
#echo "Experiments ended"