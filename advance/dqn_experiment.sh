#!/bin/bash

LOGDIR="log/batch_exs"
TASK=("Pong" "Breakout" "Enduro" "Qbert" "MsPacman" "Seaquest" "SpaceInvaders" "BeamRider" "Freeway" "Riverraid")

echo ${TASK[0]}

for element in ${TASK[*]}
do
  echo $element"env"
  /home/zhoufu/.conda/envs/pd3.7.5_zf/bin/python \
  -u /home/zhoufu/drl_iot/RL_IoT_distillation/advance/atari_dqn.py \
  --task $element"NoFrameskip-v4"\
  --test-num 100
  --epoch 130
done
echo "Experiments ended"

#for seed in $(seq 0 9)
#do
#    python mujoco_${ALGO}.py --task $TASK --epoch 200 --seed $seed --logdir $LOGDIR > ${TASK}_`date '+%m-%d-%H-%M-%S'`_seed_$seed.txt 2>&1 &
#done
#echo "Experiments ended."