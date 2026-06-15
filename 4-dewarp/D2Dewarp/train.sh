#!/usr/bin/env bash

if [ $# -lt 1 ]
then
    echo "Usage: bash $0 OUTPUT_DIR"
    exit
fi

OUTPUT_DIR=$1

exp_name=YOUR_EXP_NAME
seed=24610
input_size=448

CUDA_VISIBLE_DEVICES=0,1 nohup python train.py \
  --exp_name ${exp_name}  \
  --input_size ${input_size} \
  --in_chans 4 \
  --hv_out_chans 1 \
  --d_model 448 \
  --epochs 80 \
  --data_path /DATA/PATH \
  --seed ${seed} \
  --save_interval 2000 \
  --batch_size 28 \
  --show_iter 20 \
  --lr 0.0001 \
  --min_lr 1e-7 \
  --warmup_steps 10000 \
  --weight_decay 0.01 \
  --save_path ${OUTPUT_DIR} > ${OUTPUT_DIR}/${exp_name}_in${input_size}-Seed${seed}.log 2>&1 &

# bash train_HV.sh OUTPUT_DIR