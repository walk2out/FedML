#!/usr/bin/env bash

#!/bin/bash

set -ex

# code checking
# pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login b44e448d3bcd54bdb2dd9ed906341d5f55e93e55
wandb off

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    echo "$expected != $actual"
    return 1
  fi
}

round() {
  printf "%.${2}f" "${1}"
}

GPU=0

CLIENT_NUM=10

WORKER_NUM=10

BATCH_SIZE=10

DATASET=fed_cifar100

DATA_PATH=./../../../data/fed_cifar100/datasets

MODEL=resnet18_gn

DISTRIBUTION=hetero

ROUND=4000

EPOCH=1

LR=0.1

OPT=sgd

CI=0

BW=16

echo ${CLIENT_NUM}
echo ${WORKER_NUM}
echo ${BATCH_SIZE}
echo ${ROUND}
echo ${EPOCH}
echo ${LR}
echo ${OPT}
echo ${CI}
echo ${BW}

n=1
g=$(($n<8?$n:8))
srun --mpi=pmi2 -p ad_rd -n$n --gres=gpu:$g --ntasks-per-node=$g \
python3 ./main_fedavg.py \
--gpu $GPU \
--dataset $DATASET \
--data_dir $DATA_PATH \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--batch_size $BATCH_SIZE \
--client_optimizer $OPT \
--lr $LR \
--ci $CI \
--b_w $BW \
2>&1 | tee logs/0505_quant_lr_${LR}_client_num_${WORKER_NUM}_bw_${BW}.log
