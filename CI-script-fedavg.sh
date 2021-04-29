#!/bin/bash

set -ex

# code checking
# pyflakes .

# activate the fedml environment
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate fedml

wandb login 8b7e845b2460582a675ce06499dd7babfeb2dd59
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

# 1. MNIST standalone FedAvg
cd ./fedml_experiments/standalone/fedavg
sh run_fedavg_standalone_pytorch.sh 0 10 10 10 fed_cifar100 ./../../../data/fed_cifar100/datasets resnet18_gn hetero 4000 1 0.1 sgd 0 3