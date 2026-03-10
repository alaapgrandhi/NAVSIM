#!/usr/bin/zsh

# echo "HERE"
NAVSIM_PATH=/home/mila/g/grandhia/NAVSIM
cd ${NAVSIM_PATH}
echo ${PWD}
module load miniconda/3
conda activate hugsim_ltf
# echo "CUDA_VISIBLE_DEVICES=${1}"
CUDA_VISIBLE_DEVICES=${1} python ltf_e2e.py output=$2
cd -
