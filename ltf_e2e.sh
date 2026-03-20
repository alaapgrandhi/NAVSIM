#!/usr/bin/zsh

# echo "HERE"
NAVSIM_PATH=/home/mila/g/grandhia/NAVSIM
cd ${NAVSIM_PATH}
echo ${PWD}
module load miniconda/3
conda activate hugsim_ltf
# echo "CUDA_VISIBLE_DEVICES=${1}"
if [ -z "$4" ]; then
    CUDA_VISIBLE_DEVICES=${1} python ltf_e2e.py output=$2 agent.checkpoint_path=$3
else
    IMAGE_SIZE=${4}
    CUDA_VISIBLE_DEVICES=${1} python ltf_e2e.py output=$2 agent.checkpoint_path=$3 agent.config.image_size="[$IMAGE_SIZE]"
fi
cd -
