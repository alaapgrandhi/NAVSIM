#!/usr/bin/zsh
# AD-side launcher for the implied-velocity sweep (mirrors dump_e2e.sh / ltf_e2e.sh).
# Pointed at by `ltf_path` in HUGSIM/configs/sim/nuscenes_base_implied_velocity.yaml.

NAVSIM_PATH=/home/mila/l/luke.rowe/gigapixel-dev/HUGSIM_ltf
cd ${NAVSIM_PATH}
echo ${PWD}
module load miniconda/3
conda activate /network/scratch/l/luke.rowe/conda-envs/hugsim_ltf

if [ -z "$4" ]; then
    CUDA_VISIBLE_DEVICES=${1} python compute_implied_velocity_hugsim.py output=$2 agent.checkpoint_path=$3
else
    IMAGE_SIZE=${4}
    CUDA_VISIBLE_DEVICES=${1} python compute_implied_velocity_hugsim.py output=$2 agent.checkpoint_path=$3 agent.config.image_size="[$IMAGE_SIZE]"
fi
cd -
