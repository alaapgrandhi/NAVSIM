#!/usr/bin/zsh
# AD-side launcher for the HUGSIM DrivoR feature dump (mirrors ltf_e2e.sh).
# Pointed at by `ltf_path` in HUGSIM/configs/sim/kitti360_base_dump.yaml, so
# closed_loop.py launches this instead of ltf_e2e.sh. Runs in the hugsim_ltf env.

NAVSIM_PATH=/home/mila/l/luke.rowe/gigapixel-dev/HUGSIM_ltf
cd ${NAVSIM_PATH}
echo ${PWD}
module load miniconda/3
conda activate /network/scratch/l/luke.rowe/conda-envs/hugsim_ltf

if [ -z "$4" ]; then
    CUDA_VISIBLE_DEVICES=${1} python dump_drivor_features_hugsim.py output=$2 agent.checkpoint_path=$3
else
    IMAGE_SIZE=${4}
    CUDA_VISIBLE_DEVICES=${1} python dump_drivor_features_hugsim.py output=$2 agent.checkpoint_path=$3 agent.config.image_size="[$IMAGE_SIZE]"
fi
cd -
