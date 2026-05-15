from navsim.agents.abstract_agent import AbstractAgent
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate
import numpy as np
import argparse
import os
import pickle
from hugsim.visualize import to_video, save_frame
from hugsim.dataparser import parse_raw
from navsim.agents.drivoR.drivor_features import predictions_center_to_rear_axle
import torch

CONFIG_PATH = "navsim/planning/script/config/HUGSIM"
CONFIG_NAME = "drivor"
# CHECKPOINT_PATH = "/network/scratch/g/grandhia/hugsim_data/drivor_Nav2_10epochs.pth"


def _envflag(name: str, default: bool) -> bool:
    """Per-run override of a boolean agent-config flag via environment variable.

    Env vars inherit down the whole launch chain (sbatch -> run_eval_*.sh ->
    closed_loop.py -> launch_ad.py -> ltf_e2e.sh -> here), so each SLURM job is
    isolated — unlike editing the shared drivoR.yaml. Unset env var keeps the
    yaml default; "1"/"true"/"yes" -> True, anything else -> False.
    """
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    if "latent" in cfg.agent.config:
        cfg.agent.config.latent = True
    # cfg.agent.checkpoint_path = CHECKPOINT_PATH
    print("CHECKPOINT_PATH: ", cfg.agent.checkpoint_path)
    cfg.agent.scheduler_args.num_epochs = 10
    cfg.agent.batch_size = 64
    # Per-run overrides of the gigapixel-checkpoint compat flags (default to the
    # yaml values, which are true). Set DRIVOR_* env vars in the launching sbatch
    # script to evaluate a checkpoint trained without these.
    cfg.agent.config.pad_ego_length_width = _envflag(
        "DRIVOR_PAD_LW", cfg.agent.config.pad_ego_length_width)
    cfg.agent.config.pad_reward_conditioning = _envflag(
        "DRIVOR_REWARD_COND", cfg.agent.config.pad_reward_conditioning)
    cfg.agent.config.shift_predictions_to_rear_axle = _envflag(
        "DRIVOR_REAR_AXLE_SHIFT", cfg.agent.config.shift_predictions_to_rear_axle)
    print(
        "DRIVOR compat flags: pad_ego_length_width=%s pad_reward_conditioning=%s "
        "shift_predictions_to_rear_axle=%s"
        % (
            cfg.agent.config.pad_ego_length_width,
            cfg.agent.config.pad_reward_conditioning,
            cfg.agent.config.shift_predictions_to_rear_axle,
        )
    )
    print(cfg)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    os.makedirs(cfg.output, exist_ok=True)
    obs_pipe = os.path.join(cfg.output, 'obs_pipe')
    plan_pipe = os.path.join(cfg.output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for recieving')
    cnt = 0
    output_folder = os.path.join(cfg.output, 'ltf')
    os.makedirs(output_folder, exist_ok=True)
    
    while True:
        with open(obs_pipe, "rb") as pipe:
            raw_bytes = pipe.read()
        raw_data = pickle.loads(raw_bytes)
        print('received')
        
        if raw_data == 'Done':
            to_video(output_folder)
            exit(0)
        
        try:
            cam_params = raw_data[1]['cam_params']
        except Exception:
            cam_params = None

        data = parse_raw(raw_data)
        raw_images = data['raw_imgs']
        del data['raw_imgs']

        try:
            with torch.no_grad():
                traj = agent.compute_trajectory(data['input'])
        except RuntimeError as e:
            traj = None
            print(e)
        
        if traj is not None:
            poses = np.asarray(traj.poses)
            # Gigapixel-distilled checkpoints predict bbox-center deltas in the ego
            # frame; shift them to rear-axle deltas before the imu->lidar axis swap.
            if cfg.agent.config.get("shift_predictions_to_rear_axle", False):
                poses = predictions_center_to_rear_axle(poses)
            # imu to lidar
            imu_way_points = poses[:, :3]
            way_points = imu_way_points[:, [1, 0, 2]]
            way_points[:, 0] *= -1
            
            save_fn = os.path.join(output_folder, f'{str(cnt).zfill(4)}')
            save_frame(raw_images, way_points, cam_params, save_fn)
            with open(plan_pipe, "wb") as pipe:
                payload = way_points[:, :2]
                payload_bytes = pickle.dumps(payload)
                pipe.write(payload_bytes)
            print('sent')
        else:
            with open(plan_pipe, "wb") as pipe:
                payload_bytes = pickle.dumps(None)
                pipe.write(payload_bytes)
            print('Waiting for visualize tasks...')
            to_video(output_folder)
            exit(0)
        
        cnt += 1 

if __name__ == '__main__':
    # args = get_opts()
    main()