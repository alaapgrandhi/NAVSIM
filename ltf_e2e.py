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
import torch

CONFIG_PATH = "navsim/planning/script/config/HUGSIM"
CONFIG_NAME = "drivor"

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    return parser.parse_args()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    if "latent" in cfg.agent.config:
        cfg.agent.config.latent = True
    cfg.agent.checkpoint_path = "./ckpts/drivor_Nav2_10epochs.pth"
    cfg.agent.scheduler_args.num_epochs = 10
    cfg.agent.batch_size = 64
    print(cfg)
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    # region agent log
    # try:
    #     import json, time  # local imports for debug logging
    #     log_path = "/home/mila/g/grandhia/NAVSIM/.cursor/debug-4a084f.log"
    #     payload = {
    #         "sessionId": "4a084f",
    #         "runId": "post-fix",
    #         "hypothesisId": "H1_nuplan_import",
    #         "location": "ltf_e2e.py:26",
    #         "message": "Agent initialized",
    #         "data": {"agent_class": type(agent).__name__},
    #         "timestamp": int(time.time() * 1000),
    #     }
    #     with open(log_path, "a") as f:
    #         f.write(json.dumps(payload) + "\n")
    # except Exception:
    #     pass
    # endregion

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
            raw_data = pipe.read()
            raw_data = pickle.loads(raw_data)
        print('received')
        
        if raw_data == 'Done':
            to_video(output_folder)
            exit(0)
        
        cam_params = raw_data[1]['cam_params']
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
            
            # imu to lidar
            imu_way_points = traj.poses[:, :3]
            way_points = imu_way_points[:, [1, 0, 2]]
            way_points[:, 0] *= -1
            
            save_fn = os.path.join(output_folder, f'{str(cnt).zfill(4)}')
            save_frame(raw_images, way_points, cam_params, save_fn)
            with open(plan_pipe, "wb") as pipe:
                pipe.write(pickle.dumps(way_points[:, :2]))
            print('sent')
        else:
            with open(plan_pipe, "wb") as pipe:
                pipe.write(pickle.dumps(None))
            print('Waiting for visualize tasks...')
            to_video(output_folder)
            exit(0)
        
        cnt += 1 

if __name__ == '__main__':
    # args = get_opts()
    main()