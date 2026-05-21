"""Dump DrivoR input-feature statistics + camera images during HUGSIM eval.

AD-side diagnostic. Drop-in for ltf_e2e.py: it talks to closed_loop.py over the
same obs/plan pipes and runs the real agent so the closed-loop sim proceeds
normally -- but for the first `DRIVOR_DUMP_N_FRAMES` frames of the scenario it
also logs (console + `<output>/feat_dump/feature_stats.log`):

  - ego_status   : the 25-dim ego vector, component-labelled
  - image        : per-camera (front/left/right/rear) stats of features["image"]
                   -- normalized mean/std/min/max + per-channel mean (catches
                   RGB/BGR swaps, wrong scale, all-black cameras, etc.)
  - trajectory   : the predicted poses from agent.compute_trajectory

and saves `<output>/feat_dump/frame_<NNN>.png` -- the 4 camera images the model
is fed (front/left/right/rear), denormalised and labelled.

Launched per-scenario by closed_loop.py via dump_e2e.sh (the `ltf_path` in
configs/sim/kitti360_base_dump.yaml). Runs in the hugsim_ltf conda env.
Env vars: DRIVOR_DUMP_N_FRAMES (default 5); the usual DRIVOR_PAD_LW /
DRIVOR_REWARD_COND / DRIVOR_REAR_AXLE_SHIFT / DRIVOR_ORIGINAL_CAMERA_ORDER
still apply (see ltf_e2e.py).
"""
import logging
import os
import pickle
from typing import List

import cv2
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivoR.drivor_features import predictions_center_to_rear_axle
from hugsim.dataparser import parse_raw

CONFIG_PATH = "navsim/planning/script/config/HUGSIM"
CONFIG_NAME = "drivor"

logger = logging.getLogger("dump_drivor_features_hugsim")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# features["image"] camera order is set by DrivoRFeatureBuilder._get_camera_feature and
# depends on cfg.agent.config.use_original_camera_order (env: DRIVOR_ORIGINAL_CAMERA_ORDER):
#   False (gigapixel):       [cam_f0, cam_l0, cam_r0, cam_b0, cam_l1, cam_l2, cam_r1, cam_r2]
#   True  (public DrivoR):   [cam_f0, cam_b0, cam_l0, cam_l1, cam_l2, cam_r0, cam_r1, cam_r2]
CAM_NAMES_GIGAPIXEL = ["front", "left", "right", "rear", "left1", "left2", "right1", "right2"]
CAM_NAMES_ORIGINAL = ["front", "rear", "left", "left1", "left2", "right", "right1", "right2"]

EGO_LABELS = [
    "pose_x", "pose_y", "pose_heading",
    "vel_x", "vel_y",
    "acc_x", "acc_y",
    "cmd_0", "cmd_1", "cmd_2", "cmd_3",
    "length", "width",
    "rc_collision", "rc_offroad", "rc_comfort", "rc_lane_align",
    "rc_lane_center", "rc_velocity", "rc_traffic_light", "rc_timestep",
    "rc_reverse", "rc_goal_radius", "rc_overspeed", "rc_goal_speed_tol",
]


def _envflag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")


def _envint(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else int(v)


def _denormalize_to_uint8(image_chw_normalized: torch.Tensor) -> np.ndarray:
    """Invert DrivoRFeatureBuilder's ImageNet normalize.
    (num_cams, 3, H, W) float -> (num_cams, H, W, 3) uint8 RGB."""
    arr = image_chw_normalized.detach().cpu().numpy()
    arr = arr * IMAGENET_STD[None, :, None, None] + IMAGENET_MEAN[None, :, None, None]
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return np.transpose(arr, (0, 2, 3, 1))


def save_camera_grid(image_uint8: np.ndarray, save_path: str, header: str = "",
                     cam_names: List[str] = CAM_NAMES_GIGAPIXEL) -> None:
    """image_uint8: (num_cams, H, W, 3) RGB uint8 -> labelled hconcat PNG."""
    panels = []
    for i, img in enumerate(image_uint8):
        bgr = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        bgr = cv2.copyMakeBorder(bgr, 26, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        name = cam_names[i] if i < len(cam_names) else f"cam{i}"
        cv2.putText(bgr, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(bgr)
    grid = np.concatenate(panels, axis=1)
    if header:
        grid = cv2.copyMakeBorder(grid, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(grid, header, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 220, 255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, grid)


def log_feature_stats(tag: str, features: dict,
                      cam_names: List[str] = CAM_NAMES_GIGAPIXEL) -> None:
    """Log ego_status (labelled) + per-camera image tensor stats."""
    ego = features["ego_status"]
    ego_last = ego[-1] if ego.dim() == 2 else ego
    ego_np = ego_last.detach().cpu().numpy()
    logger.info("[%s] ego_status (dim=%d):", tag, ego_np.shape[-1])
    for i, v in enumerate(ego_np):
        lbl = EGO_LABELS[i] if i < len(EGO_LABELS) else f"ego_{i}"
        logger.info("        %-22s % .5f", lbl, float(v))

    img = features["image"]  # (num_cams, 3, H, W)
    logger.info("[%s] image  shape=%s dtype=%s", tag, tuple(img.shape), img.dtype)
    for i in range(img.shape[0]):
        cam = cam_names[i] if i < len(cam_names) else f"cam{i}"
        c = img[i]
        logger.info(
            "        %-7s norm[mean=% .4f std=%.4f min=% .4f max=% .4f]  "
            "per-chan mean=[R % .3f G % .3f B % .3f]",
            cam, c.mean().item(), c.std().item(), c.min().item(), c.max().item(),
            c[0].mean().item(), c[1].mean().item(), c[2].mean().item(),
        )


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    # Same per-run compat-flag overrides as ltf_e2e.py.
    cfg.agent.config.pad_ego_length_width = _envflag(
        "DRIVOR_PAD_LW", cfg.agent.config.pad_ego_length_width)
    cfg.agent.config.pad_reward_conditioning = _envflag(
        "DRIVOR_REWARD_COND", cfg.agent.config.pad_reward_conditioning)
    cfg.agent.config.shift_predictions_to_rear_axle = _envflag(
        "DRIVOR_REAR_AXLE_SHIFT", cfg.agent.config.shift_predictions_to_rear_axle)
    cfg.agent.config.use_original_camera_order = _envflag(
        "DRIVOR_ORIGINAL_CAMERA_ORDER",
        cfg.agent.config.get("use_original_camera_order", False))
    # proposal_num sizes the model's init_feature embedding -> must match the
    # checkpoint being loaded. See ltf_e2e.py for details.
    cfg.agent.config.proposal_num = _envint(
        "DRIVOR_PROPOSAL_NUM", cfg.agent.config.proposal_num)
    cfg.agent.scheduler_args.num_epochs = 10
    cfg.agent.batch_size = 64

    n_dump = int(os.getenv("DRIVOR_DUMP_N_FRAMES", "-1"))  # -1 = all frames
    dump_dir = os.path.join(cfg.output, "feat_dump")
    os.makedirs(dump_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(os.path.join(dump_dir, "feature_stats.log"), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    n_dump_str = "all" if n_dump < 0 else str(n_dump)
    logger.info("=== HUGSIM DrivoR feature dump ===  output=%s  n_dump_frames=%s  (DRIVOR_DUMP_N_FRAMES=%s)",
                cfg.output, n_dump_str, os.getenv("DRIVOR_DUMP_N_FRAMES", "unset"))
    logger.info("compat flags: pad_lw=%s reward_cond=%s rear_axle_shift=%s "
                "use_original_camera_order=%s",
                cfg.agent.config.pad_ego_length_width,
                cfg.agent.config.pad_reward_conditioning,
                cfg.agent.config.shift_predictions_to_rear_axle,
                cfg.agent.config.use_original_camera_order)
    logger.info("checkpoint=%s", cfg.agent.checkpoint_path)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    feature_builder = agent.get_feature_builders()[0]
    logger.info("agent=%s  feature_builder=%s", type(agent).__name__, type(feature_builder).__name__)

    cam_names = (CAM_NAMES_ORIGINAL if cfg.agent.config.use_original_camera_order
                 else CAM_NAMES_GIGAPIXEL)
    logger.info("dump cam-name labels: %s", cam_names)

    obs_pipe = os.path.join(cfg.output, "obs_pipe")
    plan_pipe = os.path.join(cfg.output, "plan_pipe")
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print("Ready for recieving")

    cnt = 0
    while True:
        with open(obs_pipe, "rb") as pipe:
            raw_data = pickle.loads(pipe.read())
        if raw_data == "Done":
            logger.info("=== done. %d frames; images + feature_stats.log in %s ===", cnt, dump_dir)
            break

        data = parse_raw(raw_data)
        data.pop("raw_imgs", None)
        agent_input = data["input"]

        do_dump = (n_dump < 0) or (cnt < n_dump)

        # Feature dump -- exactly what the model is fed.
        if do_dump:
            logger.info("-------- frame %d --------", cnt)
            try:
                features = feature_builder.compute_features(agent_input)
                log_feature_stats("HUGSIM", features, cam_names=cam_names)
                image_uint8 = _denormalize_to_uint8(features["image"])
                save_camera_grid(image_uint8, os.path.join(dump_dir, f"frame_{cnt:03d}.png"),
                                 header=f"HUGSIM  frame {cnt}", cam_names=cam_names)
            except Exception as e:
                logger.warning("frame %d feature dump failed: %s", cnt, e)

        # Run the real agent so the closed-loop sim proceeds (mirrors ltf_e2e.py).
        try:
            with torch.no_grad():
                traj = agent.compute_trajectory(agent_input)
        except RuntimeError as e:
            logger.warning("compute_trajectory failed: %s", e)
            traj = None

        if traj is not None:
            poses = np.asarray(traj.poses)
            if do_dump:
                logger.info("[HUGSIM] predicted trajectory (%s):", poses.shape)
                for j, p in enumerate(poses):
                    logger.info("        wp%-2d  dx=% .3f  dy=% .3f  dh=% .4f", j, *p[:3])
            if cfg.agent.config.get("shift_predictions_to_rear_axle", False):
                poses = predictions_center_to_rear_axle(poses)
            # imu -> lidar (same as ltf_e2e.py)
            way_points = poses[:, :3][:, [1, 0, 2]]
            way_points[:, 0] *= -1
            with open(plan_pipe, "wb") as pipe:
                pipe.write(pickle.dumps(way_points[:, :2]))
        else:
            with open(plan_pipe, "wb") as pipe:
                pipe.write(pickle.dumps(None))
            logger.info("=== agent returned None; stopping. dumps in %s ===", dump_dir)
            break

        cnt += 1


if __name__ == "__main__":
    main()
