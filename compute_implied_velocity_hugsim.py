"""Compute the model's frame-0 trajectory in HUGSIM and report wp0's implied
velocity. AD-side worker; drop-in for ltf_e2e.py with the same pipe protocol.

For each scenario, this:
  - reads the first obs from obs_pipe,
  - runs the real agent once,
  - appends a CSV row to ${DRIVOR_IMPLIED_VELO_CSV} with the wp0 prediction
    and the implied velocity (= sqrt(wp0_dx^2 + wp0_dy^2) / 0.5s, signed by
    the sign of wp0_dx so reverse predictions show up as negative),
  - sends None on plan_pipe to terminate the closed-loop sim, then exits.

Launched per-scenario by closed_loop.py via compute_implied_velocity_e2e.sh
(the `ltf_path` in configs/sim/nuscenes_base_implied_velocity.yaml). Runs in
the hugsim_ltf conda env.

Env vars:
  DRIVOR_IMPLIED_VELO_CSV   path to shared CSV (appended; required)
  DRIVOR_PAD_LW             default True
  DRIVOR_REWARD_COND        default True
  DRIVOR_REAR_AXLE_SHIFT    default True
"""
from __future__ import annotations

import logging
import os
import pickle

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

logger = logging.getLogger("compute_implied_velocity_hugsim")

# DrivoR / puffer convention: 8 waypoints at 0.5s spacing (2Hz, 4s horizon).
TRAJ_DT = 0.5

CSV_HEADER = (
    "scenario,vel_x,vel_y,acc_x,acc_y,cmd_0,cmd_1,cmd_2,cmd_3,"
    "wp0_dx,wp0_dy,wp0_dh,implied_velo_signed,implied_velo_mag,wp7_dx,wp7_dy"
)


def _envflag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.agent.config.pad_ego_length_width = _envflag(
        "DRIVOR_PAD_LW", cfg.agent.config.pad_ego_length_width)
    cfg.agent.config.pad_reward_conditioning = _envflag(
        "DRIVOR_REWARD_COND", cfg.agent.config.pad_reward_conditioning)
    cfg.agent.config.shift_predictions_to_rear_axle = _envflag(
        "DRIVOR_REAR_AXLE_SHIFT", cfg.agent.config.shift_predictions_to_rear_axle)
    cfg.agent.scheduler_args.num_epochs = 10
    cfg.agent.batch_size = 64

    csv_path = os.getenv("DRIVOR_IMPLIED_VELO_CSV")
    if not csv_path:
        raise RuntimeError("DRIVOR_IMPLIED_VELO_CSV env var must be set")
    scenario_name = os.path.basename(cfg.output.rstrip("/"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.setLevel(logging.INFO)
    logger.info("=== compute_implied_velocity_hugsim ===  scenario=%s  csv=%s",
                scenario_name, csv_path)

    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)

    obs_pipe = os.path.join(cfg.output, "obs_pipe")
    plan_pipe = os.path.join(cfg.output, "plan_pipe")
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print("Ready for recieving")

    # ---- Read exactly one frame, compute trajectory, dump CSV, send None. ----
    with open(obs_pipe, "rb") as pipe:
        raw_data = pickle.loads(pipe.read())
    if raw_data == "Done":
        logger.warning("got Done immediately for %s; nothing to record", scenario_name)
        return

    data = parse_raw(raw_data)
    data.pop("raw_imgs", None)
    agent_input = data["input"]
    ego = agent_input.ego_statuses[-1]
    vel = np.asarray(ego.ego_velocity).reshape(-1)
    acc = np.asarray(ego.ego_acceleration).reshape(-1)
    cmd = np.asarray(ego.driving_command).reshape(-1)

    try:
        with torch.no_grad():
            traj = agent.compute_trajectory(agent_input)
    except Exception as e:
        logger.warning("compute_trajectory failed for %s: %s", scenario_name, e)
        traj = None

    if traj is not None:
        poses = np.asarray(traj.poses)
        if cfg.agent.config.get("shift_predictions_to_rear_axle", False):
            poses = predictions_center_to_rear_axle(poses)
        wp0_dx, wp0_dy, wp0_dh = float(poses[0, 0]), float(poses[0, 1]), float(poses[0, 2])
        mag = float(np.hypot(wp0_dx, wp0_dy)) / TRAJ_DT
        # Sign by wp0_dx so reverse predictions read negative.
        signed = (mag if wp0_dx >= 0 else -mag)
        wp7_dx, wp7_dy = float(poses[-1, 0]), float(poses[-1, 1])
    else:
        wp0_dx = wp0_dy = wp0_dh = wp7_dx = wp7_dy = float("nan")
        mag = signed = float("nan")

    row = [
        scenario_name,
        f"{float(vel[0]):.6f}", f"{float(vel[1]):.6f}",
        f"{float(acc[0]):.6f}", f"{float(acc[1]):.6f}",
        f"{float(cmd[0]):.0f}", f"{float(cmd[1]):.0f}",
        f"{float(cmd[2]):.0f}", f"{float(cmd[3]):.0f}",
        f"{wp0_dx:.6f}", f"{wp0_dy:.6f}", f"{wp0_dh:.6f}",
        f"{signed:.6f}", f"{mag:.6f}",
        f"{wp7_dx:.6f}", f"{wp7_dy:.6f}",
    ]
    # First-writer wins on the header (atomic-ish: only first non-existent open writes it).
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write(CSV_HEADER + "\n")
        f.write(",".join(row) + "\n")
    logger.info("[%s] vel_x=%.3f  wp0_dx=%.3f  wp0_dy=%.3f  implied_signed=%.3f",
                scenario_name, float(vel[0]), wp0_dx, wp0_dy, signed)

    # Send None to terminate closed_loop.py cleanly, then keep the FIFOs alive
    # long enough for closed_loop's final 'Done' write so it doesn't block on
    # opening obs_pipe for write with no reader.
    with open(plan_pipe, "wb") as pipe:
        pipe.write(pickle.dumps(None))
    try:
        with open(obs_pipe, "rb") as pipe:
            pipe.read()
    except Exception:
        pass


if __name__ == "__main__":
    main()
