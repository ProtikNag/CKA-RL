"""Final-policy cross-task retention evaluation (Table-3 style row).

Loads the final GLOBAL agent and, for each mode in the sequence, runs
--eval-episodes GREEDY episodes using that mode's head. Reports per-mode mean
return and success (mean return >= the per-mode hardcoded threshold from
process_results.py), plus the average. Writes table3_final_policy.json.
"""
import os
import json
import argparse

import numpy as np
import torch
import gymnasium as gym
from loguru import logger
from tabulate import tabulate

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from models.ours import OursAgent


def make_env(env_id, idx, capture_video, run_name, mode=None):
    def thunk():
        if mode is None:
            env = gym.make(env_id)
        else:
            env = gym.make(env_id, mode=mode)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk

# thresholds copied from process_results.py
SPACE_INVADERS_SCORES = [
    279.848, 326.349, 323.880,
    274.811, 369.782, 243.812,
    332.885, 333.781, 462.198, 383.627,
]
FREEWAY_SCORES = [
    21.1455, 19.177, 10.5049,
    21.7043, 23.5458, 11.9761,
    11.6078, 17.7292,
]

NUM_ENVS = 8


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="ALE/SpaceInvaders-v5")
    p.add_argument("--modes", type=str, default="0,1,2")
    p.add_argument("--tag", type=str, default="smoke")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--cuda", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
                   default=True)
    return p.parse_args()


def build_envs(env_id, mode, seed, run_name):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, False, run_name, mode=mode) for i in range(NUM_ENVS)]
    )
    return envs


def greedy_eval(agent, env_id, mode, task_idx, seed, n_episodes, device):
    envs = build_envs(env_id, mode, seed, f"finaleval_{mode}")
    returns = []
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    with torch.no_grad():
        while len(returns) < n_episodes:
            action = agent.greedy_action(next_obs / 255.0, task_idx)
            next_obs, _, term, trunc, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs).to(device)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        returns.append(float(np.asarray(info["episode"]["r"]).item()))
                        if len(returns) >= n_episodes:
                            break
    envs.close()
    return returns[:n_episodes]


def main():
    args = parse_args()
    modes = [int(m) for m in args.modes.split(",")]
    env_name = args.env_id.split("/")[1].split("-")[0]
    thresholds = SPACE_INVADERS_SCORES if env_name == "SpaceInvaders" else FREEWAY_SCORES

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")

    probe = build_envs(args.env_id, modes[0], args.seed, "probe")
    final_dir = f"./agents/{env_name}/{args.tag}/final_global"
    agent = OursAgent.load(final_dir, probe, map_location=device).to(device)
    agent.eval()
    probe.close()
    logger.info(f"loaded final global from {final_dir} with {len(agent.actors)} heads")

    per_mode = {}
    rows = []
    successes = []
    means = []
    for k, mode in enumerate(modes):
        rets = greedy_eval(agent, args.env_id, mode, k, args.seed,
                           args.eval_episodes, device)
        mean_ret = float(np.mean(rets))
        thr = float(thresholds[mode])
        success = bool(mean_ret >= thr)
        per_mode[str(mode)] = {
            "task_idx": k,
            "mean_return": mean_ret,
            "std_return": float(np.std(rets)),
            "threshold": thr,
            "success": success,
            "n_episodes": len(rets),
        }
        means.append(mean_ret)
        successes.append(1.0 if success else 0.0)
        rows.append([mode, k, round(mean_ret, 3), round(thr, 3), success])

    avg_ret = float(np.mean(means))
    success_rate = float(np.mean(successes))
    rows.append(["Avg.", "", round(avg_ret, 3), "", round(success_rate, 3)])

    print("\n----- FINAL-POLICY CROSS-TASK RETENTION (Table 3 style) -----\n")
    print(tabulate(rows,
                   headers=["Mode", "TaskIdx", "MeanReturn", "Threshold", "Success"],
                   tablefmt="rounded_outline"))

    out = {
        "env_id": args.env_id,
        "modes": modes,
        "eval_episodes": args.eval_episodes,
        "greedy": True,
        "per_mode": per_mode,
        "avg_mean_return": avg_ret,
        "success_rate": success_rate,
    }
    out_dir = f"./data/{env_name}/{args.tag}/Ours"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/table3_final_policy.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"wrote {out_dir}/table3_final_policy.json")


if __name__ == "__main__":
    main()
