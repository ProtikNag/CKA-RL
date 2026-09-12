"""Final-policy cross-task retention evaluation (Table-3 style, Meta-World SAC).

Meta-World/SAC analogue of experiments/atari/eval_final_policy.py. Loads the
final GLOBAL OursAgent and, for each task in the sequence, runs --eval-episodes
DETERMINISTIC (mean) episodes using that task's head. Reports per-task mean
success (and mean discounted MC return), plus the average over tasks. Writes
table3_final_policy.json.

Deterministic (argmax-equivalent: tanh(mean)) evaluation, matching the reported
greedy protocol. NOT run here (SLURM only); provided for post-run evaluation.
"""
import os
import json
import argparse

import numpy as np
import torch
import gymnasium as gym
from loguru import logger
from tabulate import tabulate

from models.ours import OursAgent
from tasks import get_task, tasks as CW20_TASKS

GAMMA = 0.99
LOG_STD_MAX = 2
LOG_STD_MIN = -20


def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def build_env(task_id):
    envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    envs.single_observation_space.dtype = np.float32
    return envs


class ActorHelper:
    """Deterministic (mean) action helper over an OursAgent head."""

    def __init__(self, agent, action_low, action_high, device):
        self.agent = agent
        self.action_scale = torch.tensor((action_high - action_low) / 2.0,
                                         dtype=torch.float32, device=device)
        self.action_bias = torch.tensor((action_high + action_low) / 2.0,
                                        dtype=torch.float32, device=device)

    def deterministic_action(self, x, task_idx):
        mean, _ = self.agent(x, task_idx=task_idx)
        return torch.tanh(mean) * self.action_scale + self.action_bias


@torch.no_grad()
def eval_task(agent, task_id, task_idx, seed, n_episodes, device):
    envs = build_env(task_id)
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high
    actor = ActorHelper(agent, action_low, action_high, device)
    obs, _ = envs.reset(seed=seed)
    disc_returns, successes = [], []
    disc, gpow = 0.0, 1.0
    while len(disc_returns) < n_episodes:
        obs_t = torch.Tensor(obs).to(device)
        action = actor.deterministic_action(obs_t, task_idx)
        obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
        disc += gpow * float(np.asarray(reward).reshape(-1)[0])
        gpow *= GAMMA
        done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
        if done:
            succ = 0.0
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "success" in info:
                        succ = float(info["success"])
                        break
            disc_returns.append(disc)
            successes.append(succ)
            disc, gpow = 0.0, 1.0
    envs.close()
    return (float(np.mean(successes[:n_episodes])),
            float(np.mean(disc_returns[:n_episodes])))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated CW20 task indices; empty -> 0..num_tasks-1")
    p.add_argument("--num-tasks", type=int, default=20)
    p.add_argument("--tag", type=str, default="smoke")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--cuda", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
                   default=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.tasks.strip():
        task_ids = [int(t) for t in args.tasks.split(",")]
    else:
        task_ids = list(range(args.num_tasks))

    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    final_dir = f"./agents/{args.tag}/final_global"
    agent = OursAgent.load(final_dir, map_location=device).to(device)
    agent.eval()
    logger.info(f"loaded final global from {final_dir} with {agent.num_tasks} heads")

    per_task, rows, successes, means = {}, [], [], []
    for k, task_id in enumerate(task_ids):
        succ, mean_ret = eval_task(agent, task_id, k, args.seed,
                                   args.eval_episodes, device)
        per_task[str(task_id)] = {
            "task_idx": k,
            "name": CW20_TASKS[task_id],
            "mean_success": succ,
            "mean_return": mean_ret,
            "n_episodes": args.eval_episodes,
        }
        successes.append(succ)
        means.append(mean_ret)
        rows.append([task_id, CW20_TASKS[task_id], k, round(succ, 3), round(mean_ret, 3)])

    avg_succ = float(np.mean(successes))
    avg_ret = float(np.mean(means))
    rows.append(["Avg.", "", "", round(avg_succ, 3), round(avg_ret, 3)])

    print("\n----- FINAL-POLICY CROSS-TASK RETENTION (Table 3 style) -----\n")
    print(tabulate(rows,
                   headers=["TaskID", "Name", "Idx", "Success", "MC-Return"],
                   tablefmt="rounded_outline"))

    out = {
        "task_ids": task_ids,
        "tasks": [CW20_TASKS[t] for t in task_ids],
        "eval_episodes": args.eval_episodes,
        "deterministic": True,
        "per_task": per_task,
        "avg_success": avg_succ,
        "avg_mean_return": avg_ret,
    }
    out_dir = f"./data/{args.tag}/Ours"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/table3_final_policy.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"wrote {out_dir}/table3_final_policy.json")


if __name__ == "__main__":
    main()
