"""Constrained min-max two-policy continual RL (PPO backend).

Single-process, full-sequence runner over a mode sequence for one ALE game.

Method summary (see task spec Part B):
  - Task 0: standard PPO -> this IS the initial global. Its greedy-100 score is
    mode[0]'s local reference.
  - Task k>=1:
      * LOCAL phase: clone global -> local, run UNCONSTRAINED pure PPO on mode k
        (encoder + head k). Freeze; greedy-100 score = local reference ref_k.
      * GLOBAL phase (consolidation): warm-start from global. Each iteration roll
        out EVERY seen mode i<=k with head i, compute per-task clipped-surrogate
        PPO loss. Actor coeffs: past i<k get omega_i = 1/k; current k gets
        mu*2*shortfall_k with shortfall_k = max(0, V_k^L - V_k^G) (MC on-policy
        stochastic returns). Normalize all actor coeffs by their sum. Critic +
        entropy are standard (not normalized, not constrained). Dual mu update
        (projected ascent) every constraint_every iters. Retention-gated early
        stop.

The returns.csv for mode k contains ONLY the LOCAL-phase (task0 for k=0)
episodic returns on mode k -- the pure current-task training curve, matching the
baselines' returns.csv for Table-1 plasticity/FWT comparability. The global
consolidation curve is written separately under consolidation/{mode}/returns.csv.
"""
import os
import json
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from loguru import logger

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from models.ours import OursAgent


def make_env(env_id, idx, capture_video, run_name, mode=None):
    # identical to run_ppo.make_env (kept local to avoid run_ppo's heavy imports)
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


# ---------------- Reference PPO hyperparameters (Table 6 / run_ppo.py) --------
NUM_ENVS = 8
NUM_STEPS = 128
NUM_MINIBATCHES = 4
UPDATE_EPOCHS = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.1
VF_COEF = 0.5
ENT_COEF = 0.01
LR = 2.5e-4
MAX_GRAD_NORM = 0.5
NORM_ADV = True
CLIP_VLOSS = True

BATCH_SIZE = NUM_ENVS * NUM_STEPS          # 1024
MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES  # 256

# dual / constraint defaults (overridable via CLI)
DUAL_LR = 0.3
MU_MAX = 5.0
EPS = 0.04
RETENTION_FRAC = 0.7
PATIENCE = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", type=str, default="ALE/SpaceInvaders-v5")
    p.add_argument("--modes", type=str, default="0,1,2",
                   help="comma-separated ALE mode integers = task sequence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", type=str, default="smoke")
    p.add_argument("--task1-steps", type=int, default=int(1e6))
    p.add_argument("--local-steps", type=int, default=int(1e6))
    p.add_argument("--global-steps", type=int, default=int(1e6),
                   help="DEPRECATED/UNUSED: global phase now uses --global-iters")
    p.add_argument("--global-iters", type=int, default=2000,
                   help="fixed # of global-phase iterations (each rolls out ALL "
                        "seen modes; total frames grow with #tasks)")
    p.add_argument("--min-iters", type=int, default=500,
                   help="floor before retention-gated early-stop may trigger")
    p.add_argument("--constraint-episodes", type=int, default=16,
                   help="# MC episodes per policy for shortfall estimate")
    p.add_argument("--constraint-every", type=int, default=20,
                   help="dual mu update cadence in global iterations")
    p.add_argument("--eps", type=float, default=EPS)
    p.add_argument("--dual-lr", type=float, default=DUAL_LR)
    p.add_argument("--mu-max", type=float, default=MU_MAX)
    p.add_argument("--retention-frac", type=float, default=RETENTION_FRAC)
    p.add_argument("--stop-eval-every", type=int, default=200,
                   help="retention-gated early-stop check cadence (global iters); "
                        "atari5 source uses 200")
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--eval-episodes", type=int, default=100,
                   help="# greedy episodes for the REPORTED local reference score")
    p.add_argument("--stop-eval-episodes", type=int, default=3,
                   help="# greedy episodes for each cheap retention-gate check "
                        "(atari5 source uses 3; NOT the reported 100)")
    p.add_argument("--log-every", type=int, default=5,
                   help="flush returns.csv + progress.jsonl + status.json every N iterations")
    p.add_argument("--ckpt-every", type=int, default=100,
                   help="within-task rolling checkpoint cadence in iterations "
                        "(0 = only at task boundaries)")
    p.add_argument("--resume", action="store_true",
                   help="resume from the last completed task boundary in this tag "
                        "(reads run_state.json + global_after_task{k}.)")
    p.add_argument("--cuda", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
                   default=True)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def build_envs(env_id, mode, seed, run_name):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, i, False, run_name, mode=mode) for i in range(NUM_ENVS)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    return envs


# ---------------- observability: crash-safe logging + checkpoints ------------
class Reporter:
    """Persists EVERYTHING important as the run proceeds, so progress can be
    inspected mid-run and nothing is lost if the process dies. Writes under
    data/{env}/{tag}/Ours/ :
      - progress.jsonl        : one row per logged iteration (append+fsync)
                                 {t_wall, task, mode, phase, iter, iters_total,
                                  global_step, sps, mu, shortfall, Vk_L, Vk_G,
                                  pg_loss, v_loss, entropy, mean_recent_return}
      - status.json           : single-object heartbeat (atomic replace) with
                                 the current task/phase/percent/eta + latest refs
      - retention_history.jsonl : per retention-gated check, per-mode greedy score
      - phase_summaries.jsonl : one row when each phase finishes
      - returns.csv (per mode): flushed incrementally, not only at the end
    Checkpoints under agents/{env}/{tag}/ (rolling within-task + at boundaries).
    """

    def __init__(self, env_name, tag, agents_root, modes, args):
        self.env_name = env_name
        self.tag = tag
        self.agents_root = agents_root
        self.modes = modes
        self.args = args
        self.run_dir = f"./data/{env_name}/{tag}/Ours"
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(agents_root, exist_ok=True)
        self.progress_path = os.path.join(self.run_dir, "progress.jsonl")
        self.retention_path = os.path.join(self.run_dir, "retention_history.jsonl")
        self.phase_path = os.path.join(self.run_dir, "phase_summaries.jsonl")
        self.status_path = os.path.join(self.run_dir, "status.json")
        self.start = time.time()

    def _append(self, path, record):
        record = {"t_wall": round(time.time() - self.start, 2), **record}
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def progress(self, **rec):
        self._append(self.progress_path, rec)

    def retention(self, **rec):
        self._append(self.retention_path, rec)

    def phase_summary(self, **rec):
        self._append(self.phase_path, rec)

    def status(self, **rec):
        rec = {"t_wall_sec": round(time.time() - self.start, 2), **rec}
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.status_path)  # atomic: readers never see a partial file

    def write_returns(self, mode, logs):
        write_returns(self.env_name, self.tag, mode, logs)

    def flush_all_returns(self, logs_by_idx):
        for i, mode in enumerate(self.modes):
            if logs_by_idx[i]["global_step"]:
                write_returns(self.env_name, self.tag, mode, logs_by_idx[i])

    def flush_all_consolidation(self, cons_logs_by_idx):
        for i, mode in enumerate(self.modes):
            if cons_logs_by_idx[i]["global_step"]:
                write_consolidation(self.env_name, self.tag, mode, cons_logs_by_idx[i])

    def checkpoint(self, agent, name):
        agent.save(f"{self.agents_root}/{name}")


def _mean_recent(logs, n=20):
    ys = logs.get("episodic_return", [])
    return float(np.mean(ys[-n:])) if ys else float("nan")


# ---------------- greedy / MC evaluation -------------------------------------
def greedy_eval(agent, env_id, mode, task_idx, seed, n_episodes, device):
    """Mean episodic return over n_episodes GREEDY (argmax) episodes.

    Uses a single env (RecordEpisodeStatistics reports true env returns,
    unaffected by ClipReward which only affects the reward signal, not the
    'episode' stat which records the *clipped* env reward here. We simply report
    what the wrapper gives -- consistent across all policies compared)."""
    envs = build_envs(env_id, mode, seed + 12345, f"eval_{mode}")
    returns = []
    next_obs, _ = envs.reset(seed=seed + 12345)
    next_obs = torch.Tensor(next_obs).to(device)
    agent.eval()
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
    agent.train()
    envs.close()
    return float(np.mean(returns[:n_episodes]))


def mc_stochastic_value(agent, env_id, mode, task_idx, seed, n_episodes, device):
    """Mean DISCOUNTED (gamma) stochastic return over n_episodes episodes.

    This is the Monte-Carlo on-policy *value* estimate V^pi used for the
    shortfall / constraint. Matches crl/ppo/evaluate.py mean_value: the
    discounted return is accumulated on the (already ClipRewardEnv-clipped)
    training reward scale, NOT the undiscounted game score. The per-env running
    discounted return is disc[e] += gpow[e]*reward[e]; gpow[e] *= GAMMA, reset
    when env e's episode ends."""
    envs = build_envs(env_id, mode, seed + 777, f"mc_{mode}")
    returns = []
    disc = np.zeros(NUM_ENVS, dtype=np.float64)   # running discounted return per env
    gpow = np.ones(NUM_ENVS, dtype=np.float64)    # gamma^t per env
    next_obs, _ = envs.reset(seed=seed + 777)
    next_obs = torch.Tensor(next_obs).to(device)
    agent.eval()
    with torch.no_grad():
        while len(returns) < n_episodes:
            action, _, _, _ = agent.get_action_and_value(next_obs / 255.0, task_idx)
            next_obs_np, reward, term, trunc, infos = envs.step(action.cpu().numpy())
            next_obs = torch.Tensor(next_obs_np).to(device)
            reward = np.asarray(reward, dtype=np.float64)
            disc += gpow * reward
            gpow *= GAMMA
            done = np.logical_or(np.asarray(term), np.asarray(trunc))
            for e in range(NUM_ENVS):
                if done[e]:
                    returns.append(float(disc[e]))
                    disc[e] = 0.0
                    gpow[e] = 1.0
    agent.train()
    envs.close()
    return float(np.mean(returns[:n_episodes]))


# ---------------- rollout + GAE ----------------------------------------------
def collect_rollout(agent, envs, task_idx, next_obs, next_done, device, logs, global_step):
    """Roll out NUM_STEPS with the given agent/head; return batch tensors + GAE.

    Appends episodic returns encountered to `logs` (acting policy's returns on
    this mode). Returns updated (next_obs, next_done, global_step)."""
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + obs_shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    for step in range(NUM_STEPS):
        global_step += NUM_ENVS
        obs[step] = next_obs
        dones[step] = next_done
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs / 255.0, task_idx)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done_np = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs = torch.Tensor(next_obs_np).to(device)
        next_done = torch.Tensor(next_done_np).to(device)

        if logs is not None and "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    logs["global_step"].append(global_step)
                    logs["episodic_return"].append(float(np.asarray(info["episode"]["r"]).item()))

    # GAE
    with torch.no_grad():
        next_value = agent.get_value(next_obs / 255.0, task_idx).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            )
        returns = advantages + values

    batch = dict(
        obs=obs.reshape((-1,) + obs_shape),
        logprobs=logprobs.reshape(-1),
        actions=actions.reshape((-1,) + envs.single_action_space.shape),
        advantages=advantages.reshape(-1),
        returns=returns.reshape(-1),
        values=values.reshape(-1),
    )
    return batch, next_obs, next_done, global_step


def ppo_losses_for_batch(agent, batch, task_idx, mb_inds, device):
    """Compute (pg_loss, v_loss, entropy_loss) for one minibatch of a task."""
    b_obs = batch["obs"]
    b_logprobs = batch["logprobs"]
    b_actions = batch["actions"]
    b_advantages = batch["advantages"]
    b_returns = batch["returns"]
    b_values = batch["values"]

    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
        b_obs[mb_inds] / 255.0, task_idx, b_actions.long()[mb_inds]
    )
    logratio = newlogprob - b_logprobs[mb_inds]
    ratio = logratio.exp()

    mb_advantages = b_advantages[mb_inds]
    if NORM_ADV:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    pg_loss1 = -mb_advantages * ratio
    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    newvalue = newvalue.view(-1)
    if CLIP_VLOSS:
        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
        v_clipped = b_values[mb_inds] + torch.clamp(
            newvalue - b_values[mb_inds], -CLIP_COEF, CLIP_COEF
        )
        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    entropy_loss = entropy.mean()
    return pg_loss, v_loss, entropy_loss


# ---------------- standard single-task PPO (task0 + local phase) -------------
def train_ppo_single_task(agent, env_id, mode, task_idx, seed, total_steps,
                          device, logs, log_global_step_offset,
                          reporter=None, phase="ppo", task_k=None, save_name=None):
    """Standard unconstrained PPO on one mode using head `task_idx`.

    Updates encoder + head task_idx (and its critic head). Returns the number of
    env steps actually consumed (approx multiple of BATCH_SIZE). Streams
    returns.csv / progress.jsonl / status.json + rolling checkpoints via
    `reporter` so the phase is inspectable mid-run and crash-resilient."""
    run_name = f"train_{mode}_{task_idx}"
    envs = build_envs(env_id, mode, seed, run_name)

    num_iterations = max(1, total_steps // BATCH_SIZE)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    log_every = getattr(reporter.args, "log_every", 5) if reporter else 10**9
    ckpt_every = getattr(reporter.args, "ckpt_every", 0) if reporter else 0

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    gstep = 0
    t0 = time.time()
    last_pg = last_v = last_ent = float("nan")

    for iteration in range(1, num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        optimizer.param_groups[0]["lr"] = frac * LR

        # rollout with local logging offset so returns.csv keeps a running step
        logs_local = {"global_step": [], "episodic_return": []}
        batch, next_obs, next_done, gstep = collect_rollout(
            agent, envs, task_idx, next_obs, next_done, device, logs_local, gstep
        )
        if logs is not None:
            for gs, r in zip(logs_local["global_step"], logs_local["episodic_return"]):
                logs["global_step"].append(log_global_step_offset + gs)
                logs["episodic_return"].append(r)

        b_inds = np.arange(BATCH_SIZE)
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb_inds = b_inds[start:start + MINIBATCH_SIZE]
                pg_loss, v_loss, entropy_loss = ppo_losses_for_batch(
                    agent, batch, task_idx, mb_inds, device
                )
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                last_pg, last_v, last_ent = (float(pg_loss.item()),
                                             float(v_loss.item()),
                                             float(entropy_loss.item()))

        # ---- stream progress / returns / checkpoint ----
        if reporter is not None and (iteration % log_every == 0 or iteration == num_iterations):
            sps = int(gstep / max(1e-9, time.time() - t0))
            reporter.write_returns(mode, logs)
            reporter.progress(task=task_k, mode=mode, phase=phase,
                              iter=iteration, iters_total=num_iterations,
                              global_step=gstep, sps=sps,
                              pg_loss=round(last_pg, 5), v_loss=round(last_v, 5),
                              entropy=round(last_ent, 4),
                              mean_recent_return=round(_mean_recent(logs), 3))
            frac_done = iteration / num_iterations
            reporter.status(task=task_k, mode=mode, phase=phase,
                            iter=iteration, iters_total=num_iterations,
                            percent=round(100 * frac_done, 1), sps=sps,
                            phase_elapsed_sec=round(time.time() - t0, 1),
                            phase_eta_sec=round((time.time() - t0) * (1 - frac_done) / max(1e-9, frac_done), 1),
                            mean_recent_return=round(_mean_recent(logs), 3))
        if reporter is not None and ckpt_every and save_name and iteration % ckpt_every == 0:
            reporter.checkpoint(agent, f"{save_name}_latest")

    envs.close()
    return gstep


# ---------------- global consolidation phase ---------------------------------
def train_global_phase(agent, local_agent, env_id, modes, seen_idx, k,
                       seed, total_steps, args, device, cons_logs_by_idx,
                       log_offset_by_idx, local_refs, ref_mode_by_idx,
                       reporter=None):
    """Consolidation. `seen_idx` = list of task indices 0..k. Current = k.

    Rolls out every seen mode each iteration, builds normalized actor coeffs,
    dual mu update, retention-gated early stop. `cons_logs_by_idx[i]` collects
    episodic returns for mode i during this CONSOLIDATION phase only (flushed to
    data/.../consolidation/{mode}.csv, NOT returns.csv).

    Two-timescale: V_k^G / shortfall_k are recomputed (and mu updated) only every
    `constraint_every` iterations; between refreshes the held shortfall_k is used
    so the 16-episode MC does not run every iteration.
    """
    run_names = {i: f"global_{modes[i]}_{i}" for i in seen_idx}
    envs_by_idx = {i: build_envs(env_id, modes[i], seed + 1000 + i, run_names[i])
                   for i in seen_idx}
    state_by_idx = {}
    gstep_by_idx = {i: 0 for i in seen_idx}
    for i in seen_idx:
        no, _ = envs_by_idx[i].reset(seed=seed + 1000 + i)
        state_by_idx[i] = (torch.Tensor(no).to(device), torch.zeros(NUM_ENVS).to(device))

    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # fixed global-phase iteration count (each iteration rolls out ALL seen modes,
    # so total frames grow with #tasks -- the disclosed asymmetry, by design)
    num_iterations = args.global_iters
    mu = 0.0  # reset each task's global phase
    consec_ok = 0
    total_consolidation_steps = 0

    # V_k^L is fixed (frozen local); computed once
    Vk_L = mc_stochastic_value(local_agent, env_id, modes[k], k, seed,
                               args.constraint_episodes, device)
    logger.info(f"[global k={k}] frozen local V_k^L (MC, {args.constraint_episodes} eps) = {Vk_L:.3f}")

    # Initial V_k^G + shortfall computed ONCE so coeff_k is defined for the early
    # iterations before the first constraint_every refresh.
    Vk_G = mc_stochastic_value(agent, env_id, modes[k], k, seed,
                               args.constraint_episodes, device)
    shortfall_k = max(0.0, Vk_L - Vk_G)
    logger.info(f"[global k={k}] initial V_k^G={Vk_G:.3f} shortfall={shortfall_k:.3f}")

    log_every = getattr(args, "log_every", 5)
    ckpt_every = getattr(args, "ckpt_every", 0)
    t0 = time.time()
    last = {"pg": float("nan"), "v": float("nan"), "ent": float("nan")}

    for iteration in range(1, num_iterations + 1):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        optimizer.param_groups[0]["lr"] = frac * LR

        # ---- roll out every seen mode with current global ----
        batches = {}
        for i in seen_idx:
            no, nd = state_by_idx[i]
            logs_local = {"global_step": [], "episodic_return": []}
            batches[i], no, nd, gstep_by_idx[i] = collect_rollout(
                agent, envs_by_idx[i], i, no, nd, device, logs_local, gstep_by_idx[i]
            )
            state_by_idx[i] = (no, nd)
            total_consolidation_steps += BATCH_SIZE
            # log returns for this mode's CONSOLIDATION csv (all seen modes get
            # logged during the global phase; kept separate from returns.csv so
            # returns.csv stays the pure local-phase Table-1 plasticity curve)
            dst = cons_logs_by_idx[i]
            for gs, r in zip(logs_local["global_step"], logs_local["episodic_return"]):
                dst["global_step"].append(gs)
                dst["episodic_return"].append(r)

        # ---- two-timescale refresh: recompute V_k^G / shortfall + update mu
        # ONLY every constraint_every iters; hold shortfall_k fixed between
        # refreshes so the 16-episode MC does not run every iteration ----
        if iteration % args.constraint_every == 0:
            Vk_G = mc_stochastic_value(agent, env_id, modes[k], k, seed,
                                       args.constraint_episodes, device)
            shortfall_k = max(0.0, Vk_L - Vk_G)
            mu = float(np.clip(mu + args.dual_lr * (shortfall_k ** 2 - args.eps),
                               0.0, args.mu_max))
            logger.info(f"[global k={k}] it={iteration} shortfall={shortfall_k:.3f} "
                        f"Vk_G={Vk_G:.3f} mu={mu:.4f}")

        # ---- actor coefficients (from currently-held shortfall_k + mu) ----
        coeffs = {}
        for i in seen_idx:
            if i < k:
                coeffs[i] = 1.0 / (k + 1)    # 1/(total tasks seen) = 1/|seen_idx|
            else:
                coeffs[i] = mu * 2.0 * shortfall_k
        Z = sum(coeffs.values())
        if Z <= 0:
            # degenerate (mu=0 and no past, or all-zero): fall back to uniform
            for i in seen_idx:
                coeffs[i] = 1.0 / len(seen_idx)
            Z = 1.0
        norm_coeffs = {i: coeffs[i] / Z for i in seen_idx}

        # ---- PPO update: normalized actor CL term + standard critic/entropy ----
        for epoch in range(UPDATE_EPOCHS):
            inds_by_idx = {i: np.random.permutation(BATCH_SIZE) for i in seen_idx}
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                total_actor = 0.0
                total_critic = 0.0
                total_entropy = 0.0
                for i in seen_idx:
                    mb_inds = inds_by_idx[i][start:start + MINIBATCH_SIZE]
                    pg_loss, v_loss, entropy_loss = ppo_losses_for_batch(
                        agent, batches[i], i, mb_inds, device
                    )
                    total_actor = total_actor + norm_coeffs[i] * pg_loss
                    total_critic = total_critic + v_loss
                    total_entropy = total_entropy + entropy_loss
                loss = total_actor + VF_COEF * total_critic - ENT_COEF * total_entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                last = {"pg": float(total_actor.item()),
                        "v": float(total_critic.item()),
                        "ent": float(total_entropy.item())}

        # ---- stream progress / returns / rolling checkpoint ----
        if reporter is not None and (iteration % log_every == 0 or iteration == num_iterations):
            done_steps = total_consolidation_steps
            sps = int(done_steps / max(1e-9, time.time() - t0))
            reporter.flush_all_consolidation(cons_logs_by_idx)
            reporter.progress(task=k, mode=modes[k], phase="global",
                              iter=iteration, iters_total=num_iterations,
                              consolidation_steps=done_steps, sps=sps, mu=round(mu, 5),
                              shortfall=round(shortfall_k, 4), Vk_L=round(Vk_L, 3),
                              Vk_G=round(Vk_G, 3),
                              actor_coeffs={int(modes[i]): round(norm_coeffs[i], 4) for i in seen_idx},
                              pg_loss=round(last["pg"], 5), v_loss=round(last["v"], 5),
                              entropy=round(last["ent"], 4))
            frac_done = iteration / num_iterations
            reporter.status(task=k, mode=modes[k], phase="global",
                            iter=iteration, iters_total=num_iterations,
                            percent=round(100 * frac_done, 1), sps=sps, mu=round(mu, 5),
                            shortfall=round(shortfall_k, 4), Vk_L=round(Vk_L, 3),
                            Vk_G=round(Vk_G, 3),
                            phase_elapsed_sec=round(time.time() - t0, 1),
                            phase_eta_sec=round((time.time() - t0) * (1 - frac_done) / max(1e-9, frac_done), 1),
                            seen_modes=[int(modes[i]) for i in seen_idx],
                            local_refs={int(modes[i]): round(local_refs[i], 3) for i in seen_idx})
        if reporter is not None and ckpt_every and iteration % ckpt_every == 0:
            reporter.checkpoint(agent, f"global_ckpt_task{k}_latest")

        # ---- retention-gated early stop (only after the min_iters floor) ----
        if iteration >= args.min_iters and iteration % args.stop_eval_every == 0:
            all_ok = True
            scores = {}
            for i in seen_idx:
                g = greedy_eval(agent, env_id, modes[i], i, seed,
                                args.stop_eval_episodes, device)
                scores[int(modes[i])] = round(g, 3)
                thr = args.retention_frac * local_refs[i]
                if g < thr:
                    all_ok = False
            consec_ok = consec_ok + 1 if all_ok else 0
            logger.info(f"[global k={k}] it={iteration} retention_all_ok={all_ok} "
                        f"consec={consec_ok}")
            if reporter is not None:
                reporter.retention(task=k, iter=iteration, all_ok=bool(all_ok),
                                   consec=consec_ok, greedy_scores=scores,
                                   thresholds={int(modes[i]): round(args.retention_frac * local_refs[i], 3)
                                               for i in seen_idx})
            if consec_ok >= args.patience:
                logger.info(f"[global k={k}] retention-gated early stop at it={iteration}")
                break

    for i in seen_idx:
        envs_by_idx[i].close()
    if reporter is not None:
        reporter.phase_summary(task=k, phase="global", iters_run=iteration,
                               consolidation_steps=total_consolidation_steps,
                               mu_final=round(mu, 5), Vk_L=round(Vk_L, 3),
                               Vk_G_last=round(Vk_G, 3), shortfall_last=round(shortfall_k, 4))
    return total_consolidation_steps


def _load_returns_csv(env_name, tag, mode):
    """Reload a previously-written returns.csv into a logs dict (for --resume),
    dropping the leading (0,0) sentinel row so re-flushing doesn't duplicate it."""
    path = f"./data/{env_name}/{tag}/Ours/{mode}/returns.csv"
    if not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    gs = df["global_step"].tolist()
    er = df["episodic_return"].tolist()
    if gs and gs[0] == 0 and er[0] == 0:
        gs, er = gs[1:], er[1:]
    return {"global_step": gs, "episodic_return": er}


def write_returns(env_name, tag, mode, logs):
    log_dir = f"./data/{env_name}/{tag}/Ours/{mode}"
    os.makedirs(log_dir, exist_ok=True)
    import pandas as pd
    # ensure a leading (0,0) row like run_ppo does
    gs = [0] + logs["global_step"]
    er = [0] + logs["episodic_return"]
    df = pd.DataFrame({"global_step": gs, "episodic_return": er})
    df = df.sort_values("global_step").reset_index(drop=True)
    # Make global_step strictly unique WITHOUT discarding any episodic return:
    # several envs can finish at the same global_step; instead of dropping all
    # but one (silent data loss), nudge collisions by +1,+2,... so the metric
    # pipeline's per-step merge/dedup keeps every acting-policy return.
    steps = df["global_step"].to_numpy().copy()
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            steps[i] = steps[i - 1] + 1
    df["global_step"] = steps
    df.to_csv(f"{log_dir}/returns.csv", index=False)
    logger.info(f"wrote {log_dir}/returns.csv ({len(df)} rows)")


def write_consolidation(env_name, tag, mode, logs):
    """Global-phase (consolidation) episodic-return curve for one mode, written
    under a separate consolidation/ subdir so returns.csv stays the pure
    local-phase Table-1 plasticity curve."""
    log_dir = f"./data/{env_name}/{tag}/Ours/consolidation/{mode}"
    os.makedirs(log_dir, exist_ok=True)
    import pandas as pd
    gs = [0] + logs["global_step"]
    er = [0] + logs["episodic_return"]
    df = pd.DataFrame({"global_step": gs, "episodic_return": er})
    df = df.sort_values("global_step").reset_index(drop=True)
    steps = df["global_step"].to_numpy().copy()
    for i in range(1, len(steps)):
        if steps[i] <= steps[i - 1]:
            steps[i] = steps[i - 1] + 1
    df["global_step"] = steps
    df.to_csv(f"{log_dir}/returns.csv", index=False)
    logger.info(f"wrote {log_dir}/returns.csv ({len(df)} rows)")


def main():
    args = parse_args()
    if not args.debug:
        import sys
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    modes = [int(m) for m in args.modes.split(",")]
    env_name = args.env_id.split("/")[1].split("-")[0]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    logger.info(f"device={device} modes={modes} env={args.env_id} tag={args.tag}")

    # build a probe env just for action space to construct the agent
    probe = build_envs(args.env_id, modes[0], args.seed, "probe")
    agent = OursAgent(probe, num_tasks=1).to(device)
    probe.close()

    local_refs = {}   # task_idx -> local reference greedy-100 score
    # per-mode LOCAL-phase logs (episodic returns of the acting policy on that
    # mode during its local / task0 phase). This IS returns.csv (Table-1 curve).
    logs_by_idx = {i: {"global_step": [], "episodic_return": []} for i in range(len(modes))}
    # per-mode CONSOLIDATION (global-phase) logs -> consolidation/{mode}/returns.csv
    cons_logs_by_idx = {i: {"global_step": [], "episodic_return": []} for i in range(len(modes))}
    # running per-mode step offset so csv steps are monotone across phases
    log_offset_by_idx = {i: 0 for i in range(len(modes))}

    frames_current = 0        # env steps spent learning the CURRENT mode (task0 + local phases)
    frames_consolidation = 0  # env steps spent replaying PAST+current modes (global phase)

    agents_root = f"./agents/{env_name}/{args.tag}"
    reporter = Reporter(env_name, args.tag, agents_root, modes, args)
    state_path = os.path.join(reporter.run_dir, "run_state.json")

    def save_state(last_completed):
        st = {"last_completed_task": last_completed,
              "modes": modes, "seed": args.seed,
              "local_refs": {int(i): local_refs[i] for i in local_refs},
              "log_offset_by_idx": {int(i): log_offset_by_idx[i] for i in log_offset_by_idx},
              "frames_current": frames_current,
              "frames_consolidation": frames_consolidation}
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, state_path)
        # persist a plain local_refs.json too (handy for eval / debugging)
        with open(os.path.join(reporter.run_dir, "local_refs.json"), "w") as f:
            json.dump({int(modes[i]): local_refs[i] for i in local_refs}, f, indent=2)

    start_k = 0
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        done = st["last_completed_task"]
        agent = OursAgent.load(f"{agents_root}/global_after_task{done}", None).to(device)
        local_refs.update({int(i): v for i, v in st["local_refs"].items()})
        log_offset_by_idx.update({int(i): v for i, v in st["log_offset_by_idx"].items()})
        frames_current = st["frames_current"]
        frames_consolidation = st["frames_consolidation"]
        # reload already-written returns so incremental flushes don't clobber them
        for i in range(done + 1):
            prior = _load_returns_csv(env_name, args.tag, modes[i])
            if prior is not None:
                logs_by_idx[i] = prior
        start_k = done + 1
        logger.info(f"[resume] loaded global_after_task{done}; continuing from task {start_k}")

    for k, mode in enumerate(modes):
        if k < start_k:
            continue
        agent.ensure_head(k)
        if k == 0:
            logger.info(f"=== Task 0 (mode {mode}): standard PPO -> initial global ===")
            reporter.status(task=0, mode=mode, phase="task0", percent=0.0)
            t_ph = time.time()
            steps = train_ppo_single_task(
                agent, args.env_id, mode, 0, args.seed, args.task1_steps,
                device, logs_by_idx[0], log_offset_by_idx[0],
                reporter=reporter, phase="task0", task_k=0, save_name="global_task0"
            )
            log_offset_by_idx[0] += steps
            frames_current += steps
            ref = greedy_eval(agent, args.env_id, mode, 0, args.seed,
                              args.eval_episodes, device)
            local_refs[0] = ref
            logger.info(f"[task0] mode{mode} local ref (greedy-{args.eval_episodes}) = {ref:.3f}")
            reporter.write_returns(mode, logs_by_idx[0])
            reporter.phase_summary(task=0, phase="task0", steps=steps,
                                   local_ref=round(ref, 3), wall_sec=round(time.time() - t_ph, 1))
            agent.save(f"{agents_root}/global_after_task0")
            save_state(0)
            continue

        # ---------------- Task k>=1 ----------------
        logger.info(f"=== Task {k} (mode {mode}): LOCAL phase (unconstrained PPO) ===")
        reporter.status(task=k, mode=mode, phase="local", percent=0.0)
        t_ph = time.time()
        local_agent = agent.clone().to(device)
        local_agent.ensure_head(k)
        steps_local = train_ppo_single_task(
            local_agent, args.env_id, mode, k, args.seed, args.local_steps,
            device, logs_by_idx[k], log_offset_by_idx[k],
            reporter=reporter, phase="local", task_k=k, save_name=f"local_task{k}"
        )
        log_offset_by_idx[k] += steps_local
        frames_current += steps_local
        for p in local_agent.parameters():
            p.requires_grad_(False)
        local_agent.eval()
        ref_k = greedy_eval(local_agent, args.env_id, mode, k, args.seed,
                            args.eval_episodes, device)
        local_refs[k] = ref_k
        logger.info(f"[task{k}] mode{mode} LOCAL ref (greedy-{args.eval_episodes}) = {ref_k:.3f}")
        # save the frozen local specialist (local_after_task{k})
        local_agent.save(f"{agents_root}/local_after_task{k}")
        reporter.phase_summary(task=k, phase="local", steps=steps_local,
                               local_ref=round(ref_k, 3), wall_sec=round(time.time() - t_ph, 1))

        logger.info(f"=== Task {k} (mode {mode}): GLOBAL consolidation phase ===")
        seen_idx = list(range(k + 1))
        cons_steps = train_global_phase(
            agent, local_agent, args.env_id, modes, seen_idx, k, args.seed,
            args.global_steps, args, device, cons_logs_by_idx, log_offset_by_idx,
            local_refs, ref_mode_by_idx=None, reporter=reporter
        )
        # returns.csv is the pure local-phase curve -> do NOT advance its offsets
        # for the global phase (consolidation is logged separately).
        frames_consolidation += cons_steps
        reporter.flush_all_returns(logs_by_idx)
        reporter.flush_all_consolidation(cons_logs_by_idx)
        agent.save(f"{agents_root}/global_after_task{k}")
        save_state(k)

    # final global
    agent.save(f"{agents_root}/final_global")

    # write per-mode returns.csv
    for k, mode in enumerate(modes):
        write_returns(env_name, args.tag, mode, logs_by_idx[k])

    # frame accounting
    total = frames_current + frames_consolidation
    accounting = {
        "env_id": args.env_id,
        "modes": modes,
        "frames_current_mode_learning": frames_current,
        "frames_past_mode_consolidation": frames_consolidation,
        "total_env_steps": total,
        "note": ("current = task0 PPO + per-task LOCAL PPO on the new mode; "
                 "consolidation = global-phase rollouts over ALL seen modes "
                 "(the disclosed past-task data-access asymmetry)."),
    }
    acc_dir = f"./data/{env_name}/{args.tag}/Ours"
    os.makedirs(acc_dir, exist_ok=True)
    with open(f"{acc_dir}/frame_accounting.json", "w") as f:
        json.dump(accounting, f, indent=2)
    logger.info(f"FRAME ACCOUNTING: current-mode={frames_current} "
                f"consolidation={frames_consolidation} total={total}")
    print(json.dumps(accounting, indent=2))


if __name__ == "__main__":
    main()
