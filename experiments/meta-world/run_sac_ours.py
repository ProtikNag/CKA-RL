"""Constrained min-max two-policy continual RL (SAC backend, Meta-World CW20).

Single-process, full-sequence runner over the CW20 task list. This is the
SAC/Meta-World analogue of the verified PPO port in
``experiments/atari/run_ours.py`` -- same method, same phase structure, same
Reporter pattern; only the single-task optimizer is swapped PPO -> SAC.

METHOD SUMMARY
--------------
  * Task 0: standard SAC on task 0 -> this IS the initial global. Its
    deterministic(mean)-policy discounted MC return is task 0's LOCAL value
    reference V_0^L; its deterministic success rate is the success reference.
  * Task k >= 1:
      - LOCAL phase: clone global -> local specialist, run UNCONSTRAINED
        standard SAC on task k from the warm start (updates the shared encoder,
        actor head k, and task-k critics). Freeze. Its deterministic MC return
        = local value reference V_k^L; its deterministic success = success ref.
        Save as local_after_task{k}.
      - GLOBAL consolidation phase: warm-start from the global. For
        ``global_iters`` iterations: collect a fresh on-policy rollout batch
        from each ACTIVE task (current task k + active past tasks) into that
        task's replay buffer, then run SAC gradient updates. The ACTOR loss is
        the NORMALIZED weighted SUM over active tasks of per-task SAC actor
        losses, with coeff_i = 1/(k+1) for past active tasks and
        mu*2*shortfall_k for the current task, divided by the sum of coeffs
        (exactly the PPO port). The per-task CRITICS are trained with the
        STANDARD SAC critic loss (NOT normalized, NOT constrained). Entropy
        alpha is standard (auto-tuned, shared). Two-timescale dual mu update
        every ``constraint_every`` iterations:
          shortfall_k = max(0, V_k^L - V_k^G),
          mu <- clip(mu + dual_lr*(shortfall_k^2 - eps_eff), 0, mu_max),
        where the tolerance is RELATIVE to the local reference value and matches
        Meta-World's dense-reward scale: eps_eff = (tol_frac * V_k^L)^2 (squared,
        same units as shortfall_k^2), with tol_frac default 0.1. A fixed
        absolute tolerance may be forced with --eps-abs (>= 0); the legacy
        --eps arg (Atari clipped-reward scale) is retained for compat only and
        is no longer used in the dual update.
        V_k^G is the deterministic(mean)-policy discounted MC return of the
        CURRENT global on task k (a separate eval rollout, NOT from replay),
        held fixed between refreshes. Retention-gated early stop over all seen
        tasks (success >= retention_frac * local success, or MC return >=
        retention_frac * V^L), a min_iters floor, and patience.

MIN-MAX -> SAC MAPPING (the key decision, also in models/ours.py docstring)
---------------------------------------------------------------------------
The min-max actor objective -- maximize past-task return while keeping the
current task's value at least its local reference -- is realized as a NORMALIZED
weighted sum of per-task SAC actor losses:

    actor_loss = sum_i  (coeff_i / Z) * L_actor^SAC(task i) ,   Z = sum_i coeff_i

where L_actor^SAC(task i) = E[ alpha*log pi_i(a|s) - min(Q1_i, Q2_i)(s,a) ] is
the standard SAC actor loss for task i using head i and critics i, over a fresh
on-policy batch for task i. The critics stay standard per task (summed, not
normalized). The constraint VALUE is the on-policy deterministic-MC return of
the current global; PAST-TASK ENV ACCESS is allowed -- that is this method's
premise (every iteration rolls out the active past tasks' real envs).

METRIC
------
The per-task LOCAL-phase per-episode success is logged to TensorBoard as
``charts/success`` under ``runs/{tag}/task_{k}__ours__run_sac__{seed}/`` with a
``hyperparameters/text_summary`` carrying seed/task_id/model_type so
extract_results.py picks it up UNCHANGED -- this is the plasticity-comparable
Table-1 curve (the pure current-task learning curve, matching the baselines'
returns). The GLOBAL/consolidation success curve is kept SEPARATE under the
SIBLING top-level dir ``cons_runs/{tag}/task_{k}__ours__run_sac__{seed}/`` --
NOT under ``runs/{tag}/`` -- so extract_results.py's recursive glob over
``runs/{tag}`` can never reach it and it never contaminates the Table-1 metric.

CRITICAL: this module is NOT run here; it is launched via SLURM. py_compile
only. Do not run on the login node.
"""
import os
import json
import math
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer

from models.ours import OursAgent
from tasks import get_task, tasks as CW20_TASKS
from contract_logging import ContractLogger


# ---------------- Reference SAC hyperparameters (run_sac.py) -----------------
GAMMA = 0.99
TAU = 0.005
BATCH_SIZE = 128
LEARNING_STARTS = 5_000
RANDOM_ACTIONS_END = 10_000
POLICY_LR = 1e-3
Q_LR = 1e-3
POLICY_FREQUENCY = 2
TARGET_NETWORK_FREQUENCY = 1
BUFFER_SIZE = int(1e6)
EP_LEN = 500  # Meta-World episode length
# Fixed entropy scale for the windowed-bootstrap value's -alpha*log_pi term.
# Used on BOTH sides of the shortfall (Vk_L and Vk_G) so the entropy bias is
# identical and cancels; matches the non-autotune SAC default (0.2).
BOOT_ALPHA = 0.2

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def build_env(task_id, seed, n_envs=1):
    """Build a vectorized env with ``n_envs`` copies of ``task_id``.

    n_envs>1 uses AsyncVectorEnv (subprocess parallelism); it is spawn-fragile
    on some clusters, so construction is wrapped in try/except and falls back to
    a (serial) SyncVectorEnv of the same width on failure.
    """
    n_envs = max(1, int(n_envs))
    if n_envs > 1:
        try:
            envs = gym.vector.AsyncVectorEnv([make_env(task_id) for _ in range(n_envs)])
        except Exception as e:
            logger.warning(f"AsyncVectorEnv(n_envs={n_envs}) failed ({e!r}); "
                           f"falling back to SyncVectorEnv")
            envs = gym.vector.SyncVectorEnv([make_env(task_id) for _ in range(n_envs)])
    else:
        envs = gym.vector.SyncVectorEnv([make_env(task_id)])
    envs.single_observation_space.dtype = np.float32
    return envs


# ---- persistent eval-env cache (deterministic_eval / windowed_shortfall reuse) ---
# Building + closing a (possibly Async) vector env on every eval call is slow;
# cache one per (task_id, n_envs) and reuse. close_eval_envs() tears them down.
_EVAL_ENV_CACHE = {}


def get_eval_env(task_id, seed, n_envs=1):
    key = (int(task_id), int(n_envs))
    if key not in _EVAL_ENV_CACHE:
        _EVAL_ENV_CACHE[key] = build_env(task_id, seed, n_envs=n_envs)
    return _EVAL_ENV_CACHE[key]


def close_eval_envs():
    for e in _EVAL_ENV_CACHE.values():
        try:
            e.close()
        except Exception:
            pass
    _EVAL_ENV_CACHE.clear()


def _action_bounds(task_id, seed):
    """(action_low, action_high) for a task's Box action space, env closed."""
    e = build_env(task_id, seed)
    low = e.single_action_space.low
    high = e.single_action_space.high
    e.close()
    return low, high


# ---------------- Actor squash/rescale (copied verbatim from run_sac.py) -----
class ActorHelper:
    """Holds the action rescale buffers and implements get_action() exactly as
    run_sac.py's Actor, but driving an OursAgent head selected by ``task_idx``.

    Not an nn.Module: it only wraps the agent + fixed rescale tensors, so its
    presence does not add parameters to any optimizer (the agent's actor params
    are optimized directly).
    """

    def __init__(self, agent, action_low, action_high, device):
        self.agent = agent
        self.action_scale = torch.tensor(
            (action_high - action_low) / 2.0, dtype=torch.float32, device=device
        )
        self.action_bias = torch.tensor(
            (action_high + action_low) / 2.0, dtype=torch.float32, device=device
        )

    def _mean_logstd(self, x, task_idx):
        mean, log_std = self.agent(x, task_idx=task_idx)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x, task_idx):
        mean, log_std = self._mean_logstd(x, task_idx)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_a = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_a

    def deterministic_action(self, x, task_idx):
        mean, _ = self._mean_logstd(x, task_idx)
        return torch.tanh(mean) * self.action_scale + self.action_bias


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=str, default="",
                   help="comma-separated CW20 task indices = task sequence; "
                        "empty -> use --num-tasks")
    p.add_argument("--num-tasks", type=int, default=20,
                   help="if --tasks empty, run tasks 0..num_tasks-1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", type=str, default="smoke")
    p.add_argument("--task1-steps", type=int, default=int(1e6),
                   help="SAC env steps for task 0 (standard SAC -> initial global)")
    p.add_argument("--local-steps", type=int, default=int(1e6),
                   help="SAC env steps for each task-k LOCAL specialist phase")
    p.add_argument("--global-iters", type=int, default=2000,
                   help="# global-phase iterations; each rolls out ALL ACTIVE "
                        "tasks one episode and does SAC updates")
    p.add_argument("--min-iters", type=int, default=500,
                   help="floor before retention-gated early-stop may trigger")
    p.add_argument("--consolidate-mode", type=str, default="all",
                   choices=["all", "needy"],
                   help="global-phase past-task set each iter: 'all' seen tasks "
                        "(canonical min-max) or 'needy' = only past tasks below "
                        "their retention bar (+ current) -- targeted variant")
    p.add_argument("--constraint-episodes", type=int, default=8,
                   help="# deterministic MC episodes per policy for the shortfall")
    p.add_argument("--constraint-every", type=int, default=20,
                   help="dual mu refresh cadence in global iterations")
    p.add_argument("--eps", type=float, default=0.04,
                   help="legacy absolute tolerance (Atari scale); NOT used for "
                        "the dual update unless --eps-abs is set. Kept for "
                        "backward compat / logging only.")
    p.add_argument("--tol-frac", type=float, default=0.1,
                   help="relative dual tolerance: eps_eff = (tol_frac * V_k^L)^2 "
                        "per task, matching Meta-World's dense-reward scale")
    p.add_argument("--eps-abs", type=float, default=-1.0,
                   help="OPTIONAL absolute eps override: if >= 0, use this fixed "
                        "value as the dual tolerance instead of eps_eff "
                        "(=(tol_frac*V_k^L)^2). Default (<0) -> use eps_eff.")
    p.add_argument("--dual-lr", type=float, default=0.3)
    p.add_argument("--mu-max", type=float, default=5.0)
    p.add_argument("--retention-frac", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--stop-eval-every", type=int, default=200,
                   help="retention-gated early-stop check cadence (global iters)")
    p.add_argument("--stop-eval-episodes", type=int, default=3,
                   help="# deterministic episodes for each cheap retention check")
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="# deterministic episodes for the REPORTED local reference")
    p.add_argument("--log-every", type=int, default=5,
                   help="flush progress.jsonl / status.json every N iterations")
    p.add_argument("--ckpt-every", type=int, default=100,
                   help="within-task rolling checkpoint cadence in iterations "
                        "(0 = only at task boundaries)")
    p.add_argument("--global-updates-per-iter", type=int, default=500,
                   help="# SAC gradient steps per global-phase iteration "
                        "(after collecting one episode per active task)")
    p.add_argument("--resume", action="store_true",
                   help="resume from the last completed task boundary in this tag")
    p.add_argument("--cuda", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
                   default=True)
    p.add_argument("--debug", action="store_true")
    # ---- speedup stack (each behind its own flag; defaults keep new path on) --
    _bool = lambda v: str(v).lower() in ("1", "true", "yes", "y")
    p.add_argument("--n-envs", type=int, default=8,
                   help="# parallel envs per task (Async if >1, else Sync). "
                        "total env-step budgets are unchanged; --n-envs 1 == "
                        "original single-env behavior.")
    p.add_argument("--value-mode", type=str, default="bootstrap",
                   choices=["mc", "bootstrap"],
                   help="TRAINING constraint value: 'mc' = full deterministic "
                        "MC rollout (original); 'bootstrap' = windowed "
                        "value-gap with a gamma^H critic bootstrap.")
    p.add_argument("--window-H", type=int, default=200,
                   help="window length H for --value-mode bootstrap")
    p.add_argument("--n-boot-samples", type=int, default=4,
                   help="# sampled actions averaged for each soft state value "
                        "V(s) endpoint in --value-mode bootstrap (variance "
                        "reduction on the -alpha*log_pi soft-V estimate)")
    p.add_argument("--fixed-alpha", type=float, default=-1.0,
                   help="if >=0, disable SAC entropy autotune and pin alpha to this "
                        "value (exploration knob; autotune collapses alpha on hard tasks)")
    p.add_argument("--crossq", type=_bool, default=True,
                   help="CrossQ critic (BatchNorm, no target net, joint "
                        "forward). Critic-only; actor unchanged. False -> "
                        "standard SAC target-net critic.")
    p.add_argument("--contract-eval-every", type=int, default=25000,
                   help="within-phase greedy-eval cadence (env steps) for the "
                        "ContractLogger FWT curve during single-task phases")
    p.add_argument("--torch-threads", type=int, default=0,
                   help="if >0, torch.set_num_threads(this) to avoid "
                        "oversubscription with N async env processes")
    return p.parse_args()


# ---------------- observability: crash-safe logging + checkpoints ------------
class Reporter:
    """Persists progress as the run proceeds (mirrors the Atari Reporter).

    Writes under data/{tag}/Ours/ :
      - progress.jsonl          : one row per logged iteration (append+fsync)
      - status.json             : single-object heartbeat (atomic replace)
      - retention_history.jsonl : per retention-gated check, per-task score
      - phase_summaries.jsonl   : one row when each phase finishes
      - run_state.json          : --resume bookmark
      - local_refs.json         : task_idx -> {value, success}
    Checkpoints under agents/{tag}/ (rolling within-task + at boundaries).
    """

    def __init__(self, tag, agents_root, task_ids, args):
        self.tag = tag
        self.agents_root = agents_root
        self.task_ids = task_ids
        self.args = args
        self.run_dir = f"./data/{tag}/Ours"
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
        os.replace(tmp, self.status_path)

    def checkpoint(self, agent, name):
        agent.save(f"{self.agents_root}/{name}")


def _hparams_text(d):
    return "|param|value|\n|-|-|\n%s" % (
        "\n".join([f"|{k}|{v}|" for k, v in d.items()])
    )


def make_tb_writer(tag, task_idx, seed, consolidation=False):
    """TB run matching extract_results.py's expectation.

    run_name = task_{k}__ours__run_sac__{seed}. A hyperparameters/text_summary
    carrying seed/task_id/model_type must be present so extract_results.py
    parse_metadata() can tag the rows. Consolidation curves go under the
    SIBLING top-level dir cons_runs/{tag}/ (NOT under runs/{tag}/) so the
    recursive extract glob over runs/{tag} never lands them in the Table-1
    extract."""
    run_name = f"task_{task_idx}__ours__run_sac__{seed}"
    # Consolidation curves are written to a SIBLING top-level dir (cons_runs/)
    # that is NOT under runs/{tag}/, so extract_results.py's recursive
    # rglob("*events.out*") under runs/{tag} can never reach them and they
    # cannot contaminate the Table-1 (plasticity) curve. Local-phase curves
    # stay under runs/{tag}/... where extract_results.py SHOULD pick them up.
    base = "cons_runs" if consolidation else "runs"
    writer = SummaryWriter(f"{base}/{tag}/{run_name}")
    writer.add_text(
        "hyperparameters",
        _hparams_text({"seed": seed, "task_id": task_idx,
                       "model_type": "ours", "exp_name": "run_sac"}),
    )
    return writer


# ---------------- SAC update shared helpers ----------------------------------
def sac_critic_loss(agent, actor, rb_data, task_idx, alpha, device):
    """Standard SAC twin-Q critic loss for one task (run_sac.py verbatim)."""
    with torch.no_grad():
        next_actions, next_logpi, _ = actor.get_action(rb_data.next_observations, task_idx)
        qf1_next = agent.qf1_target[task_idx](rb_data.next_observations, next_actions)
        qf2_next = agent.qf2_target[task_idx](rb_data.next_observations, next_actions)
        min_q_next = torch.min(qf1_next, qf2_next) - alpha * next_logpi
        next_q = rb_data.rewards.flatten() + (1 - rb_data.dones.flatten()) * GAMMA * min_q_next.view(-1)
    qf1_a = agent.qf1[task_idx](rb_data.observations, rb_data.actions).view(-1)
    qf2_a = agent.qf2[task_idx](rb_data.observations, rb_data.actions).view(-1)
    qf1_loss = F.mse_loss(qf1_a, next_q)
    qf2_loss = F.mse_loss(qf2_a, next_q)
    return qf1_loss + qf2_loss


def sac_actor_loss(agent, actor, rb_data, task_idx, alpha, device):
    """Standard SAC actor loss for one task; also returns log_pi for alpha."""
    pi, log_pi, _ = actor.get_action(rb_data.observations, task_idx)
    qf1_pi = agent.qf1[task_idx](rb_data.observations, pi)
    qf2_pi = agent.qf2[task_idx](rb_data.observations, pi)
    min_q_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
    return actor_loss, log_pi


# ---------------- CrossQ update helpers (critic-only; no target net) ----------
def crossq_critic_loss(agent, actor, rb_data, task_idx, alpha, device):
    """CrossQ (Bhatt et al. ICLR 2024) twin-Q critic loss -- NO target net.

    Current (obs, act) and next (next_obs, a') are forwarded JOINTLY through each
    online critic so BatchNorm normalizes both with the SAME batch statistics
    (the crux of CrossQ). The policy that produces a' carries no gradient; the
    Q(next) branch is forwarded WITH grad through the critic then the whole
    target y is detached, so only the current-Q branch trains the critic.
    """
    qf1, qf2 = agent.qf1[task_idx], agent.qf2[task_idx]
    with torch.no_grad():
        next_actions, next_logpi, _ = actor.get_action(rb_data.next_observations, task_idx)
    cat_obs = torch.cat([rb_data.observations, rb_data.next_observations], dim=0)
    cat_act = torch.cat([rb_data.actions, next_actions], dim=0)
    n = rb_data.observations.shape[0]
    qf1.train(); qf2.train()
    q1_all = qf1(cat_obs, cat_act)
    q2_all = qf2(cat_obs, cat_act)
    q1_sa, q1_next = q1_all[:n], q1_all[n:]
    q2_sa, q2_next = q2_all[:n], q2_all[n:]
    with torch.no_grad():
        min_q_next = torch.min(q1_next, q2_next) - alpha * next_logpi
        y = (rb_data.rewards.flatten()
             + (1 - rb_data.dones.flatten()) * GAMMA * min_q_next.view(-1)).detach()
    loss = F.mse_loss(q1_sa.view(-1), y) + F.mse_loss(q2_sa.view(-1), y)
    return loss


def crossq_actor_loss(agent, actor, rb_data, task_idx, alpha, device):
    """CrossQ actor loss: identical to sac_actor_loss but the task's critics are
    put in .eval() around the Q evaluation so BatchNorm uses running stats and
    the actor step does NOT pollute the critic's batch statistics."""
    pi, log_pi, _ = actor.get_action(rb_data.observations, task_idx)
    qf1, qf2 = agent.qf1[task_idx], agent.qf2[task_idx]
    was1, was2 = qf1.training, qf2.training
    qf1.eval(); qf2.eval()
    qf1_pi = qf1(rb_data.observations, pi)
    qf2_pi = qf2(rb_data.observations, pi)
    qf1.train(was1); qf2.train(was2)
    min_q_pi = torch.min(qf1_pi, qf2_pi)
    actor_loss = ((alpha * log_pi) - min_q_pi).mean()
    return actor_loss, log_pi


# ---------------- deterministic MC value + success ---------------------------
@torch.no_grad()
def deterministic_eval(actor, task_id, task_idx, seed, n_episodes, device,
                      gamma=GAMMA, n_envs=1):
    """Run >= n_episodes DETERMINISTIC (mean) episodes on task `task_id` with head
    `task_idx`. Returns (mc_discounted_return, success_rate).

    mc_discounted_return is the Monte-Carlo discounted return V^pi used for the
    shortfall / constraint (gamma^t reward accumulation, reset per episode),
    analogous to the Atari mc_stochastic_value but with the DETERMINISTIC policy
    (the spec requires the deterministic(mean) policy for the constraint value).
    success_rate is the mean of native info["success"] at episode end. With
    n_envs>1, episodes are collected in parallel (each env's per-episode
    discount resets independently)."""
    envs = get_eval_env(task_id, seed + 777, n_envs=n_envs)
    N = envs.num_envs
    obs, _ = envs.reset(seed=seed + 777)
    disc_returns = []
    successes = []
    disc = np.zeros(N, dtype=np.float64)
    gpow = np.ones(N, dtype=np.float64)
    while len(disc_returns) < n_episodes:
        obs_t = torch.Tensor(obs).to(device)
        action = actor.deterministic_action(obs_t, task_idx)
        obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        disc += gpow * reward
        gpow *= gamma
        done = np.logical_or(np.asarray(term).reshape(-1),
                             np.asarray(trunc).reshape(-1))
        if "final_info" in infos:
            final_info = infos["final_info"]
            for idx in range(N):
                if not done[idx]:
                    continue
                succ = 0.0
                info = final_info[idx] if final_info is not None else None
                if info is not None and "success" in info:
                    succ = float(info["success"])
                disc_returns.append(disc[idx])
                successes.append(succ)
                disc[idx] = 0.0
                gpow[idx] = 1.0
    return float(np.mean(disc_returns[:n_episodes])), float(np.mean(successes[:n_episodes]))


@torch.no_grad()
def _soft_value(owner, actor, task_idx, obs_np, alpha, device, n_boot_samples=4):
    """Per-state SAC soft state value V(s) using ``owner``'s task-idx critics.

    V(s) = mean over n_boot_samples SAMPLED actions of
        [ min(Q1,Q2)(s,a) - alpha*log pi(a|s) ]
    (variance-reduced soft-V; sampled actions rather than the deterministic mean,
    whose log-density is pathological near tanh saturation). ``owner``'s critics
    are put in .eval() around the Q evaluations (BN running stats; no-op for
    plain SAC critics) and their prior training mode is restored. Returns a
    per-state numpy vector (one entry per row of ``obs_np``)."""
    s = torch.Tensor(obs_np).to(device)
    qf1, qf2 = owner.qf1[task_idx], owner.qf2[task_idx]
    was1, was2 = qf1.training, qf2.training
    qf1.eval(); qf2.eval()
    v_samples = []
    for _ in range(max(1, int(n_boot_samples))):
        a, logp_a, _ = actor.get_action(s, task_idx)
        q1 = qf1(s, a).view(-1)
        q2 = qf2(s, a).view(-1)
        v_samples.append(torch.min(q1, q2) - alpha * logp_a.view(-1))
    qf1.train(was1); qf2.train(was2)
    return torch.stack(v_samples, dim=0).mean(dim=0).detach().cpu().numpy()


@torch.no_grad()
def windowed_shortfall(local_owner, local_actor, roll_actor, task_id, task_idx,
                       seed, n_starts, H, device, alpha, gamma=GAMMA, n_envs=1,
                       n_boot_samples=4):
    """Single-critic windowed shortfall endpoints (Vk_L, Vk_G).

    BOTH state values V(S_0) and V(S_H) are computed with the SAME frozen LOCAL
    specialist's critic + local actor (``local_owner`` / ``local_actor``), so the
    local critic's systematic bias largely cancels in the shortfall
    Vk_L - Vk_G. Only the H-step reward term is rolled with the CURRENT GLOBAL
    policy (``roll_actor``):

        Vk_L = V_L(S_0)                              (local-critic baseline value)
        Vk_G = sum_{t<H} gamma^t r_t  +  gamma^H * V_L(S_H)
               (global policy's H-step rollout reward + local-critic tail)

    The global critic is NO LONGER used for the constraint value (it is still
    used for the SAC/CrossQ actor-improvement updates elsewhere).

    Mid-window termination: once an env terminates, its reward accumulation and
    its bootstrap term are masked to 0 (a per-env done mask + gpow freeze).

    Returns (Vk_L, Vk_G) each averaged over the first min(n_starts, N) envs."""
    envs = get_eval_env(task_id, seed + 777, n_envs=n_envs)
    N = envs.num_envs
    obs, _ = envs.reset(seed=seed + 777)
    S0 = obs
    # Vk_L baseline: local-critic soft value at the start states (per-env).
    v0 = _soft_value(local_owner, local_actor, task_idx, S0, alpha, device,
                     n_boot_samples)
    ret = np.zeros(N, dtype=np.float64)
    gpow = np.ones(N, dtype=np.float64)
    done_mask = np.zeros(N, dtype=bool)  # True once an env has terminated
    last_obs = obs
    for t in range(H):
        obs_t = torch.Tensor(obs).to(device)
        action = roll_actor.deterministic_action(obs_t, task_idx)  # GLOBAL policy
        obs, reward, term, trunc, infos = envs.step(action.cpu().numpy())
        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        active = ~done_mask
        ret[active] += gpow[active] * reward[active]
        gpow[active] *= gamma
        # freeze envs that terminated THIS step (their S_H bootstrap is masked).
        term = np.asarray(term).reshape(-1)
        trunc = np.asarray(trunc).reshape(-1)
        # a genuine termination ends the episode's value; a time-limit trunc
        # does NOT (autoreset happens, but we treat it as continuation and just
        # stop trusting that env's window -> mask it too, conservatively).
        newly = np.logical_and(active, np.logical_or(term, trunc))
        done_mask = np.logical_or(done_mask, newly)
        last_obs = obs  # S_H candidate (autoreset-safe: masked envs ignored)
        if done_mask.all():
            break
    # bootstrap tail V_L(S_H) with the SAME frozen LOCAL critic (not the global).
    vH = _soft_value(local_owner, local_actor, task_idx, last_obs, alpha, device,
                     n_boot_samples)
    boot = (gamma ** H) * vH
    boot[done_mask] = 0.0  # terminated envs contribute no bootstrap
    est = ret + boot       # Vk_G, per-env
    m = min(n_starts, N)
    return float(np.mean(v0[:m])), float(np.mean(est[:m]))


# ---------------- standard single-task SAC (task0 + local phase) -------------
def train_sac_single_task(agent, task_id, task_idx, seed, total_steps, device,
                          writer, reporter=None, phase="sac", task_k=None,
                          save_name=None, autotune=True, clog=None,
                          clog_task_idx=None, clog_phase=None):
    """Standard unconstrained SAC on one task using head `task_idx`.

    Updates the shared encoder, actor head task_idx, and task-idx critics. Logs
    per-episode charts/success, charts/episodic_return, charts/episodic_length
    to `writer` (the plasticity-comparable Table-1 curve). Returns env steps
    consumed. Mirrors run_sac.py's training loop exactly, restricted to head
    task_idx's parameters."""
    n_envs = getattr(reporter.args, "n_envs", 1) if reporter else 1
    crossq = getattr(reporter.args, "crossq", False) if reporter else False
    envs = build_env(task_id, seed, n_envs=n_envs)
    N = envs.num_envs
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high
    actor = ActorHelper(agent, action_low, action_high, device)
    critic_loss_fn = crossq_critic_loss if crossq else sac_critic_loss
    actor_loss_fn = crossq_actor_loss if crossq else sac_actor_loss

    # optimizers: actor side = shared encoder + this head's mean/logstd; q side
    # = this task's twin critics. (Only head task_idx trains in this phase.)
    actor_params = (list(agent.actor.fc.parameters())
                    + list(agent.actor.fc_mean[task_idx].parameters())
                    + list(agent.actor.fc_logstd[task_idx].parameters()))
    q_params = (list(agent.qf1[task_idx].parameters())
                + list(agent.qf2[task_idx].parameters()))
    actor_optimizer = optim.Adam(actor_params, lr=POLICY_LR)
    # CrossQ uses Adam betas=(0.5, 0.999) for the critic (paper recommendation).
    q_betas = (0.5, 0.999) if crossq else (0.9, 0.999)
    q_optimizer = optim.Adam(q_params, lr=Q_LR, betas=q_betas)

    # --fixed-alpha (>=0) disables entropy autotune and PINS alpha (exploration knob);
    # a higher fixed alpha keeps the policy stochastic so it keeps exploring on hard
    # contact-rich tasks where autotune collapses alpha (~0.005) into a no-reward basin.
    fixed_alpha = getattr(reporter.args, "fixed_alpha", -1.0) if reporter else -1.0
    if autotune and fixed_alpha < 0:
        target_entropy = -float(np.prod(envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=Q_LR)
    else:
        autotune = False
        alpha = fixed_alpha if fixed_alpha >= 0 else 0.2

    # SB3 divides buffer_size by n_envs internally, so pass the FULL BUFFER_SIZE
    # (not BUFFER_SIZE//N) to keep total capacity ~constant across n_envs.
    rb = ReplayBuffer(BUFFER_SIZE, envs.single_observation_space,
                      envs.single_action_space, device, n_envs=N,
                      handle_timeout_termination=False)

    log_every = getattr(reporter.args, "log_every", 5) if reporter else 10**9
    ckpt_every = getattr(reporter.args, "ckpt_every", 0) if reporter else 0
    contract_eval_every = getattr(reporter.args, "contract_eval_every", 0) if reporter else 0
    ce_episodes = getattr(reporter.args, "stop_eval_episodes", 3) if reporter else 3
    ep_count = 0
    t0 = time.time()
    last_actor = last_q = float("nan")
    # best-checkpoint: snapshot the peak greedy-success weights and restore at the
    # end, so a late collapse/oscillation (e.g. peg-unplug 1.0->0.0) doesn't cost
    # us the learned policy. Mirrors the Atari local trainer's select-best.
    best_succ, best_state = -1.0, None
    grad_steps = 0  # global gradient-step counter (UTD cadence, n_envs-invariant)

    obs, _ = envs.reset(seed=seed)
    # total_steps is an ENV-STEP budget; each iteration consumes N env steps, so
    # loop total_steps // N times. env-step thresholds compare against step * N.
    for step in range(total_steps // N):
        env_step = step * N  # # env steps consumed BEFORE this iteration
        if env_step < RANDOM_ACTIONS_END:
            actions = np.array([envs.single_action_space.sample() for _ in range(N)])
        else:
            a, _, _ = actor.get_action(torch.Tensor(obs).to(device), task_idx)
            actions = a.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for idx in range(N):
                info = infos["final_info"][idx]
                if info is None:
                    continue
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], env_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], env_step)
                writer.add_scalar("charts/success", info["success"], env_step)
                ep_count += 1

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        if env_step > LEARNING_STARTS:
            # UTD-preserving: do N gradient steps per iteration (one per env step
            # collected this iter) so the TOTAL update count == the n_envs=1 run.
            # n_envs then changes only env-stepping wall-clock, NOT the learning
            # dynamics. A GLOBAL grad_steps counter keeps the POLICY_FREQUENCY and
            # target-sync cadence identical across n_envs (a per-iter g%FREQ gate
            # would fire every iteration and double the actor rate at n_envs=1).
            for _ in range(N):
                data = rb.sample(BATCH_SIZE)
                q_loss = critic_loss_fn(agent, actor, data, task_idx, alpha, device)
                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()
                last_q = float(q_loss.item())

                if grad_steps % POLICY_FREQUENCY == 0:
                    for _ in range(POLICY_FREQUENCY):
                        actor_loss, _ = actor_loss_fn(agent, actor, data, task_idx, alpha, device)
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        last_actor = float(actor_loss.item())
                        if autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations, task_idx)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                # CrossQ has no target net -> skip sync (guarded no-op anyway).
                if not crossq and grad_steps % TARGET_NETWORK_FREQUENCY == 0:
                    agent.sync_targets(task_idx, TAU)
                grad_steps += 1

        # ---- stream progress / status / rolling checkpoint ----
        if reporter is not None and env_step > 0 and env_step % (log_every * 1000) < N:
            sps = int(env_step / max(1e-9, time.time() - t0))
            frac_done = env_step / max(1, total_steps)
            reporter.progress(task=task_k, task_id=task_id, phase=phase,
                              global_step=env_step, total_steps=total_steps,
                              sps=sps, episodes=ep_count,
                              actor_loss=round(last_actor, 5), q_loss=round(last_q, 5),
                              alpha=round(float(alpha), 5))
            reporter.status(task=task_k, task_id=task_id, phase=phase,
                            global_step=env_step, total_steps=total_steps,
                            percent=round(100 * frac_done, 1), sps=sps,
                            phase_elapsed_sec=round(time.time() - t0, 1),
                            phase_eta_sec=round((time.time() - t0) * (1 - frac_done) / max(1e-9, frac_done), 1))
        if reporter is not None and ckpt_every and save_name and \
                env_step > 0 and env_step % (ckpt_every * 1000) < N:
            reporter.checkpoint(agent, f"{save_name}_latest")

        # ---- within-phase greedy eval -> ContractLogger (FWT curve) ----
        if (clog is not None and contract_eval_every > 0 and env_step > 0
                and env_step % contract_eval_every < N):
            succ = deterministic_eval(actor, task_id, task_idx, seed,
                                      ce_episodes, device, n_envs=n_envs)[1]
            if succ > best_succ:
                best_succ = succ
                best_state = {k: v.detach().clone() for k, v in agent.state_dict().items()}
            ti = clog_task_idx if clog_task_idx is not None else (task_k or 0)
            clog.eval(task_idx=ti, phase=(clog_phase or phase), it=int(env_step),
                      evaluated_on=ti, evaluated_on_task=CW20_TASKS[task_id],
                      raw=succ, episodes=ce_episodes, greedy=True, seen=True)

    # best-checkpoint restore: if the peak greedy beat the final policy, keep the peak.
    if best_state is not None:
        final_succ = deterministic_eval(actor, task_id, task_idx, seed,
                                        ce_episodes, device, n_envs=n_envs)[1]
        if best_succ > final_succ:
            agent.load_state_dict(best_state)
            print(f"[best-ckpt task{task_idx}] restored peak greedy={best_succ:.3f} "
                  f"(final was {final_succ:.3f})", flush=True)
    envs.close()
    return (total_steps // N) * N


# ---------------- global consolidation phase ---------------------------------
def train_global_phase(agent, local_agent, task_ids, seen_idx, k, seed, args,
                       device, local_value_refs, local_success_refs,
                       cons_writers, reporter=None, autotune=True, clog=None):
    """Min-max consolidation. `seen_idx` = task indices 0..k; current = k.

    Each iteration: for every ACTIVE task, collect ONE fresh on-policy episode
    into that task's replay buffer, then run `global_updates_per_iter` SAC
    gradient steps. The ACTOR loss each step is the NORMALIZED weighted sum of
    per-task SAC actor losses (coeff_i = 1/(k+1) past, mu*2*shortfall_k current);
    per-task critics use the STANDARD SAC loss. Two-timescale mu update every
    constraint_every iters with the deterministic-MC shortfall. Retention-gated
    early stop over all seen tasks.
    """
    needy_mode = (getattr(args, "consolidate_mode", "all") == "needy")
    n_envs = getattr(args, "n_envs", 1)
    crossq = getattr(args, "crossq", False)
    value_mode = getattr(args, "value_mode", "mc")
    window_H = getattr(args, "window_H", 200)
    critic_loss_fn = crossq_critic_loss if crossq else sac_critic_loss
    actor_loss_fn = crossq_actor_loss if crossq else sac_actor_loss
    past = [i for i in seen_idx if i < k]

    # per-active-task env + replay buffer + obs state, managed lazily so only
    # |active| envs/buffers are open in 'needy' mode (memory + speed).
    envs_by_idx, rb_by_idx, obs_by_idx = {}, {}, {}
    # one shared Actor wrapper over the global agent (rescale buffers identical
    # across Meta-World tasks: same Box action space).
    probe = build_env(task_ids[k], seed)
    action_low = probe.single_action_space.low
    action_high = probe.single_action_space.high
    obs_space = probe.single_observation_space
    act_space = probe.single_action_space
    probe.close()
    actor = ActorHelper(agent, action_low, action_high, device)
    # Frozen LOCAL specialist actor -- drives BOTH endpoints' soft-V via the
    # frozen local critic (bootstrap mode). Built once; None when no local_agent
    # (task 0 has no local phase).
    local_actor_ref = (ActorHelper(local_agent, action_low, action_high, device)
                       if local_agent is not None else None)

    def open_task(i):
        if i in envs_by_idx:
            return
        e = build_env(task_ids[i], seed + 1000 + i, n_envs=n_envs)
        o, _ = e.reset(seed=seed + 1000 + i)
        envs_by_idx[i] = e
        obs_by_idx[i] = o
        rb_by_idx[i] = ReplayBuffer(BUFFER_SIZE, obs_space, act_space, device,
                                    n_envs=e.num_envs,
                                    handle_timeout_termination=False)

    def close_task(i):
        if i in envs_by_idx:
            envs_by_idx[i].close()
            del envs_by_idx[i]
            obs_by_idx.pop(i, None)
            rb_by_idx.pop(i, None)

    def set_active(active):
        for i in active:
            open_task(i)
        for i in list(envs_by_idx.keys()):
            if i not in active:
                close_task(i)

    def det_scores_all():
        """cheap deterministic (value, success) for every seen task."""
        out = {}
        for i in seen_idx:
            v, s = deterministic_eval(actor, task_ids[i], i, seed,
                                      args.stop_eval_episodes, device, n_envs=n_envs)
            out[i] = (v, s)
        return out

    def needy_past(scores):
        # a past task is 'needy' if BELOW its retention bar on BOTH success and
        # value (mirrors the Atari needy check; here we require either metric
        # below bar -> still needs replay).
        res = []
        for i in past:
            v, s = scores[i]
            below_succ = s < args.retention_frac * local_success_refs[i]
            below_val = v < args.retention_frac * local_value_refs[i]
            if below_succ or below_val:
                res.append(i)
        return res

    # ---- collect one episode worth of env steps into task i's replay buffer --
    # With N envs, collect N parallel partial episodes for ceil(EP_LEN/N) loop
    # steps and count EXACTLY EP_LEN env steps (last step clamped). Returns the
    # ENV-STEP count added (== EP_LEN).
    def collect_episode(i):
        e = envs_by_idx[i]
        N = e.num_envs
        obs = obs_by_idx[i]
        loop_steps = max(1, math.ceil(EP_LEN / N))
        env_steps = 0
        for _ in range(loop_steps):
            if env_steps >= EP_LEN:
                break
            obs_t = torch.Tensor(obs).to(device)
            with torch.no_grad():
                a, _, _ = actor.get_action(obs_t, i)
            actions = a.detach().cpu().numpy()
            next_obs, rewards, terminations, truncations, infos = e.step(actions)
            if "final_info" in infos:
                for idx in range(N):
                    info = infos["final_info"][idx]
                    if info is None:
                        continue
                    # log to the SEPARATE consolidation writer for task i
                    w = cons_writers[i]
                    w.add_scalar("charts/episodic_return", info["episode"]["r"], iteration_global_step[0])
                    w.add_scalar("charts/success", info["success"], iteration_global_step[0])
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            rb_by_idx[i].add(obs, real_next_obs, actions, rewards, terminations, infos)
            obs = next_obs
            # count env steps EXACTLY: the last partial loop step over-steps the
            # vector env by (N * loop_steps - EP_LEN); clamp the ACCOUNTING to
            # EP_LEN so the returned/global env-step total is exact (n_envs=1 is
            # unchanged: EP_LEN divides evenly). Extra transitions in the buffer
            # are harmless.
            add = min(N, EP_LEN - env_steps)
            env_steps += add
            iteration_global_step[0] += add
        obs_by_idx[i] = obs
        return env_steps

    # ---- initial active set ----
    if needy_mode:
        active_past = needy_past(det_scores_all())
    else:
        active_past = list(past)
    active = active_past + [k]
    set_active(active)
    logger.info(f"[global k={k}] consolidate={'needy' if needy_mode else 'all'} "
                f"initial active tasks={[task_ids[i] for i in active]}")

    # optimizers over ALL active heads + their critics (+ shared encoder once).
    # Rebuilt when the active set changes (needy mode) so new heads are included.
    def build_optimizers(active):
        actor_params = list(agent.actor.fc.parameters())
        for i in active:
            actor_params += list(agent.actor.fc_mean[i].parameters())
            actor_params += list(agent.actor.fc_logstd[i].parameters())
        q_params = []
        for i in active:
            q_params += list(agent.qf1[i].parameters())
            q_params += list(agent.qf2[i].parameters())
        q_betas = (0.5, 0.999) if crossq else (0.9, 0.999)
        return (optim.Adam(actor_params, lr=POLICY_LR),
                optim.Adam(q_params, lr=Q_LR, betas=q_betas))

    actor_optimizer, q_optimizer = build_optimizers(active)

    if autotune:
        target_entropy = -float(np.prod(act_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=Q_LR)
    else:
        alpha = 0.2

    num_iterations = args.global_iters
    mu = 0.0
    consec_ok = 0
    # best-checkpoint: snapshot the model with the highest MEAN greedy success over
    # all seen tasks (= the PERF metric) and restore it at the end, so a late
    # consolidation collapse/drift doesn't cost retention (the Atari Boxing lesson).
    best_ret, best_state = -1.0, None
    total_consolidation_steps = 0
    iteration_global_step = [0]  # mutable counter shared with collect_episode

    # Both shortfall endpoints use the SAME frozen local specialist's critic:
    #   Vk_L = V_L(S_0)                              (local-critic baseline value)
    #   Vk_G = sum_{t<H} gamma^t r_t + gamma^H V_L(S_H)  (global rollout + local tail)
    # so the local critic's systematic bias largely cancels in Vk_L - Vk_G. In
    # 'mc' mode Vk_L falls back to the local MC return (local_value_refs[k]) and
    # Vk_G is the global's deterministic-MC return. Retention still uses
    # local_value_refs (MC) -- reported retention is kept separate.
    def compute_shortfall():
        if value_mode == "bootstrap" and local_actor_ref is not None:
            # BOOT_ALPHA (not the live alpha) so both endpoints' entropy scale
            # is identical -> the -alpha*log_pi bias cancels in the shortfall.
            return windowed_shortfall(local_agent, local_actor_ref, actor,
                                      task_ids[k], k, seed,
                                      args.constraint_episodes, window_H, device,
                                      BOOT_ALPHA, n_envs=n_envs,
                                      n_boot_samples=args.n_boot_samples)
        vg, _ = deterministic_eval(actor, task_ids[k], k, seed,
                                   args.constraint_episodes, device, n_envs=n_envs)
        return local_value_refs[k], vg

    # initial (Vk_L, Vk_G) + shortfall so coeff_k is defined before first refresh.
    Vk_L, Vk_G = compute_shortfall()
    # Dual tolerance, RELATIVE to the (initial) local reference value:
    #   eps_eff = (tol_frac * Vk_L)^2   (squared, matching shortfall_k^2).
    # Computed ONCE from the initial Vk_L and held FIXED for the phase so the
    # constraint bar doesn't drift as Vk_L is re-estimated each refresh.
    # --eps-abs (>= 0) overrides with a fixed absolute value if the user opts in.
    if getattr(args, "eps_abs", -1.0) >= 0.0:
        eps_eff = float(args.eps_abs)
    else:
        eps_eff = float((args.tol_frac * Vk_L) ** 2)
    shortfall_k = max(0.0, Vk_L - Vk_G)
    logger.info(f"[global k={k}] Vk_L={Vk_L:.3f} tol_frac={args.tol_frac} "
                f"eps_eff={eps_eff:.5f} value_mode={value_mode} "
                f"(eps_abs override={'yes' if getattr(args, 'eps_abs', -1.0) >= 0.0 else 'no'})")
    logger.info(f"[global k={k}] Vk_L={Vk_L:.3f} initial Vk_G={Vk_G:.3f} "
                f"shortfall={shortfall_k:.3f}")

    log_every = getattr(args, "log_every", 5)
    ckpt_every = getattr(args, "ckpt_every", 0)
    t0 = time.time()
    last = {"actor": float("nan"), "q": float("nan"), "alpha": float(alpha)}
    norm_coeffs = {}

    for iteration in range(1, num_iterations + 1):
        # ---- collect one fresh episode per active task ----
        for i in active:
            total_consolidation_steps += collect_episode(i)

        # ---- two-timescale refresh: Vk_G / shortfall + mu every constraint_every
        if iteration % args.constraint_every == 0:
            Vk_L, Vk_G = compute_shortfall()
            shortfall_k = max(0.0, Vk_L - Vk_G)
            mu = float(np.clip(mu + args.dual_lr * (shortfall_k ** 2 - eps_eff),
                               0.0, args.mu_max))
            logger.info(f"[global k={k}] it={iteration} shortfall={shortfall_k:.3f} "
                        f"Vk_G={Vk_G:.3f} mu={mu:.4f}")
            if clog is not None:
                clog.dual(task_idx=k, it=iteration, mu=mu, lam=None,
                          shortfall_current=shortfall_k, shortfall_past=None,
                          coeff_current=(mu * 2.0 * shortfall_k),
                          grad_share_past=None)

        # ---- actor coefficients over the ACTIVE set (held shortfall_k + mu) ----
        coeffs = {}
        for i in active:
            if i < k:
                coeffs[i] = 1.0 / (k + 1)
            else:
                coeffs[i] = mu * 2.0 * shortfall_k
        Z = sum(coeffs.values())
        if Z <= 0:
            for i in active:
                coeffs[i] = 1.0 / len(active)
            Z = 1.0
        norm_coeffs = {i: coeffs[i] / Z for i in active}

        # ---- SAC gradient updates ----
        # only tasks whose buffer is warm enough to sample
        ready = [i for i in active if rb_by_idx[i].size() > BATCH_SIZE]
        n_updates = args.global_updates_per_iter if ready else 0
        for upd in range(n_updates):
            data_by_idx = {i: rb_by_idx[i].sample(BATCH_SIZE) for i in ready}

            # --- critics: STANDARD per-task SAC (or CrossQ) loss, summed ---
            total_q = 0.0
            for i in ready:
                total_q = total_q + critic_loss_fn(agent, actor, data_by_idx[i], i, alpha, device)
            q_optimizer.zero_grad()
            total_q.backward()
            q_optimizer.step()
            last["q"] = float(total_q.item())

            # --- actor: NORMALIZED weighted sum of per-task SAC actor losses ---
            # Renormalize the (active-set) coeffs over the READY subset so they
            # sum to 1 even when some active task's buffer isn't warm yet (e.g.
            # first iteration): avoids a benign scale shrink in the actor step.
            # If no ready coeff mass (current task k not yet ready), fall back to
            # the unnormalized coeffs -- this self-heals after one episode.
            if upd % POLICY_FREQUENCY == 0:
                ready_Z = sum(norm_coeffs.get(i, 0.0) for i in ready)
                for _ in range(POLICY_FREQUENCY):
                    total_actor = 0.0
                    logpi_cat = []
                    for i in ready:
                        a_loss, log_pi = actor_loss_fn(agent, actor, data_by_idx[i], i, alpha, device)
                        w = norm_coeffs.get(i, 0.0)
                        if ready_Z > 0:
                            w = w / ready_Z
                        total_actor = total_actor + w * a_loss
                        logpi_cat.append(log_pi)
                    actor_optimizer.zero_grad()
                    total_actor.backward()
                    actor_optimizer.step()
                    last["actor"] = float(total_actor.item())
                    if autotune and logpi_cat:
                        with torch.no_grad():
                            log_pi_all = torch.cat(logpi_cat, dim=0)
                        alpha_loss = (-log_alpha.exp() * (log_pi_all + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        last["alpha"] = float(alpha)

            # target sync (per active task) each update; CrossQ has no targets.
            if not crossq:
                for i in ready:
                    agent.sync_targets(i, TAU)

        # ---- stream progress / status / rolling checkpoint ----
        if reporter is not None and (iteration % log_every == 0 or iteration == num_iterations):
            sps = int(total_consolidation_steps / max(1e-9, time.time() - t0))
            frac_done = iteration / num_iterations
            reporter.progress(task=k, task_id=task_ids[k], phase="global",
                              iter=iteration, iters_total=num_iterations,
                              consolidation_steps=total_consolidation_steps, sps=sps,
                              mu=round(mu, 5), shortfall=round(shortfall_k, 4),
                              Vk_L=round(Vk_L, 3), Vk_G=round(Vk_G, 3),
                              tol_frac=args.tol_frac, eps_eff=round(eps_eff, 5),
                              alpha=round(last["alpha"], 5),
                              actor_coeffs={int(task_ids[i]): round(norm_coeffs[i], 4) for i in active},
                              active_tasks=[int(task_ids[i]) for i in active],
                              actor_loss=round(last["actor"], 5), q_loss=round(last["q"], 5))
            reporter.status(task=k, task_id=task_ids[k], phase="global",
                            iter=iteration, iters_total=num_iterations,
                            percent=round(100 * frac_done, 1), sps=sps, mu=round(mu, 5),
                            shortfall=round(shortfall_k, 4), Vk_L=round(Vk_L, 3),
                            Vk_G=round(Vk_G, 3),
                            phase_elapsed_sec=round(time.time() - t0, 1),
                            phase_eta_sec=round((time.time() - t0) * (1 - frac_done) / max(1e-9, frac_done), 1),
                            seen_tasks=[int(task_ids[i]) for i in seen_idx],
                            active_tasks=[int(task_ids[i]) for i in active])
            if clog is not None:
                clog.heartbeat(task_idx=k, phase="global",
                               percent=round(100 * frac_done, 1))
        if reporter is not None and ckpt_every and iteration % ckpt_every == 0:
            reporter.checkpoint(agent, f"global_ckpt_task{k}_latest")

        # ---- retention check -> needy-set refresh + gated early stop ----
        # NOTE: reported retention stays greedy-MC (deterministic_eval via
        # det_scores_all + local_value_refs/local_success_refs); the windowed
        # bootstrap is the TRAINING signal (shortfall Vk_G/Vk_L) ONLY.
        if iteration % args.stop_eval_every == 0:
            sc = det_scores_all()
            mean_ret = float(np.mean([sc[i][1] for i in seen_idx]))
            if mean_ret > best_ret:
                best_ret = mean_ret
                best_state = {kk: v.detach().clone() for kk, v in agent.state_dict().items()}
            scores = {int(task_ids[i]): (round(sc[i][0], 3), round(sc[i][1], 3)) for i in seen_idx}
            # ---- bootstrap-vs-MC diagnostic (CL bias check) ----
            # VkG_MC = greedy-MC value actually reported; VkG_boot/VkL_boot = the
            # in-phase single-critic endpoints (global rollout + local tail, and
            # local-critic V(S_0)). The gap measures how biased the training
            # proxy is against the reported metric. In mc mode the in-phase
            # values ARE the mc values (compute_shortfall returns MC).
            VkG_MC = sc[k][0]
            VkL_MC = local_value_refs[k]
            VkG_boot = Vk_G  # in-phase Vk_G (global rollout + local-critic tail)
            VkL_boot = Vk_L  # in-phase Vk_L (local-critic V(S_0))
            gap = VkG_boot - VkG_MC
            boot_vs_mc = {
                "kind": "boot_vs_mc", "task_idx": k, "iter": iteration,
                "VkG_boot": VkG_boot, "VkG_MC": VkG_MC, "gap": gap,
                "VkL_boot": VkL_boot, "VkL_MC": VkL_MC,
            }
            if clog is not None:
                clog.note(json.dumps(boot_vs_mc))
            all_ok = all(
                (sc[i][1] >= args.retention_frac * local_success_refs[i]) or
                (sc[i][0] >= args.retention_frac * local_value_refs[i])
                for i in seen_idx
            )
            if needy_mode:
                active_past = needy_past(sc)
                active = active_past + [k]
                set_active(active)
                actor_optimizer, q_optimizer = build_optimizers(active)

            past_min = iteration >= args.min_iters
            if past_min:
                consec_ok = consec_ok + 1 if all_ok else 0
            logger.info(f"[global k={k}] it={iteration} retention_all_ok={all_ok} "
                        f"consec={consec_ok} active={[task_ids[i] for i in active]}")
            if reporter is not None:
                reporter.retention(task=k, iter=iteration, all_ok=bool(all_ok),
                                   consec=consec_ok, scores=scores,
                                   active_tasks=[int(task_ids[i]) for i in active],
                                   success_thr={int(task_ids[i]): round(args.retention_frac * local_success_refs[i], 3) for i in seen_idx},
                                   value_thr={int(task_ids[i]): round(args.retention_frac * local_value_refs[i], 3) for i in seen_idx},
                                   boot_vs_mc=boot_vs_mc)
            if past_min and consec_ok >= args.patience:
                logger.info(f"[global k={k}] retention-gated early stop at it={iteration}")
                break

    # best-checkpoint restore: keep the peak mean-success model over this phase.
    if best_state is not None:
        final_sc = det_scores_all()
        final_ret = float(np.mean([final_sc[i][1] for i in seen_idx]))
        if best_ret > final_ret:
            agent.load_state_dict(best_state)
            logger.info(f"[global k={k}] best-ckpt restored mean_success={best_ret:.3f} "
                        f"(final was {final_ret:.3f})")
    for i in list(envs_by_idx.keys()):
        close_task(i)
    if reporter is not None:
        reporter.phase_summary(task=k, phase="global", iters_run=iteration,
                               consolidate_mode=("needy" if needy_mode else "all"),
                               consolidation_steps=total_consolidation_steps,
                               final_active_tasks=[int(task_ids[i]) for i in active],
                               mu_final=round(mu, 5), Vk_L=round(Vk_L, 3),
                               tol_frac=args.tol_frac, eps_eff=round(eps_eff, 5),
                               Vk_G_last=round(Vk_G, 3), shortfall_last=round(shortfall_k, 4))
    return total_consolidation_steps


def main():
    args = parse_args()
    if not args.debug:
        import sys
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    if args.tasks.strip():
        task_ids = [int(t) for t in args.tasks.split(",")]
    else:
        task_ids = list(range(args.num_tasks))

    if getattr(args, "torch_threads", 0) and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda) else "cpu")
    logger.info(f"device={device} task_ids={task_ids} tasks={[CW20_TASKS[t] for t in task_ids]} "
                f"tag={args.tag} consolidate={args.consolidate_mode}")

    # probe env for obs/act dims
    probe = build_env(task_ids[0], args.seed)
    obs_dim = int(np.array(probe.single_observation_space.shape).prod())
    act_dim = int(np.prod(probe.single_action_space.shape))
    probe.close()

    agent = OursAgent(obs_dim=obs_dim, act_dim=act_dim, num_tasks=1,
                      crossq=args.crossq).to(device)

    local_value_refs = {}
    local_success_refs = {}
    frames_current = 0
    frames_consolidation = 0

    agents_root = f"./agents/{args.tag}"
    reporter = Reporter(args.tag, agents_root, task_ids, args)
    state_path = os.path.join(reporter.run_dir, "run_state.json")

    # ---- ContractLogger (ADDITIVE; separate dir so no collision with Reporter) ----
    contract_dir = f"./data/{args.tag}/Ours/contract"
    clog = ContractLogger(
        contract_dir,
        method="constrained",
        seed=args.seed,
        env_family="metaworld",
        tasks=[CW20_TASKS[t] for t in task_ids],
        task_order=",".join(str(t) for t in task_ids),
        # reported metric is SUCCESS RATE in [0,1] -> normalization is identity.
        reference={"random": [0.0] * len(task_ids), "ceiling": [1.0] * len(task_ids)},
        config={"n_envs": args.n_envs, "value_mode": args.value_mode,
                "window_H": args.window_H, "crossq": args.crossq,
                "task1_steps": args.task1_steps, "local_steps": args.local_steps,
                "global_iters": args.global_iters,
                "constraint_every": args.constraint_every,
                "tol_frac": args.tol_frac, "dual_lr": args.dual_lr,
                "retention_frac": args.retention_frac,
                "consolidate_mode": args.consolidate_mode},
        frames_per_iter=0,  # frames_total is set manually per phase.
    )
    clog.note(f"meta-world ours; value_mode={args.value_mode}, "
              f"crossq={args.crossq}, n_envs={args.n_envs}, window_H={args.window_H}")

    # eval_matrix[k] = [greedy success on task 0..k] after task k's global phase.
    eval_matrix = []
    eval_matrix_path = f"./data/{args.tag}/Ours/contract/eval_matrix.json"

    def set_frames_total():
        clog.frames_total = frames_current + frames_consolidation

    def end_of_phase_eval(k):
        """Greedy-eval EVERY seen task, emit end-of-phase eval records, append the
        row to eval_matrix, and write eval_matrix.json (crash-safe)."""
        row = []
        for i in range(k + 1):
            tid = task_ids[i]
            succ = deterministic_eval(
                ActorHelper(agent, *_action_bounds(tid, args.seed), device),
                tid, i, args.seed, args.eval_episodes, device, n_envs=args.n_envs)[1]
            row.append(succ)
            clog.eval(task_idx=k, phase="eval", it=-1, evaluated_on=i,
                      evaluated_on_task=CW20_TASKS[tid], raw=succ,
                      episodes=args.eval_episodes, greedy=True, seen=True)
        eval_matrix.append(row)
        with open(eval_matrix_path, "w") as f:
            json.dump(eval_matrix, f, indent=2)
        return row

    def save_state(last_completed):
        st = {"last_completed_task": last_completed,
              "task_ids": task_ids, "seed": args.seed,
              "local_value_refs": {int(i): local_value_refs[i] for i in local_value_refs},
              "local_success_refs": {int(i): local_success_refs[i] for i in local_success_refs},
              "frames_current": frames_current,
              "frames_consolidation": frames_consolidation}
        tmp = state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f, indent=2); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, state_path)
        with open(os.path.join(reporter.run_dir, "local_refs.json"), "w") as f:
            json.dump({int(i): {"value": local_value_refs[i],
                                "success": local_success_refs[i]}
                       for i in local_value_refs}, f, indent=2)

    start_k = 0
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            st = json.load(f)
        done = st["last_completed_task"]
        agent = OursAgent.load(f"{agents_root}/global_after_task{done}", map_location=device).to(device)
        local_value_refs.update({int(i): v for i, v in st["local_value_refs"].items()})
        local_success_refs.update({int(i): v for i, v in st["local_success_refs"].items()})
        frames_current = st["frames_current"]
        frames_consolidation = st["frames_consolidation"]
        start_k = done + 1
        logger.info(f"[resume] loaded global_after_task{done}; continuing from task {start_k}")
        # reload eval_matrix so end-of-phase rows aren't lost on resume.
        if os.path.exists(eval_matrix_path):
            with open(eval_matrix_path) as f:
                eval_matrix.extend(json.load(f))

    for k, task_id in enumerate(task_ids):
        if k < start_k:
            continue
        agent.ensure_head(k)

        if k == 0:
            logger.info(f"=== Task 0 (CW20 {task_id}={CW20_TASKS[task_id]}): standard SAC -> initial global ===")
            reporter.status(task=0, task_id=task_id, phase="task0", percent=0.0)
            clog.heartbeat(task_idx=0, phase="task1", percent=0.0)
            clog.phase_start(task_idx=0, task=CW20_TASKS[task_id], phase="task1")
            t_ph = time.time()
            writer = make_tb_writer(args.tag, 0, args.seed, consolidation=False)
            steps = train_sac_single_task(
                agent, task_id, 0, args.seed, args.task1_steps, device, writer,
                reporter=reporter, phase="task0", task_k=0, save_name="global_task0",
                clog=clog, clog_task_idx=0, clog_phase="task1")
            writer.close()
            frames_current += steps
            set_frames_total()
            clog.phase_end(task_idx=0, task=CW20_TASKS[task_id], phase="task1",
                           iters=steps, frames_phase=steps,
                           wall_s_phase=round(time.time() - t_ph, 1))
            # local reference: deterministic MC value + success of the global
            v0, s0 = deterministic_eval(
                ActorHelper(agent, *_action_bounds(task_id, args.seed), device),
                task_id, 0, args.seed, args.eval_episodes, device, n_envs=args.n_envs)
            local_value_refs[0] = v0
            local_success_refs[0] = s0
            logger.info(f"[task0] {CW20_TASKS[task_id]} local ref V={v0:.3f} success={s0:.3f}")
            reporter.phase_summary(task=0, phase="task0", steps=steps,
                                   local_value=round(v0, 3), local_success=round(s0, 3),
                                   wall_sec=round(time.time() - t_ph, 1))
            agent.save(f"{agents_root}/global_after_task0")
            save_state(0)
            end_of_phase_eval(0)  # end-of-phase greedy eval + eval_matrix row
            continue

        # ---------------- Task k>=1: LOCAL phase ----------------
        logger.info(f"=== Task {k} (CW20 {task_id}={CW20_TASKS[task_id]}): LOCAL phase (unconstrained SAC) ===")
        reporter.status(task=k, task_id=task_id, phase="local", percent=0.0)
        clog.heartbeat(task_idx=k, phase="local", percent=0.0)
        clog.phase_start(task_idx=k, task=CW20_TASKS[task_id], phase="local")
        t_ph = time.time()
        local_agent = agent.clone().to(device)
        local_agent.ensure_head(k)
        writer = make_tb_writer(args.tag, k, args.seed, consolidation=False)
        steps_local = train_sac_single_task(
            local_agent, task_id, k, args.seed, args.local_steps, device, writer,
            reporter=reporter, phase="local", task_k=k, save_name=f"local_task{k}",
            clog=clog, clog_task_idx=k, clog_phase="local")
        writer.close()
        frames_current += steps_local
        set_frames_total()
        clog.phase_end(task_idx=k, task=CW20_TASKS[task_id], phase="local",
                       iters=steps_local, frames_phase=steps_local,
                       wall_s_phase=round(time.time() - t_ph, 1))
        for p in local_agent.parameters():
            p.requires_grad_(False)
        local_agent.eval()
        local_actor = ActorHelper(local_agent, *_action_bounds(task_id, args.seed), device)
        vk, sk = deterministic_eval(local_actor, task_id, k, args.seed,
                                    args.eval_episodes, device, n_envs=args.n_envs)
        local_value_refs[k] = vk
        local_success_refs[k] = sk
        # Bootstrap-mode Vk_L is NO LONGER precomputed here: it is the local
        # critic's soft V(S_0), evaluated in-phase from the frozen local_agent
        # (train_global_phase / windowed_shortfall) so BOTH shortfall endpoints
        # share the same frozen local critic. No local rollout needed at freeze.
        logger.info(f"[task{k}] {CW20_TASKS[task_id]} LOCAL ref V={vk:.3f} success={sk:.3f}")
        local_agent.save(f"{agents_root}/local_after_task{k}")
        reporter.phase_summary(task=k, phase="local", steps=steps_local,
                               local_value=round(vk, 3), local_success=round(sk, 3),
                               wall_sec=round(time.time() - t_ph, 1))

        # ---------------- Task k>=1: GLOBAL consolidation phase ----------------
        logger.info(f"=== Task {k} (CW20 {task_id}={CW20_TASKS[task_id]}): GLOBAL consolidation ===")
        clog.phase_start(task_idx=k, task=CW20_TASKS[task_id], phase="global")
        t_gph = time.time()
        seen_idx = list(range(k + 1))
        cons_writers = {i: make_tb_writer(args.tag, i, args.seed, consolidation=True)
                        for i in seen_idx}
        cons_steps = train_global_phase(
            agent, local_agent, task_ids, seen_idx, k, args.seed, args, device,
            local_value_refs, local_success_refs, cons_writers, reporter=reporter,
            clog=clog)
        for w in cons_writers.values():
            w.close()
        frames_consolidation += cons_steps
        set_frames_total()
        clog.phase_end(task_idx=k, task=CW20_TASKS[task_id], phase="global",
                       iters=args.global_iters, frames_phase=cons_steps,
                       wall_s_phase=round(time.time() - t_gph, 1))
        agent.save(f"{agents_root}/global_after_task{k}")
        save_state(k)
        end_of_phase_eval(k)  # end-of-phase greedy eval + eval_matrix row

    # final global
    agent.save(f"{agents_root}/final_global")
    clog.close()
    close_eval_envs()  # tear down the persistent eval-env cache

    total = frames_current + frames_consolidation
    accounting = {
        "task_ids": task_ids,
        "tasks": [CW20_TASKS[t] for t in task_ids],
        "frames_current_task_learning": frames_current,
        "frames_past_task_consolidation": frames_consolidation,
        "total_env_steps": total,
        "note": ("current = task0 SAC + per-task LOCAL SAC on the new task; "
                 "consolidation = global-phase rollouts over ACTIVE (all or needy) "
                 "seen tasks (the disclosed past-task env-access premise)."),
    }
    acc_dir = f"./data/{args.tag}/Ours"
    os.makedirs(acc_dir, exist_ok=True)
    with open(f"{acc_dir}/frame_accounting.json", "w") as f:
        json.dump(accounting, f, indent=2)
    logger.info(f"FRAME ACCOUNTING: current={frames_current} "
                f"consolidation={frames_consolidation} total={total}")
    print(json.dumps(accounting, indent=2))


if __name__ == "__main__":
    main()
