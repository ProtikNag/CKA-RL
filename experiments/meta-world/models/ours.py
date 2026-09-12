"""Constrained min-max two-policy continual-RL agent (SAC backend).

This is the Meta-World / SAC analogue of the verified PPO port in
``experiments/atari/models/ours.py``. A single shared encoder feeds PER-TASK
actor heads (a mean head + a log_std head per task index) and PER-TASK SAC
critics (two SoftQNetworks + their targets per task). The task index selects
its own heads; the encoder is shared across all tasks. The same module class is
used for both the accumulating GLOBAL policy and the cloned/frozen LOCAL
specialist references.

Design mapping (PPO -> SAC), documented so it can be reviewed / verified:

  * PPO's shared Nature-CNN encoder  -> shared_arch.shared(obs_dim) MLP encoder
    (2x256 ReLU). The actor encoder is shared across tasks.
  * PPO's per-task Categorical actor head -> per-task (mean Linear, log_std
    Linear) pair, exactly the two heads SimpleAgent.reset_heads() builds, so the
    existing run_sac.py Actor wrapper (tanh-squash + rescale + log_prob
    correction) works unchanged. forward(x, task_idx) -> (mean, log_std).
  * PPO's per-task critic head (a single V(s) Linear) -> per-task SAC twin Q
    critics qf1/qf2 with their targets, each a standard run_sac.py SoftQNetwork
    over [obs, act]. SAC is off-policy and needs Q-critics rather than a V head;
    the CRITIC IS NOT CONSTRAINED in the min-max objective (mirrors the PPO port
    where only the actor carries the CL coefficients). Here the per-task critics
    are trained with the STANDARD SAC critic loss.

The encoder is a property of the ACTOR side only (it sits inside the per-task
actor heads' input path). SAC critics in CleanRL own their own encoder
(SoftQNetwork.fc), so each per-task critic here is fully standard and
self-contained, exactly as in run_sac.py. The actor's shared MLP encoder is the
object that carries cross-task interference / the shared representation that
makes consolidation meaningful.
"""
import os
import copy

import numpy as np
import torch
import torch.nn as nn

from .shared_arch import shared


class SoftQNetwork(nn.Module):
    """Standard SAC twin-Q network (verbatim structure from run_sac.py).

    Kept here (rather than imported from run_sac.py) so models/ours.py has no
    dependency on the training script. Input is [obs, act]; output is a scalar
    Q value. Each task owns two of these (qf1, qf2) plus frozen targets.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = shared(int(obs_dim) + int(act_dim))
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


class OursActor(nn.Module):
    """Shared encoder + per-task (mean, log_std) heads.

    forward(x, task_idx) -> (mean, log_std). The shared encoder is updated on
    every task (cross-task interference lives here); head ``task_idx`` is the
    task-specific readout. This is the object the run_sac.py ``Actor`` wrapper
    wraps (so it must expose forward(x, task_idx=...) returning (mean, log_std)).
    """

    def __init__(self, obs_dim, act_dim, num_tasks=1):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.fc = shared(input_dim=self.obs_dim)
        self.fc_mean = nn.ModuleList(
            [nn.Linear(256, self.act_dim) for _ in range(num_tasks)]
        )
        self.fc_logstd = nn.ModuleList(
            [nn.Linear(256, self.act_dim) for _ in range(num_tasks)]
        )

    def ensure_head(self, task_idx):
        while len(self.fc_mean) <= task_idx:
            self.fc_mean.append(nn.Linear(256, self.act_dim))
            self.fc_logstd.append(nn.Linear(256, self.act_dim))
        device = next(self.fc.parameters()).device
        self.fc_mean.to(device)
        self.fc_logstd.to(device)

    def forward(self, x, task_idx=0):
        h = self.fc(x)
        mean = self.fc_mean[task_idx](h)
        log_std = self.fc_logstd[task_idx](h)
        return mean, log_std


class OursAgent(nn.Module):
    """Continual min-max multi-head SAC agent.

    Holds:
      * self.actor : OursActor (shared encoder + per-task mean/log_std heads).
      * self.qf1 / self.qf2 : per-task twin Q critics (nn.ModuleList).
      * self.qf1_target / self.qf2_target : per-task frozen targets.

    The actor heads are what the run_sac.py Actor wraps. Each task's SAC update
    uses that task's critics; the actor CL objective (normalized weighted sum of
    per-task SAC actor losses) is assembled in run_sac_ours.py, not here.
    """

    def __init__(self, obs_dim, act_dim, num_tasks=1):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.actor = OursActor(self.obs_dim, self.act_dim, num_tasks=num_tasks)
        self.qf1 = nn.ModuleList(
            [SoftQNetwork(self.obs_dim, self.act_dim) for _ in range(num_tasks)]
        )
        self.qf2 = nn.ModuleList(
            [SoftQNetwork(self.obs_dim, self.act_dim) for _ in range(num_tasks)]
        )
        self.qf1_target = nn.ModuleList(
            [SoftQNetwork(self.obs_dim, self.act_dim) for _ in range(num_tasks)]
        )
        self.qf2_target = nn.ModuleList(
            [SoftQNetwork(self.obs_dim, self.act_dim) for _ in range(num_tasks)]
        )
        # initialise targets == online for every initial head
        for i in range(num_tasks):
            self.qf1_target[i].load_state_dict(self.qf1[i].state_dict())
            self.qf2_target[i].load_state_dict(self.qf2[i].state_dict())

    # ------------------------------------------------------------------ heads
    def ensure_head(self, task_idx):
        """Grow actor heads + critic lists so ``task_idx`` is valid."""
        self.actor.ensure_head(task_idx)
        while len(self.qf1) <= task_idx:
            self.qf1.append(SoftQNetwork(self.obs_dim, self.act_dim))
            self.qf2.append(SoftQNetwork(self.obs_dim, self.act_dim))
            self.qf1_target.append(SoftQNetwork(self.obs_dim, self.act_dim))
            self.qf2_target.append(SoftQNetwork(self.obs_dim, self.act_dim))
            self.qf1_target[-1].load_state_dict(self.qf1[-1].state_dict())
            self.qf2_target[-1].load_state_dict(self.qf2[-1].state_dict())
        device = next(self.actor.fc.parameters()).device
        self.qf1.to(device)
        self.qf2.to(device)
        self.qf1_target.to(device)
        self.qf2_target.to(device)

    @property
    def num_tasks(self):
        return len(self.actor.fc_mean)

    # -------------------------------------------------------------- forward(s)
    def forward(self, x, task_idx=0):
        """(mean, log_std) for head ``task_idx`` (pre-Actor-wrapper squash)."""
        return self.actor(x, task_idx=task_idx)

    def sync_targets(self, task_idx, tau):
        for p, tp in zip(self.qf1[task_idx].parameters(),
                         self.qf1_target[task_idx].parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for p, tp in zip(self.qf2[task_idx].parameters(),
                         self.qf2_target[task_idx].parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # --------------------------------------------------------------- clone/io
    def clone(self):
        """Deep copy (encoder + all heads + critics) -> fresh LOCAL policy."""
        return copy.deepcopy(self)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.actor, f"{dirname}/actor.pt")
        torch.save(self.qf1, f"{dirname}/qf1.pt")
        torch.save(self.qf2, f"{dirname}/qf2.pt")
        torch.save(self.qf1_target, f"{dirname}/qf1_target.pt")
        torch.save(self.qf2_target, f"{dirname}/qf2_target.pt")
        with open(f"{dirname}/meta.txt", "w") as f:
            f.write(
                f"num_tasks={self.num_tasks}\n"
                f"obs_dim={self.obs_dim}\n"
                f"act_dim={self.act_dim}\n"
            )

    @staticmethod
    def load(dirname, map_location=None):
        model = OursAgent.__new__(OursAgent)
        nn.Module.__init__(model)
        model.actor = torch.load(f"{dirname}/actor.pt", map_location=map_location)
        model.qf1 = torch.load(f"{dirname}/qf1.pt", map_location=map_location)
        model.qf2 = torch.load(f"{dirname}/qf2.pt", map_location=map_location)
        model.qf1_target = torch.load(f"{dirname}/qf1_target.pt", map_location=map_location)
        model.qf2_target = torch.load(f"{dirname}/qf2_target.pt", map_location=map_location)
        model.obs_dim = model.actor.obs_dim
        model.act_dim = model.actor.act_dim
        try:
            with open(f"{dirname}/meta.txt") as f:
                for line in f:
                    if line.startswith("obs_dim="):
                        model.obs_dim = int(line.strip().split("=")[1])
                    elif line.startswith("act_dim="):
                        model.act_dim = int(line.strip().split("=")[1])
        except Exception:
            pass
        return model
