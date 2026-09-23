import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from .cnn_encoder import CnnEncoder


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def _make_actor_head(n_actions):
    return nn.Sequential(
        layer_init(nn.Linear(512, 512)),
        nn.ReLU(),
        layer_init(nn.Linear(512, n_actions), std=0.01),
    )


def _make_critic_head():
    return layer_init(nn.Linear(512, 1), std=1)


class OursAgent(nn.Module):
    """Constrained min-max two-policy agent (PPO backend).

    A single shared Nature-CNN encoder feeds per-task actor heads and per-task
    critic heads. Each task index selects its own actor/critic head; the encoder
    is shared across all tasks. This is used for both the accumulating GLOBAL
    policy and the specialized (cloned) LOCAL policy.
    """

    def __init__(self, envs, num_tasks=1, shared_head=False):
        super().__init__()
        self.n_actions = envs.single_action_space.n
        self.shared_head = shared_head
        self.network = CnnEncoder(hidden_dim=512, layer_init=layer_init)
        # shared_head=True => ONE actor + ONE critic head for ALL tasks (matches the
        # authors' single-head cnn_simple/cka_rl design). Every task_idx routes to
        # head 0, so the shared representation must retain all modes (the hard,
        # apples-to-apples setting). shared_head=False keeps the per-task-head variant.
        n_heads = 1 if shared_head else num_tasks
        self.actors = nn.ModuleList([_make_actor_head(self.n_actions) for _ in range(n_heads)])
        self.critics = nn.ModuleList([_make_critic_head() for _ in range(n_heads)])

    def _h(self, task_idx):
        """Head index: always 0 in shared-head mode, else the task's own head."""
        return 0 if getattr(self, "shared_head", False) else task_idx

    def ensure_head(self, task_idx):
        """Grow the actor/critic head lists so that `task_idx` is valid.
        No-op in shared-head mode (a single head serves every task)."""
        if getattr(self, "shared_head", False):
            return
        while len(self.actors) <= task_idx:
            self.actors.append(_make_actor_head(self.n_actions))
            self.critics.append(_make_critic_head())
        # keep new heads on the same device as the encoder
        device = next(self.network.parameters()).device
        self.actors.to(device)
        self.critics.to(device)

    def get_value(self, x, task_idx):
        return self.critics[self._h(task_idx)](self.network(x))

    def get_action_and_value(self, x, task_idx, action=None):
        hidden = self.network(x)
        logits = self.actors[self._h(task_idx)](hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critics[self._h(task_idx)](hidden)

    def greedy_action(self, x, task_idx):
        with torch.no_grad():
            logits = self.actors[self._h(task_idx)](self.network(x))
            return torch.argmax(logits, dim=-1)

    def clone(self):
        """Deep copy (encoder + all heads) for creating a fresh LOCAL policy."""
        return copy.deepcopy(self)

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.network, f"{dirname}/encoder.pt")
        torch.save(self.actors, f"{dirname}/actors.pt")
        torch.save(self.critics, f"{dirname}/critics.pt")
        with open(f"{dirname}/meta.txt", "w") as f:
            f.write(f"num_tasks={len(self.actors)}\nn_actions={self.n_actions}\n"
                    f"shared_head={int(getattr(self, 'shared_head', False))}\n")

    @staticmethod
    def load(dirname, envs=None, map_location=None):
        # Reconstruct WITHOUT needing `envs`: the saved modules already define the
        # architecture; n_actions comes from meta.txt (fallback: last actor layer).
        model = OursAgent.__new__(OursAgent)
        nn.Module.__init__(model)
        model.network = torch.load(f"{dirname}/encoder.pt", map_location=map_location)
        model.actors = torch.load(f"{dirname}/actors.pt", map_location=map_location)
        model.critics = torch.load(f"{dirname}/critics.pt", map_location=map_location)
        model.n_actions = None
        model.shared_head = (len(model.actors) == 1)  # fallback: 1 head => shared
        try:
            with open(f"{dirname}/meta.txt") as f:
                for line in f:
                    if line.startswith("n_actions="):
                        model.n_actions = int(line.strip().split("=")[1])
                    if line.startswith("shared_head="):
                        model.shared_head = bool(int(line.strip().split("=")[1]))
        except Exception:
            pass
        if model.n_actions is None:
            model.n_actions = model.actors[0][-1].out_features
        return model
