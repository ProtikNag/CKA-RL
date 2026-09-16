"""Logging-contract emitter (docs/LOGGING_CONTRACT.md).

One schema for every method so a single metrics module (analysis.contract_metrics)
computes PERF / FWT / BWT / forgetting / retention / compute for all of them. This
is ADDITIVE: the legacy RunLogger (logs.jsonl) keeps working alongside it. A field a
method cannot fill is written ``null`` rather than invented.

Writes into ``results/<run>/``:
  * run.json        -- once, at start (identity + reference for normalization).
  * progress.jsonl  -- append-only, flushed every write; typed records.
  * status.json     -- overwritten each heartbeat (the "is it alive" file).
  * checkpoints/    -- after_task{k}.pt, local_after_task{k}.pt, optim_after_task{k}.pt.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "n/a"


class ContractLogger:
    """Append-only contract emitter bound to one run directory."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        method: str,
        seed: int,
        env_family: str,
        tasks: list[str],
        task_order: str,
        reference: dict[str, list[float]],
        config: dict[str, Any],
        frames_per_iter: int,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._t0 = time.time()
        self.frames_total = 0
        self.frames_per_iter = int(frames_per_iter)
        run_json = self.run_dir / "run.json"
        if run_json.exists():
            # Resume: run.json is written once and never mutated; keep its reference.
            existing = json.loads(run_json.read_text())
            ref = existing.get("reference", {})
            self._random = list(ref.get("random", []))
            self._ceiling = list(ref.get("ceiling", []))
        else:
            self._random = list(reference.get("random", []))
            self._ceiling = list(reference.get("ceiling", []))
            run_meta = {
                "run_name": self.run_dir.name,
                "method": method,
                "seed": seed,
                "git_sha": _git_sha(),
                "started": datetime.now(timezone.utc).isoformat(),
                "env_family": env_family,
                "tasks": tasks,
                "task_order": task_order,
                "reference": {"random": self._random, "ceiling": self._ceiling},
                "config": config,
            }
            run_json.write_text(json.dumps(run_meta, indent=2))
        self._prog = open(self.run_dir / "progress.jsonl", "a")

    # ------------------------------------------------------------------ #
    def _t(self) -> float:
        return round(time.time() - self._t0, 3)

    def _emit(self, rec: dict[str, Any]) -> None:
        rec = {"t_wall": self._t(), "frames_total": int(self.frames_total), **rec}
        self._prog.write(json.dumps(rec) + "\n")
        self._prog.flush()

    def normalized(self, raw: float, j: int) -> float:
        """(raw - random[j]) / (ceiling[j] - random[j]); divide-by-zero safe."""
        rnd = self._random[j] if j < len(self._random) else 0.0
        cel = self._ceiling[j] if j < len(self._ceiling) else 1.0
        denom = (cel - rnd) if (math.isfinite(cel) and cel != rnd) else 1.0
        return (raw - rnd) / denom

    # -- record types (docs/LOGGING_CONTRACT.md §3) --------------------- #
    def phase_start(self, task_idx: int, task: str, phase: str) -> None:
        self._emit({"type": "phase_start", "task_idx": task_idx, "task": task,
                    "phase": phase})

    def phase_end(self, task_idx: int, task: str, phase: str, iters: int,
                  frames_phase: int, wall_s_phase: float) -> None:
        self._emit({"type": "phase_end", "task_idx": task_idx, "task": task,
                    "phase": phase, "iters": iters, "frames_phase": int(frames_phase),
                    "wall_s_phase": wall_s_phase})

    def eval(self, task_idx: int, phase: str, it: int, evaluated_on: int,
             evaluated_on_task: str, raw: float, episodes: int, greedy: bool,
             seen: bool) -> None:
        self._emit({"type": "eval", "task_idx": task_idx, "phase": phase, "iter": it,
                    "evaluated_on": evaluated_on, "evaluated_on_task": evaluated_on_task,
                    "raw": raw, "normalized": self.normalized(raw, evaluated_on),
                    "episodes": episodes, "greedy": greedy, "seen": seen})

    def dual(self, task_idx: int, it: int, mu: float, lam, shortfall_current: float,
             shortfall_past, coeff_current: float, grad_share_past) -> None:
        self._emit({"type": "dual", "task_idx": task_idx, "iter": it, "mu": mu,
                    "lambda": lam, "shortfall_current": shortfall_current,
                    "shortfall_past": shortfall_past, "coeff_current": coeff_current,
                    "grad_share_past": grad_share_past})

    def note(self, text: str) -> None:
        self._emit({"type": "note", "text": text})

    def heartbeat(self, **status: Any) -> None:
        status["alive"] = datetime.now(timezone.utc).isoformat()
        status["t_wall"] = self._t()
        status["frames_total"] = int(self.frames_total)
        (self.run_dir / "status.json").write_text(json.dumps(status, indent=2))

    def save_checkpoint(self, k: int, global_sd, local_sd=None, optim_state=None) -> None:
        """Per-task checkpoint. ``optim_state`` carries the dual + cumulative-step
        state so :meth:`resume` restores the method exactly, not just the weights."""
        ck = self.run_dir / "checkpoints"
        torch.save(global_sd, ck / f"after_task{k}.pt")
        if local_sd is not None:
            torch.save(local_sd, ck / f"local_after_task{k}.pt")
        if optim_state is not None:
            torch.save(optim_state, ck / f"optim_after_task{k}.pt")

    def close(self) -> None:
        try:
            self._prog.close()
        except Exception:
            pass
