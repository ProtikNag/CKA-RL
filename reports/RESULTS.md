# Ours (min-max) vs CKA-RL (NeurIPS'25) — results (seed 0)

Status 2026-09-14. **2 of 3 benchmarks done** (Freeway, SpaceInvaders); Meta-World
running. Single seed, PROVISIONAL. This file is the source of truth for the
visualization session.

> **New session: orient with the `graphify` plugin first** (`graphify query "..."`,
> `graphify-out/graph.json` exists) before reading source. Build figures via the
> `visualization` agent / visualization-expert gate; keep the caveats in captions.

## 2026-09-16 — Meta-World speedup + rerun (SUPERSEDES the "Meta running" rows below)

The original Meta-World run (job 21910655) was too slow (66h → task 10/20). Replaced by
a **speedup stack in `experiments/meta-world/run_sac_ours.py` + `models/ours.py`** (this
commit), user-approved, code-verifier PASS, CL-expert reviewed:
- **CrossQ critic** (`--crossq`, BN critics, no target net, joint current+next forward) —
  the real sample-efficiency lever.
- **Windowed critic value-gap** (`--value-mode bootstrap --window-H 200`): shortfall uses
  the **frozen LOCAL critic for BOTH V(S_0) and V(S_H)**; only the H-step reward is rolled
  by the current global policy: `A_k = Σ_{t<H} γ^t r_t + γ^H V_L(S_H) − V_L(S_0)`. Replaces
  the full-episode MC constraint eval. 4-sample variance reduction.
- **Parallel envs** (`--n-envs`, UTD-preserving) — but the demo showed **no GPU speedup**
  (transfer/overhead-bound), so the real run uses n_envs=1.
- **Contract logging** (`contract_logging.py`, copied from CRL-Minimax) → `data/<tag>/Ours/
  contract/` with run.json/progress.jsonl/eval_matrix.json, so `analysis/contract_metrics.py`
  computes PERF/FWT/BWT identically to GridWorld/Atari. Reported metric = success rate.

**CL-expert flag (advisory, surfaced):** the windowed bootstrap is a *biased proxy* for the
greedy-MC value it claims to bound (truncation + critic error). Instrumented with a live
**boot-vs-MC diagnostic** (`kind:boot_vs_mc` notes). Retention/reported metrics stay greedy-MC.

**Status:** 150k/task run FAILED (hard tasks hammer/push-wall don't learn at 150k → untrained
critics → broken bootstrap). Now: **full 20-task @300k bootstrap (job 21921681, L40S)** running
speculatively + a 3-way parallel validation (hammer@300k, pushwall@300k, faucet→window
boot-vs-MC, ~3.5h). Decision rule: tasks learn + boot tracks → keep it; else switch
`--value-mode mc` or bump budget. See memory `metaworld-speedup-design` for the full log.

## Table 1 (PERF = plasticity / end-of-task success; FWT = forward transfer)
| Env | Ours PERF | CKA-RL | CompoNet | FT-N | Ours FWT | CKA-RL FWT |
|---|--:|--:|--:|--:|--:|--:|
| Freeway | 0.753 | 0.792 | 0.763 | 0.753 | 0.661 | 0.743 |
| SpaceInvaders | 0.981 | 0.993 | 0.983 | 0.979 | 0.650 | 0.775 |
| Meta-World | — running — | 0.464 | 0.413 | 0.377 | — | -0.003 |

## Table 3 style — final-policy retention (Ours, 100-ep)
| Env | success rate | retention vs specialist |
|---|--:|--:|
| Freeway | 0.75 (6/8) | ~82% |
| SpaceInvaders | 1.00 (10/10) | ~90% |
| Meta-World | — running — | — |
Paper Table-3 (final-policy avg over all 3 envs): CKA-RL 0.3966 — need our Meta row
to form the comparable 3-env average.

## Read (honest)
- PERF: Ours competitive, slightly below CKA-RL on both. FWT: mid-pack.
- Retention: strong (SI full 1.0; Freeway 6/8) — **enabled by ours' disclosed live
  past-task env access** (we re-simulate + re-consolidate old modes); baselines have
  none. Not a clean "beats the paper" claim. Single seed.
- SI's mid-run 64% snapshot was transient/noisy (in-consolidation, 3-ep); the final
  model recovered all modes (success 1.0).

## Where the data is (gitignored, regenerate on the cluster)
- Runs: `experiments/{atari,meta-world}/data/<tag>/Ours/` — `table3_final_policy.json`,
  `returns.csv`, `retention_history.jsonl`, `phase_summaries.jsonl`, `status.json`.
  Tags: Freeway `fw_s0`, SpaceInvaders `si_s0`, Meta `meta_s0`. Agents:
  `experiments/*/agents/.../final_global`.
- Metric pipeline: Atari `gather_rt_results.py` + `process_results.py` (PERF/FWT) +
  `eval_final_policy.py` (100-ep retention, needs a GPU — dgx free). Meta:
  `extract_results.py` + `process_results.py` + `eval_final_policy_metaworld.py`.
- Launchers: `scripts/hpc_ours_{atari,meta}.sbatch`, `scripts/hpc_baseline_{atari,meta}.sbatch`.

## TODO for the viz session
1. SpaceInvaders figure — mirror `reports/freeway/make_fig.py`.
2. Combined 3-benchmark figure (once Meta done): PERF / FWT / retention, Ours vs paper.
3. Table-3 3-env average vs CKA-RL 0.3966.
4. Refresh this file + push (main → origin `feature/updated-objective`; here → `fork`
   = ProtikNag/CKA-RL, branch `ours-minmax-row`).
