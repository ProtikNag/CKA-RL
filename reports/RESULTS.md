# Ours (min-max) vs CKA-RL (NeurIPS'25) — results (seed 0)

Status 2026-09-14. **2 of 3 benchmarks done** (Freeway, SpaceInvaders); Meta-World
running. Single seed, PROVISIONAL. This file is the source of truth for the
visualization session.

> **New session: orient with the `graphify` plugin first** (`graphify query "..."`,
> `graphify-out/graph.json` exists) before reading source. Build figures via the
> `visualization` agent / visualization-expert gate; keep the caveats in captions.

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
