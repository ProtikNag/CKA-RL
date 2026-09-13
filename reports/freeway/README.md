# Freeway — Ours (min-max) vs CKA-RL (NeurIPS'25), seed 0 (PROVISIONAL)

First completed benchmark of our min-max method dropped into the CKA-RL harness.
**Single seed, provisional** — SpaceInvaders and Meta-World still running.

Figure: `png/freeway_ours_vs_paper.png` (`svg/` too), from `make_fig.py`.
visualization-expert: faithfulness-checked.

## Table 1 (Freeway column)
| Method | PERF | FWT |
|---|---:|---:|
| CKA-RL (paper) | 0.792 | 0.743 |
| CReLUs | 0.784 | 0.730 |
| CbpNet | 0.768 | 0.720 |
| CompoNet | 0.763 | 0.712 |
| FT-N | 0.753 | 0.694 |
| **Ours** | **0.753** | **0.661** |
| Baseline | 0.125 | 0.000 |

Ours' PERF (plasticity, = local-phase curve) is **mid-pack, tied with FT-N, below
CKA-RL** — competitive, not better. FWT a bit below the top methods.
(Ours FWT is measured against *our* Baseline run.)

## Retention (final policy on all 8 modes)
Score / local specialist, **3-ep greedy (noisy) — clean 100-ep GPU eval pending**:

| mode | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | mean |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| retention % | 100 | 72 | 67 | 100 | 68 | 75 | 66 | 99 | **81** |

~81% of specialist retained, **no catastrophic forgetting** (worst 66%). Contrast
with our cross-game Atari5 where Boxing collapsed — Freeway's single-game variants
retain far more easily, and the `needy` consolidation re-patched modes that dipped.

## Caveats / disclosure
- Single seed; provisional (retention from 3-ep noisy checks).
- Ours PERF = local-phase (plasticity) curve; retention here = score/specialist,
  **not** the paper's success-rate Table-3/forgetting units (comparison in matching
  units needs the clean GPU eval).
- Ours uses **live past-task env access** the paper methods do not; Meta-World (not
  here) uses reduced ∆=300k.
- **Not established as "better than the paper."** On the comparable metric (PERF)
  ours is slightly below CKA-RL; strong absolute retention, but the head-to-head in
  the paper's units is pending.
