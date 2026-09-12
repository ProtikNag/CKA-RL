#!/bin/bash
# ── One "Ours" (min-max) full-sequence Meta-World (SAC) run: one seed ──
#   sbatch scripts/hpc_ours_meta.sbatch <SEED> <TAG> [extra run_sac_ours.py args...]
# e.g. real:  sbatch scripts/hpc_ours_meta.sbatch 0 meta_s0 --num-tasks 20 --global-iters 800 --consolidate-mode needy
#      smoke: sbatch scripts/hpc_ours_meta.sbatch 0 smoke_meta --tasks 0,1 --task1-steps 6000 ...
#SBATCH --job-name=ours-meta
#SBATCH --partition=AI_Center_L40S
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-ours-meta-%j.out
#SBATCH --error=slurm-ours-meta-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail

SEED=${1:-0}
TAG=${2:-meta_s0}
shift 2 || true   # remaining args forwarded to run_sac_ours.py (--tasks/--num-tasks, budgets, etc.)

# env's interpreter directly (conda install on /work, not /home); MuJoCo state-obs
# needs no display but set EGL to be safe; TMPDIR off the tiny /tmp.
export PATH=/home/pnag/.conda/envs/cka-rl-meta/bin:$PATH
cd /work/pnag/CKA-RL-compare/experiments/meta-world
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/work/pnag/pip-tmp
export MUJOCO_GL=egl

echo "host=$(hostname) seed=$SEED tag=$TAG args=$* commit=$(git -C /work/pnag/CKA-RL-compare rev-parse --short HEAD 2>/dev/null || echo n/a)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

python run_sac_ours.py --seed "$SEED" --tag "$TAG" "$@"
echo "DONE tag=$TAG"
