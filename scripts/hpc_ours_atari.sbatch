#!/bin/bash
# ── One "Ours" (min-max) full-sequence Atari run: one game, one seed ──
#   sbatch scripts/hpc_ours_atari.sbatch <ENV_ID> <MODES> <SEED> <TAG> [extra run_ours.py args...]
# e.g. sbatch scripts/hpc_ours_atari.sbatch ALE/SpaceInvaders-v5 0,1,2,3,4,5,6,7,8,9 0 ours_si_s0
# Runs on a stable GPU partition (NOT V100 — those preempt/requeue; see HANDOFF).
#SBATCH --job-name=ours-atari
#SBATCH --partition=AI_Center_L40S
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-ours-atari-%j.out
#SBATCH --error=slurm-ours-atari-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
set -euo pipefail

ENV_ID=${1:?env id e.g. ALE/SpaceInvaders-v5}
MODES=${2:?comma modes e.g. 0,1,2,3,4,5,6,7,8,9}
SEED=${3:-0}
TAG=${4:-ours_s0}
shift 4 || true   # any remaining args are forwarded to run_ours.py (budgets etc.)

# Use the env's interpreter directly (conda install lives on /work, not /home;
# no `conda activate` needed to run a script with an absolute python path).
PY=/home/pnag/.conda/envs/cka-rl-atari/bin/python
cd /work/pnag/CKA-RL-compare/experiments/atari
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/work/pnag/pip-tmp   # avoid the tiny 2GB /tmp

echo "host=$(hostname) env=$ENV_ID modes=$MODES seed=$SEED tag=$TAG commit=$(git -C /work/pnag/CKA-RL-compare rev-parse --short HEAD 2>/dev/null || echo n/a)"
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# Budgets are forwarded via "$@" so they can be tuned per launch. Defaults below
# are the real-run values; override by passing e.g. --global-iters 2000.
$PY run_ours.py \
  --env-id "$ENV_ID" --modes "$MODES" --seed "$SEED" --tag "$TAG" \
  --task1-steps 1000000 --local-steps 1000000 --global-iters 2000 \
  --eval-episodes 100 --constraint-episodes 16 \
  --log-every 5 --ckpt-every 100 \
  "$@"
echo "DONE tag=$TAG"
