#!/bin/bash
# ── Baseline (train-from-scratch per mode) Atari sequence: needed for the FWT ──
# column of Table 1 (forward transfer is measured against Baseline's curve).
#   sbatch scripts/hpc_baseline_atari.sbatch <ENV_ID> <FIRST_MODE> <LAST_MODE> <SEED> <TAG>
# e.g. sbatch scripts/hpc_baseline_atari.sbatch ALE/SpaceInvaders-v5 0 9 0 ours_si_s0
# NOTE: use the SAME <TAG> as the "Ours" run so gather_rt_results merges them.
#SBATCH --job-name=base-atari
#SBATCH --partition=AI_Center_L40S
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-base-atari-%j.out
#SBATCH --error=slurm-base-atari-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail

ENV_ID=${1:?env id e.g. ALE/SpaceInvaders-v5}
FIRST=${2:?first mode}
LAST=${3:?last mode}
SEED=${4:-0}
TAG=${5:-ours_s0}

# Prepend the env's bin so BOTH this script AND run_experiments.py's spawned
# `python run_ppo.py` subprocess resolve to the cka-rl-atari interpreter
# (conda install is on /work, not /home; no `conda activate` needed).
export PATH=/home/pnag/.conda/envs/cka-rl-atari/bin:$PATH
cd /work/pnag/CKA-RL-compare/experiments/atari
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/work/pnag/pip-tmp   # avoid the tiny 2GB /tmp

echo "host=$(hostname) env=$ENV_ID modes=$FIRST..$LAST seed=$SEED tag=$TAG"
python -c "import torch; print('cuda', torch.cuda.is_available())"

python run_experiments.py --method_type Baseline --env "$ENV_ID" \
  --first-mode "$FIRST" --last-mode "$LAST" --seed "$SEED" --tag "$TAG" \
  --total_timesteps 1000000
echo "DONE baseline tag=$TAG"
