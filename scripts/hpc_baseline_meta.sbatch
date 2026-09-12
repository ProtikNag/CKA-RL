#!/bin/bash
# ── Meta-World "simple" baseline (train-from-scratch per task) for the FWT column ──
#   sbatch scripts/hpc_baseline_meta.sbatch <SEED> <TAG> [--num-tasks N]
# Use the SAME <TAG> as the Ours Meta run so extract_results/process_results merge them.
# run_experiments.py spawns `python3 run_sac.py` per task -> PATH must resolve to cka-rl-meta.
#SBATCH --job-name=base-meta
#SBATCH --partition=dgx_aic
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-base-meta-%j.out
#SBATCH --error=slurm-base-meta-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
set -euo pipefail

SEED=${1:-0}
TAG=${2:-meta_s0}
shift 2 || true

export PATH=/home/pnag/.conda/envs/cka-rl-meta/bin:$PATH
cd /work/pnag/CKA-RL-compare/experiments/meta-world
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/work/pnag/pip-tmp
export MUJOCO_GL=egl

echo "host=$(hostname) seed=$SEED tag=$TAG args=$*"
python -c "import torch; print('cuda', torch.cuda.is_available())"

python run_experiments.py --algorithm simple --seed "$SEED" --tag "$TAG" \
  --num-tasks 20 --total-timesteps 1000000 "$@"
echo "DONE baseline-meta tag=$TAG"
