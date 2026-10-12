#!/bin/bash
#SBATCH -J bigg-sweep3
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 05:00:00
#SBATCH --array=0-6
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# Array index = sweep group in sweeps/sweep3.py (0: both 26k SAEs, 1-6: one 53k SAE each). Rerun some: sbatch --array=2,5 slurm/sweep3.sbatch
export SWEEP_GROUP=$SLURM_ARRAY_TASK_ID
# Less allocator fragmentation (the 32x OOM in sweep 1 showed 13.9 GB reserved but unallocated).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --no-sync: array tasks start together, and concurrent `uv run` syncs would race to rebuild the shared .venv.
# Do NOT pass --lr, --sae.d-sae, --sae.activation.*, --objective.* here: CLI values override the sweep file.
uv run --no-sync launch.py train \
  --runs-root $RUNS_ROOT \
  --train-data.shards $SHARDS_ROOT/$TRAIN_HASH --train-data.layer 48 \
  --val-data.shards $SHARDS_ROOT/$VAL_HASH --val-data.layer 48 \
  --sae.d-model 1664 \
  --n-train 125000000 --n-val 5000000 \
  --wandb-project bigg-openimages-sae \
  --sweep contrib/bigg_openimages/sweeps/sweep3.py \
  sae.activation:batch-top-k
