#!/bin/bash
#SBATCH -J bigg-sweep2
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 02:00:00
#SBATCH --array=0-1
#SBATCH -o logs/%x-%A_%a.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# Array index = n_train in sweeps/sweep2.py (0: 32M tokens, 1: 64M tokens). Baseline hyperparameters, as in train.sbatch.
export SWEEP_GROUP=$SLURM_ARRAY_TASK_ID
# --no-sync: array tasks start together, and concurrent `uv run` syncs would race to rebuild the shared .venv.
# Do NOT pass --n-train here: CLI values override the sweep file.
uv run --no-sync launch.py train \
  --runs-root $RUNS_ROOT \
  --train-data.shards $SHARDS_ROOT/$TRAIN_HASH --train-data.layer 48 \
  --val-data.shards $SHARDS_ROOT/$VAL_HASH --val-data.layer 48 \
  --sae.d-model 1664 --sae.d-sae 26624 \
  --objective.n-prefixes 1 \
  --n-val 5000000 \
  --lr 4e-4 \
  --wandb-project bigg-openimages-sae \
  --sweep contrib/bigg_openimages/sweeps/sweep2.py \
  sae.activation:batch-top-k --sae.activation.top-k 64
