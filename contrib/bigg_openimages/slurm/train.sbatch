#!/bin/bash
#SBATCH -J bigg-sae
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
N_TRAIN=${N_TRAIN:-125000000}   # smoke test: sbatch --export=ALL,N_TRAIN=2000000 slurm/train.sbatch
# Parent flags first, then the activation subcommand and its flags (tyro parsing order).
uv run launch.py train \
  --runs-root $RUNS_ROOT \
  --train-data.shards $SHARDS_ROOT/$TRAIN_HASH --train-data.layer 48 \
  --val-data.shards $SHARDS_ROOT/$VAL_HASH --val-data.layer 48 \
  --sae.d-model 1664 --sae.d-sae 26624 \
  --objective.n-prefixes 1 \
  --n-train $N_TRAIN --n-val 5000000 \
  --lr 4e-4 \
  --wandb-project bigg-openimages-sae \
  sae.activation:batch-top-k --sae.activation.top-k 64
