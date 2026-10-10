#!/bin/bash
#SBATCH -J bigg-best
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# One SAE. Defaults: best nMSE so far (53k latents, k=128, AuxK alpha 1/32) with the 2M dead threshold, at lr 1e-3. Override: sbatch --export=ALL,LR=4e-4,TOP_K=256 slurm/train_best.sbatch
LR=${LR:-1e-3}
D_SAE=${D_SAE:-53248}
TOP_K=${TOP_K:-128}
DEAD_TOKENS=${DEAD_TOKENS:-2000000}
N_TRAIN=${N_TRAIN:-125000000}
# --no-sync: avoids rebuilding the shared .venv when other jobs run. Parent flags first, then the activation subcommand and its flags.
uv run --no-sync launch.py train \
  --runs-root $RUNS_ROOT \
  --train-data.shards $SHARDS_ROOT/$TRAIN_HASH --train-data.layer 48 \
  --val-data.shards $SHARDS_ROOT/$VAL_HASH --val-data.layer 48 \
  --sae.d-model 1664 --sae.d-sae $D_SAE \
  --objective.n-prefixes 1 --objective.dead-threshold-tokens $DEAD_TOKENS \
  --n-train $N_TRAIN --n-val 5000000 \
  --lr $LR \
  --wandb-project bigg-openimages-sae \
  sae.activation:batch-top-k --sae.activation.top-k $TOP_K
