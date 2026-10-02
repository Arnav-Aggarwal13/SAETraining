#!/bin/bash
#SBATCH -J bigg-shards
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
SPLIT=${SPLIT:-train}   # sbatch --export=ALL,SPLIT=val slurm/shards.sbatch
uv run launch.py shards \
  --shards-root $SHARDS_ROOT \
  --family clip \
  --ckpt ViT-bigG-14/laion2b_s39b_b160k \
  --d-model 1664 \
  --layers 48 \
  --content-tokens-per-example 256 \
  --batch-size 128 \
  --n-workers 12 \
  data:img-folder \
  --data.root $OI_DIR/$SPLIT
