#!/bin/bash
#SBATCH -J bigg-inference-all
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# Metrics-only inference on the val split for many runs, one after another (writes metrics.json with the cosine fields; no token_acts).
# Override the list: sbatch --export=ALL,RUNS="9za00zoc 0jhp03qu" slurm/inference_all.sbatch
RUNS=${RUNS:-"0jhp03qu 88oufjem cs0vbj13 s8nya5no"}
for run in $RUNS; do
  echo "=== $run ==="
  uv run --no-sync launch.py inference \
    --run $RUNS_ROOT/$run \
    --data.shards $SHARDS_ROOT/$VAL_HASH --data.layer 48 \
    --no-save --force-recompute
done
