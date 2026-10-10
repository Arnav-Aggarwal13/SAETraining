#!/bin/bash
#SBATCH -J bigg-inference
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
# RUN is the run id under $RUNS_ROOT: sbatch --export=ALL,RUN=9za00zoc slurm/inference.sbatch
RUN=${RUN:-9za00zoc}
SPLIT=${SPLIT:-val}   # val or train. token_acts.npz is sparse but ~65 GB for the full train set, so val is the default.
case $SPLIT in
  val) HASH=$VAL_HASH ;;
  train) HASH=$TRAIN_HASH ;;
  *) echo "SPLIT must be val or train, got '$SPLIT'" >&2; exit 1 ;;
esac
# Writes metrics.json + sparse token_acts.npz under $RUNS_ROOT/$RUN/inference/<shards hash>/.
uv run launch.py inference \
  --run $RUNS_ROOT/$RUN \
  --data.shards $SHARDS_ROOT/$HASH --data.layer 48
