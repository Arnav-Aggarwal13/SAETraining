#!/bin/bash
#SBATCH -J bigg-inference
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=128G
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
# --data.buffer-size 4: for big SAEs (53k latents, k=128) the consumer is slower than the loader. With the default 64-batch buffer the loader finishes early, waits 60 s, and tears down shared memory while the consumer still has minutes of batches left ("Manager process died unexpectedly"). A small buffer keeps it from running far ahead.
uv run launch.py inference \
  --run $RUNS_ROOT/$RUN \
  --data.shards $SHARDS_ROOT/$HASH --data.layer 48 --data.buffer-size 4
