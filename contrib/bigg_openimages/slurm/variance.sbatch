#!/bin/bash
#SBATCH -J bigg-variance
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# RUN is the run id under $RUNS_ROOT: sbatch --export=ALL,RUN=9za00zoc slurm/variance.sbatch
RUN=${RUN:?Set RUN to a run id in $RUNS_ROOT, e.g. --export=ALL,RUN=9za00zoc}
# Val shards only. Outputs go to $RUNS_ROOT/$RUN/inference/$VAL_HASH/variance_analysis/.
uv run --no-sync python contrib/bigg_openimages/variance_analysis.py \
  --run $RUNS_ROOT/$RUN --shards $SHARDS_ROOT/$VAL_HASH
