#!/bin/bash
#SBATCH -J oi-stream
#SBATCH -p mit_normal
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
uv run python contrib/bigg_openimages/stream_openimages.py \
  --ids-csv $OI_DIR/oidv6-train-images-with-labels-with-rotation.csv \
  --split train --out $OI_DIR/train --n 500000 --threads 64
uv run python contrib/bigg_openimages/stream_openimages.py \
  --ids-csv $OI_DIR/validation-images-with-rotation.csv \
  --split validation --out $OI_DIR/val --n 20000 --threads 64
