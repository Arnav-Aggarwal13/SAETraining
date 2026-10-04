#!/bin/bash
#SBATCH -J oi-dedup
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_gpu
#SBATCH --qos=mit_amf_standard_gpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
DRY=${DRY:---dry-run}   # submit with DRY= to actually move files: sbatch --export=ALL,DRY= slurm/dedup.sbatch
for s in train val; do
  uv run python contrib/bigg_openimages/dedup_nsd.py \
    --root $OI_DIR/$s --nsd-hdf5 $NSD_HDF5 --quarantine $OI_DIR/quarantine_$s $DRY
done
