#!/bin/bash
#SBATCH -J nsd-download
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

DEST=$HOME/orcd/pool/nsd_stimuli
mkdir -p "$DEST"
echo "Node: $(hostname)  Start: $(date)"

uvx --from awscli aws s3 cp --no-sign-request \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  "$DEST/"

# Verify the size matches the bucket listing (39556877048 bytes)
SIZE=$(stat -c %s "$DEST/nsd_stimuli.hdf5")
echo "Downloaded $SIZE bytes. End: $(date)"
[ "$SIZE" -eq 39556877048 ] && echo "SIZE OK" || { echo "SIZE MISMATCH"; exit 1; }
