#!/bin/bash
#SBATCH -J bigg-visuals
# EDIT the partition below (see `sinfo`)
#SBATCH -p mit_normal
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
# Needs inference.sbatch to have finished for the same RUN and SPLIT (reads token_acts.npz, mean_values.pt, sparsity.pt).
# sbatch --export=ALL,RUN=9za00zoc,LATENTS="12 345" slurm/visuals.sbatch
RUN=${RUN:?Set RUN to a run id in $RUNS_ROOT, e.g. --export=ALL,RUN=9za00zoc}
SPLIT=${SPLIT:-val}
case $SPLIT in
  val) HASH=$VAL_HASH ;;
  train) HASH=$TRAIN_HASH ;;
  *) echo "SPLIT must be val or train, got '$SPLIT'" >&2; exit 1 ;;
esac
# Latents are sampled at random (N_LATENTS of them) from those inside both log10 ranges, plus any in LATENTS.
# Want high value, low frequency, so tighten these after looking at the sparsity/value histograms (see saev/interactive/features.py).
LOG_FREQ=${LOG_FREQ:--6.0 1.0}      # log10 fraction of tokens a latent fires on: "MIN MAX"
LOG_VALUE=${LOG_VALUE:--3.0 3.0}    # log10 mean activation when firing: "MIN MAX"
N_LATENTS=${N_LATENTS:-400}
TOP_K=${TOP_K:-20}                  # top images saved per latent
LATENTS=${LATENTS:-}                # extra fixed latents, space separated
latent_args=()
[ -n "$LATENTS" ] && latent_args=(--latents $LATENTS)
# Run from the repo root so the saev env is used (uv run inside contrib/trait_discovery syncs only that member and drops saev's deps, e.g. beartype). --with-editable adds tdiscovery and its deps on top.
# Images go to $RUNS_ROOT/$RUN/inference/$HASH/images/<latent>/.
uv run --with-editable contrib/trait_discovery python contrib/trait_discovery/scripts/launch.py visuals \
  --run $RUNS_ROOT/$RUN \
  --shards $SHARDS_ROOT/$HASH \
  --log-freq-range $LOG_FREQ --log-value-range $LOG_VALUE \
  --n-latents $N_LATENTS --top-k $TOP_K \
  ${latent_args[@]+"${latent_args[@]}"}
