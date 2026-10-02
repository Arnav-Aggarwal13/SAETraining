# Paths for the bigG SAE project. Edit once, then: source contrib/bigg_openimages/env.sh
export REPO=$HOME/SAETraining
export POOL=$HOME/orcd/pool
export SCRATCH=$HOME/orcd/scratch
export OI_DIR=$POOL/openimages              # ID CSVs + thumbnails (train/, val/)
export SHARDS_ROOT=$SCRATCH/saev/shards     # must end in saev/shards
export RUNS_ROOT=$HOME/sae_runs/saev/runs   # must end in saev/runs
export HF_HOME=$POOL/hf_cache               # bigG weights (~10 GB) land here, not in home
export NSD_HDF5=$HOME/orcd/pool/nsd_stimuli/nsd_stimuli.hdf5
# Filled in after making shards (ls $SHARDS_ROOT):
export TRAIN_HASH=
export VAL_HASH=
mkdir -p "$OI_DIR" "$SHARDS_ROOT" "$RUNS_ROOT" "$HF_HOME" "$REPO/logs"
