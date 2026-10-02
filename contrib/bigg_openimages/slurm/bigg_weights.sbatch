#!/bin/bash
#SBATCH -J bigg-weights
#SBATCH -p mit_normal
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --account=mit_amf_standard_cpu
#SBATCH --qos=mit_amf_standard_cpu
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd $HOME/SAETraining && source contrib/bigg_openimages/env.sh
echo "Node: $(hostname)  HF_HOME=$HF_HOME  Start: $(date)"

# Downloads OpenCLIP ViT-bigG-14 (laion2b_s39b_b160k) into $HF_HOME and loads it once to confirm it is intact.
uv run python -c "
import open_clip, torch
model, _ = open_clip.create_model_from_pretrained('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
n = sum(p.numel() for p in model.visual.parameters())
print(f'Loaded OK. Visual params: {n/1e9:.2f}B, width: {model.visual.transformer.width}')
"

echo "Cache contents:"
du -sh "$HF_HOME"/hub/* 2>/dev/null || true
echo "End: $(date)"
