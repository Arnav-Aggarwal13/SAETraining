# SAE on OpenCLIP ViT-bigG/14 final patch tokens (Open Images v7) with saev

Verified against Imageomics/saev @ 9715c45 (2026-06-18) and open_clip_torch 3.3.0. Re-check if versions differ.

## Project files (contrib/bigg_openimages/)
- `env.sh`: all paths (pool/scratch/shards/runs/HF cache/NSD) + TRAIN_HASH/VAL_HASH. Every job sources it.
- `stream_openimages.py`: S3 -> 256px thumbnails, ImageFolder layout `<out>/<id[:2]>/<id>.jpg`, resumable, skips Rotation != 0, writes `<out>_ids.txt`. Don't re-run after dedup (it would re-fetch quarantined IDs).
- `dedup_nsd.py`: ViT-B-32 cosine vs NSD `imgBrick`; moves matches >= thresh (0.95) to a quarantine dir; `matches.csv` for eyeballing; `--dry-run` first.
- `check_tokens.py`: asserts shard tokens == open_clip `model.visual(x)` output_tokens (ln_post, unprojected). Must PASS before training.
- `slurm/{stream,dedup,shards,train}.sbatch`: partitions are placeholders; submit from repo root; `logs/` must exist before `sbatch`. Overrides via `--export=ALL,VAR=...` (SPLIT, DRY, N_TRAIN).
- tyro CLI order: parent flags first, then `sae.activation:batch-top-k` and its `--sae.activation.*` flags last. `objective` is not a subcommand (single type); use `--objective.n-prefixes`.
- Root CLAUDE.md/AGENTS.md are saev's; their "scripts/launch.py" mention is stale (launch.py is at the root). The CLAUDE.md/AGENTS.md under contrib/trait_discovery/docs/papers/ are for LaTeX papers and are irrelevant.

## Repo setup
- Repo: github.com/Arnav-Aggarwal13/SAETraining. A clone of Imageomics/saev with full saev history (not a GitHub fork; standalone, can be private).
- Remotes: `origin` = Arnav-Aggarwal13/SAETraining (push here); `upstream` = Imageomics/saev (pull fixes: `git fetch upstream && git merge upstream/main`).
- Project code goes in `contrib/bigg_openimages/` (saev's convention, cf. `contrib/trait_discovery`): streaming, shard wrapper, token check, sweeps, slurm. This notes file lives there too. Leave saev's root `CLAUDE.md`/`AGENTS.md` (their conventions, uv, lint) in place.
- Make core saev changes (e.g. clip.py, the fp16 writer) directly in `src/saev/`, kept small to limit upstream merge conflicts.
- Where things happen. Write code in `contrib/bigg_openimages/`, make minimal core edits in `src/saev/`, and ALWAYS run commands from the repo root:
  - saev steps: `uv run launch.py {shards,train,inference} ...` (launch.py lives at the root).
  - Our scripts: `uv run python contrib/bigg_openimages/<script>.py`.
  - Edits outside contrib: `src/saev/data/clip.py` (ln_post, once), `src/saev/data/shards.py` + loaders (only for fp16/token subsampling), `.gitignore`.
  - Extra deps (e.g. boto3 for S3 streaming): `uv add boto3` in the root env. Only make contrib/bigg_openimages a uv workspace member (own pyproject.toml, listed under `[tool.uv.workspace]` in the root pyproject, like trait_discovery) if it needs many deps of its own.
- Data never goes in git: thumbnails in pool, shards in scratch, runs elsewhere (see Storage). Add `*.bin`, `wandb/`, and data paths to `.gitignore`.

## Goal
Train an SAE on the 256 x 1664 patch tokens that `model.visual(img)` returns with `output_tokens=True`, one token per training example, CLS excluded.

## What "final image embedding" means in open_clip (critical)
- `ViT-bigG-14` / pretrained `laion2b_s39b_b160k`: 48 resblocks, width 1664, patch 14, 224px, so 16x16 = 256 patches + CLS = 257 tokens. Transformer is batch_first, so resblock outputs are [B, 257, 1664].
- VisionTransformer forward: `x = transformer(x)`, then (bigG uses pool_type='tok', final_ln_after_pool=False) `x = ln_post(x)`, `pooled, tokens = x[:, 0], x[:, 1:]`. `proj` (1664 -> 1280) applies ONLY to pooled, never to tokens.
- So the target is `ln_post(resblocks[-1] output)[:, 1:]`, unprojected, 1664-d.
- saev does NOT record this by default. It hooks `model.transformer.resblocks[i]`, so layer -1 is the residual stream BEFORE ln_post. Must patch (below).
- Preprocessing (open_clip default, which saev uses via `create_model_from_pretrained`): resize shortest side 224 bicubic, center crop 224, OpenAI mean (0.48145466, 0.4578275, 0.40821073) / std (0.26862954, 0.26130258, 0.27577711).
- Before anything else: confirm numerically that tokens match the downstream pipeline's tensor for a few images.

## Patch saev to record ln_post output (edit src/saev/data/clip.py directly)
`Vit.get_residuals()` returns `self.model.transformer.resblocks`. Change it to append ln_post, so index 48 is the final token output (hook output [B, 257, 1664]; `cls_token=True` stores CLS at token 0, and the loader's `tokens="content"` drops it):

```python
def get_residuals(self):
    return [*self.model.transformer.resblocks, self.model.ln_post]
```
Existing layer indices 0-47 are unchanged, so this is backward compatible. Since it's in the source, `uv run launch.py shards ... --layers 48` works directly (no wrapper/monkeypatch). Train/val loaders use `layer=48`. Add a comment noting the change, to help with upstream merges.

## Getting images (streaming Open Images, no bulk download)
- Images are in public S3 bucket `open-images-dataset`, key `{split}/{image_id}.jpg`, readable unsigned (boto3 `Config(signature_version=botocore.UNSIGNED)`; this is what the official `downloader.py` does). Get the train image-ID list from the V7 download page; train has ~9M IDs.
- saev datasets are map-style only (`ImgFolder`, `Imagenet` via HF, etc.), no streaming, and `n_examples` must be known up front.
- Recommended: stream by ID, resize to shortest side 256 (or 224 + center crop), save JPEG into `root/all/*.jpg`, then use `data:img-folder --data.root root` (ImgFolder needs a class subfolder; a dummy "all" works). ~20 KB/img, so ~40 GB for 2M vs ~500 GB originals. Keeps saev's visualization working (it re-reads images via the pickled dataset config in metadata.json).
- Some IDs are dead/corrupt: skip failures, `.convert("RGB")`, check the CSV `Rotation` column.
- Dedup against NSD (embedding cosine / pHash) BEFORE writing shards.
- Also build a small held-out val folder (e.g. from the Open Images validation split), shard it separately, pass as `--val-data.shards`.

## saev shards (activation cache)
- Layout: `<shards_root>/<8-char hash>/{metadata.json, shards.json, acts000000.bin, ...}`. `shards_root` MUST end in `saev/shards` and exist; runs root MUST end in `saev/runs` (asserted).
- float32 only (hardcoded memmap). Full token axis stored, no per-image token subsampling. bigG final layer: 257 x 1664 x 4 B = 1.71 MB/img, so 1M imgs = 1.7 TB, 2M = 3.4 TB. Shrinking needs code changes in ShardWriter, Metadata.dtype, and the loaders (they assume 4-byte floats).
- `max_tokens_per_shard` default 2.4M, so ~9.3k imgs and ~16 GB per shard at d=1664.
- Shard args to set (CLI flags or worker_fn kwargs): `family="clip"`, `ckpt="ViT-bigG-14/laion2b_s39b_b160k"`, `d_model=1664`, `content_tokens_per_example=256`, `cls_token=True`, `layers=[48]` (after the clip.py edit), `data=datasets.ImgFolder(root=...)`, `shards_root`, `batch_size` (default 1024 is too big for bigG; start ~128), `n_workers`. Runs fp32 with TF32 matmuls enabled. Slurm via submitit if using the CLI.
- Token count: 1M imgs x 256 = 256M tokens, above the 100M default `n_train`.

## Storage on MIT ORCD Engaging (where each thing lives)
- Home `/home/<user>` (200 GB, flash, backed up): saev code, env, SAE checkpoints/runs, image-ID lists, and the scripts that regenerate shards.
- Pool `/home/<user>/orcd/pool` (1 TB, HDD, not backed up): Open Images thumbnails (~20 KB/img).
- Scratch `/home/<user>/orcd/scratch` (1 TB, flash, not backed up, wiped after 6 months without login): activation shards (train + val). Training is disk-I/O bound with random reads, so shards must be on scratch, not pool (HDD).
- saev needs one shards dir per set, so a set can't be split across scratch and pool. The paths must still end in `saev/shards` and `saev/runs` (e.g. `~/orcd/scratch/saev/shards`, `~/saev_runs/saev/runs`).
- Budget: 1 TB scratch holds ~580k imgs at default fp32 x 257 tokens. Plan ~500k train imgs (~850 GB, 128M patch tokens) + ~20k val imgs (~35 GB).
- To scale beyond that: patch the writer/loaders for fp16 + a 64-token subsample (~215 KB/img, so 2M imgs is ~430 GB), or have the PI request the 5 TB shared pool / rent storage (HDD, slower).
- Check usage in `~/orcd/.quota` (updates every 30 min).

## Training (`uv run launch.py train ...`; launch.py is at repo root, docs saying scripts/launch.py or train.py are stale)
Key config (`saev/framework/train.py`, `saev/nn/modeling.py`, `saev/nn/objectives.py`, `saev/data/shuffled.py`):
- Data: `--train-data.shards DIR --train-data.layer 48` (same for `--val-data`). `tokens` default "content" (patches only). `batch_size` default 16384. Loader shuffles across images and patch positions and logs `loader/*entropy` diagnostics; check them.
- `scale_norm=True` raises NotImplementedError. Input normalization isn't available, so if needed apply a constant scale inside the hook before writing (ln_post output is already LayerNorm'd, so likely fine; check the token-norm distribution for outliers first).
- SAE: `--sae.d-model 1664 --sae.d-sae N`. Activation is a tyro subcommand (`sae.activation:relu` per docs; BatchTopK key is `batch-top-k`; confirm spelling with `--help`):
  - `TopK` (default): top_k=32, NoSparsity, AuxK.
  - `BatchTopK`: top_k=32, momentum=0.1, AuxK. Train mode: top (batch*k) over the whole batch. Eval mode: JumpReLU with EMA threshold (EMA of min surviving activation), so per-sample independent.
  - `Relu`: L1Sparsity coeff 4e-4, NoAux.
  - `AuxK`: k_aux=512, alpha=1/32; a latent is "dead" after `objective.dead_threshold_tokens` (10M) tokens without firing.
- Objective: ONLY Matryoshka exists and it is the default (n_prefixes=10, Pareto-sampled prefix cuts, last prefix is always full d_sae). For a plain SAE set `--objective.n-prefixes 1`.
- Init: `reinit_blend=0.8` datapoint init (encoder rows = 0.8 x mean-centered real activations + 0.2 x kaiming; W_dec = W_enc^T; needs >= max(d_sae, 65536) samples). b_dec starts at 0. `normalize_w_dec=True`, `remove_parallel_grads=True`.
- Optim: Adam (or Muon), lr 4e-4, 500 warmup steps, then cosine to 0 over n_train (`WarmupCosine`), grad_clip 1.0, `n_train` 100M, `n_val` 10M. Sparsity-coeff scheduling is commented out, so `n_sparsity_warmup` does nothing.
- Logs (wandb, `track=True`): normalized_mse (SSE / mean-baseline SSE), L0, L1, dead_unit_pct, lr. Post-train eval: n_dead, n_almost_dead (<1e-7 freq), n_dense.
- Sweeps: `--sweep file.py` where file defines `make_cfgs() -> list[dict]` of nested overrides (e.g. `{"lr": 3e-4, "sae": {"d_sae": 26624}}`). SAEs with identical data config train in parallel on one GPU off one data stream. Docs showing TOML are stale.
- Eval afterward: `uv run launch.py inference --run RUN --data.shards DIR --data.layer 48` writes metrics + sparse `token_acts.npz`. Visualization lives in `contrib/trait_discovery` (`scripts/launch.py visuals`).

## Starting config and what to iterate on
- Start: BatchTopK, top_k 64, d_sae 26624 (16x), `--objective.n-prefixes 1`, lr 4e-4, batch 16384, n_train ~= available tokens (<= ~1 epoch).
- Sweep dims in priority order: top_k {32, 64, 128}, d_sae {16x=26624, 32x=53248}, lr {1e-4, 4e-4, 1e-3}. Then Matryoshka (n_prefixes 10) as a variant.
- Targets: normalized MSE roughly < 0.1-0.15 at k=64, dead < few %. If dead is high: raise alpha or k_aux, lower dead_threshold_tokens, or lower lr. If MSE is poor: raise k or d_sae.
- Real test: swap SAE reconstructions for true tokens in the downstream pipeline and measure degradation.

## Known rough edges in saev code
- BatchTopK threshold update checks `pos.numel() >= 0` (always true), so an all-zero batch would crash on `pos.min()`.
- Matryoshka decode path has an author TODO saying it needs cleanup.
- Docs lag the code in several places (launch paths, sweep format, the activations vs shards naming). Trust the source.
