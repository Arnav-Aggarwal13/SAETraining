"""Remove thumbnails that are near-duplicates of NSD stimuli (run BEFORE making shards).

Embeds NSD images and thumbnails with a small CLIP (ViT-B-32/openai, cheap and good at near-dups),
and moves any thumbnail whose max cosine similarity to an NSD image is >= --thresh into --quarantine
(outside the ImageFolder root, so saev never sees it). Writes <quarantine>/matches.csv with every pair
>= --thresh - 0.05 so you can eyeball pairs near the cutoff and adjust.

Usage (GPU node, from repo root):
    uv run python contrib/bigg_openimages/dedup_nsd.py \
        --root $OI_DIR/train --nsd-hdf5 /path/to/nsd_stimuli.hdf5 \
        --quarantine $OI_DIR/quarantine_train --dry-run
"""

import argparse
import pathlib
import shutil

import h5py
import open_clip
import polars as pl
import torch
from PIL import Image
from tqdm import tqdm


class Files(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.inference_mode()
def embed(model, loader, dev):
    out = []
    for x in tqdm(loader):
        f = model.encode_image(x.to(dev))
        out.append(torch.nn.functional.normalize(f.float(), dim=-1).half())
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=pathlib.Path, required=True, help="Thumbnail ImageFolder root.")
    p.add_argument("--nsd-hdf5", type=pathlib.Path, required=True)
    p.add_argument("--nsd-key", default="imgBrick")
    p.add_argument("--quarantine", type=pathlib.Path, required=True)
    p.add_argument("--thresh", type=float, default=0.95)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="Only write matches.csv; move nothing.")
    a = p.parse_args()

    dev = "cuda"
    model, tf = open_clip.create_model_from_pretrained("ViT-B-32", pretrained="openai")
    model = model.to(dev).eval()
    a.quarantine.mkdir(parents=True, exist_ok=True)

    cache = a.quarantine.parent / "nsd_b32_feats.pt"  # reused across train/val runs
    if cache.exists():
        nsd = torch.load(cache).to(dev)
    else:
        with h5py.File(a.nsd_hdf5, "r") as f:
            brick = f[a.nsd_key]
            feats = []
            for s in tqdm(range(0, len(brick), a.batch), desc="NSD"):
                x = torch.stack([tf(Image.fromarray(im)) for im in brick[s : s + a.batch]])
                with torch.inference_mode():
                    feats.append(
                        torch.nn.functional.normalize(
                            model.encode_image(x.to(dev)).float(), dim=-1
                        ).half()
                    )
        nsd = torch.cat(feats)
        torch.save(nsd.cpu(), cache)

    paths = sorted(a.root.rglob("*.jpg"))
    loader = torch.utils.data.DataLoader(
        Files(paths, tf), batch_size=a.batch, num_workers=a.workers
    )
    oi = embed(model, loader, dev)

    best_sim, best_idx = [], []
    for s in range(0, len(oi), 4096):
        sim = oi[s : s + 4096] @ nsd.T
        v, i = sim.max(dim=1)
        best_sim.append(v.float().cpu())
        best_idx.append(i.cpu())
    best_sim, best_idx = torch.cat(best_sim), torch.cat(best_idx)

    near = (best_sim >= a.thresh - 0.05).nonzero().flatten().tolist()
    rows = [
        {"path": str(paths[k]), "nsd_index": int(best_idx[k]), "sim": float(best_sim[k])}
        for k in near
    ]
    pl.DataFrame(rows, schema=["path", "nsd_index", "sim"]).sort("sim", descending=True).write_csv(
        a.quarantine / "matches.csv"
    )

    hits = [r for r in rows if r["sim"] >= a.thresh]
    print(f"{len(hits):,} of {len(paths):,} thumbnails >= {a.thresh} (near-cutoff pairs in matches.csv)")
    if not a.dry_run:
        for r in hits:
            src = pathlib.Path(r["path"])
            shutil.move(src, a.quarantine / src.name)
        print(f"Moved {len(hits):,} files to {a.quarantine}")


if __name__ == "__main__":
    main()
