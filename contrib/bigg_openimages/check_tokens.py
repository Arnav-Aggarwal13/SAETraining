"""Verify a saev shard dir holds exactly open_clip's final bigG patch tokens.

Reference = `model.visual(x)` with output_tokens=True, i.e. ln_post(last block)[:, 1:], unprojected.
Compares the first --n images of the same ImageFolder root (saev uses torchvision ImageFolder order).
To check against your downstream pipeline instead, replace `reference_tokens` with its extractor.

Usage (GPU node, from repo root):
    uv run python contrib/bigg_openimages/check_tokens.py \
        --shards $SHARDS_ROOT/<hash> --root $OI_DIR/train
"""

import argparse
import pathlib

import numpy as np
import open_clip
import torch
from torchvision.datasets import ImageFolder

from saev.data.shards import Metadata


def reference_tokens(x: torch.Tensor, model) -> torch.Tensor:
    model.visual.output_tokens = True
    _, tokens = model.visual(x)
    return tokens  # [B, 256, 1664]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shards", type=pathlib.Path, required=True)
    p.add_argument("--root", type=pathlib.Path, required=True)
    p.add_argument("--n", type=int, default=8)
    a = p.parse_args()

    md = Metadata.load(a.shards)
    print(f"metadata: layers={md.layers} d_model={md.d_model} cls={md.cls_token} n={md.n_examples:,}")
    assert md.layers == (48,), "Expected layers=(48,) (ln_post output after the clip.py edit)."
    acts = np.memmap(a.shards / "acts000000.bin", dtype=np.float32, mode="r", shape=md.shard_shape)
    got = torch.from_numpy(np.array(acts[: a.n, 0, 1:, :]))  # token 0 is CLS

    torch.backends.cuda.matmul.allow_tf32 = True  # saev records with TF32 on
    torch.backends.cudnn.allow_tf32 = True
    model, tf = open_clip.create_model_from_pretrained("ViT-bigG-14", pretrained="laion2b_s39b_b160k")
    model = model.cuda().eval()
    ds = ImageFolder(a.root, transform=tf)
    x = torch.stack([ds[i][0] for i in range(a.n)]).cuda()
    with torch.inference_mode():
        ref = reference_tokens(x, model).float().cpu()

    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)  # [n, 256]
    rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
    print(f"cosine  min={cos.min():.6f} mean={cos.mean():.6f}")
    print(f"rel err max={rel.max():.2e} mean={rel.mean():.2e}")
    print(f"token norms: ref mean={ref.norm(dim=-1).mean():.2f}, got mean={got.norm(dim=-1).mean():.2f}")
    print("PASS" if cos.min() > 0.999 else "FAIL: shards do not match open_clip final tokens")


if __name__ == "__main__":
    main()
