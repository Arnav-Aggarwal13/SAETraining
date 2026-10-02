"""Stream a random Open Images subset from the public S3 bucket and save small thumbnails.

Output is ImageFolder layout for saev's `data:img-folder`:
    <out>/<first 2 chars of id>/<id>.jpg
(the 2-char subfolders act as dummy "classes"; labels are ignored for SAE training).

Resumable: re-running with the same --seed walks IDs in the same order and skips files that exist.
Writes <out>_ids.txt listing the saved image IDs (keep this in git/home for reproducibility).

Usage (from repo root):
    uv run python contrib/bigg_openimages/stream_openimages.py \
        --ids-csv $OI_DIR/oidv6-train-images-with-labels-with-rotation.csv \
        --split train --out $OI_DIR/train --n 500000
"""

import argparse
import concurrent.futures as cf
import io
import pathlib
import random

import boto3
import botocore
import botocore.config
import polars as pl
from PIL import Image, ImageOps
from tqdm import tqdm

BUCKET = "open-images-dataset"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids-csv", type=pathlib.Path, required=True)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--n", type=int, required=True, help="Number of images to save.")
    p.add_argument("--size", type=int, default=256, help="Shortest side in pixels.")
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = pl.scan_csv(a.ids_csv).select("ImageID", "Rotation").collect()
    # Skip images flagged as needing rotation (a small fraction) to avoid orientation errors.
    df = df.filter(pl.col("Rotation").is_null() | (pl.col("Rotation") == 0))
    ids = df["ImageID"].to_list()
    random.Random(a.seed).shuffle(ids)
    print(f"{len(ids):,} candidate IDs; saving {a.n:,} to {a.out}")

    a.out.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client(
        "s3",
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED, max_pool_connections=a.threads
        ),
    )

    def fetch(iid: str) -> str | None:
        dst = a.out / iid[:2] / f"{iid}.jpg"
        if dst.exists():
            return iid
        try:
            body = s3.get_object(Bucket=BUCKET, Key=f"{a.split}/{iid}.jpg")["Body"].read()
            im = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
            w, h = im.size
            s = a.size / min(w, h)
            if s < 1:
                im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
            dst.parent.mkdir(exist_ok=True)
            tmp = dst.with_suffix(".tmp")  # atomic write; ImageFolder ignores .tmp
            im.save(tmp, format="JPEG", quality=90)
            tmp.rename(dst)
            return iid
        except Exception:
            return None

    saved, n_fail, pos, chunk = [], 0, 0, a.threads * 8
    with cf.ThreadPoolExecutor(a.threads) as pool, tqdm(total=a.n) as bar:
        while len(saved) < a.n and pos < len(ids):
            batch = ids[pos : pos + chunk]
            pos += len(batch)
            for r in pool.map(fetch, batch):
                if r is None:
                    n_fail += 1
                elif len(saved) < a.n:
                    saved.append(r)
                    bar.update(1)
                else:  # overshoot from the last chunk: drop extras so the folder has exactly n
                    (a.out / r[:2] / f"{r}.jpg").unlink(missing_ok=True)

    (a.out.parent / f"{a.out.name}_ids.txt").write_text("\n".join(saved) + "\n")
    print(f"Saved {len(saved):,} images, {n_fail:,} failures.")


if __name__ == "__main__":
    main()
