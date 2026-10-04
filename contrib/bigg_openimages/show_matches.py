"""Contact sheet of NSD / Open Images near-duplicate pairs from dedup_nsd.py's matches.csv.

Rows are sim percentiles (--p-lo to --p-hi, 1 percentile wide each, computed over the sims in matches.csv). Each row shows --n-per pairs, NSD image on the left and Open Images thumbnail on the right.

    uv run python contrib/bigg_openimages/show_matches.py --matches $OI_DIR/quarantine_train/matches.csv --nsd-hdf5 $NSD_HDF5 --out match_samples.png
"""

import argparse
import pathlib

import beartype
import h5py
import numpy as np
import polars as pl
from PIL import Image, ImageDraw

TILE = 160
LABEL_H = 14


@beartype.beartype
def get_oi_fpath(path: str, quarantine_dpath: pathlib.Path) -> pathlib.Path:
    """The dedup script moves matched files flat into the quarantine dir, so check both places."""
    fpath = pathlib.Path(path)
    if fpath.exists():
        return fpath
    moved_fpath = quarantine_dpath / fpath.name
    assert moved_fpath.exists(), f"Neither '{fpath}' nor '{moved_fpath}' exists."
    return moved_fpath


@beartype.beartype
def make_tile(im: Image.Image) -> Image.Image:
    return im.convert("RGB").resize((TILE, TILE), Image.BICUBIC)


@beartype.beartype
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=pathlib.Path, required=True)
    p.add_argument("--nsd-hdf5", type=pathlib.Path, required=True)
    p.add_argument("--nsd-key", default="imgBrick")
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--p-lo", type=int, default=90)
    p.add_argument("--p-hi", type=int, default=99)
    p.add_argument("--n-per", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    assert 0 <= a.p_lo <= a.p_hi <= 99, f"{a.p_lo=} {a.p_hi=}"

    df = pl.read_csv(a.matches)
    sims = df["sim"].to_numpy()
    edges = np.percentile(sims, np.arange(a.p_lo, a.p_hi + 2).clip(max=100))
    rng = np.random.default_rng(a.seed)
    quarantine_dpath = a.matches.parent

    n_rows = a.p_hi - a.p_lo + 1
    pair_w = 2 * TILE
    sheet = Image.new("RGB", (a.n_per * pair_w, n_rows * (TILE + LABEL_H)), "white")
    draw = ImageDraw.Draw(sheet)

    with h5py.File(a.nsd_hdf5, "r") as fd:
        brick = fd[a.nsd_key]
        for row, pct in enumerate(range(a.p_lo, a.p_hi + 1)):
            lo, hi = edges[row], edges[row + 1]
            # The top band includes its upper edge (the max sim).
            in_band = (sims >= lo) & ((sims <= hi) if pct == 99 else (sims < hi))
            band_i = np.flatnonzero(in_band)
            assert len(band_i) > 0, f"Empty band p{pct} [{lo}, {hi}]"
            picked_i = np.sort(rng.choice(band_i, size=min(a.n_per, len(band_i)), replace=False))
            y = row * (TILE + LABEL_H)
            for col, i in enumerate(picked_i):
                r = df.row(int(i), named=True)
                x = col * pair_w
                sheet.paste(make_tile(Image.fromarray(brick[r["nsd_index"]])), (x, y + LABEL_H))
                oi = Image.open(get_oi_fpath(r["path"], quarantine_dpath))
                sheet.paste(make_tile(oi), (x + TILE, y + LABEL_H))
                draw.text((x + 2, y + 1), f"p{pct} sim={r['sim']:.4f} nsd={r['nsd_index']}", fill="black")

    sheet.save(a.out)
    print(f"Wrote {a.out} ({n_rows} rows x {a.n_per} pairs). Left=NSD, right=Open Images.")


if __name__ == "__main__":
    main()
