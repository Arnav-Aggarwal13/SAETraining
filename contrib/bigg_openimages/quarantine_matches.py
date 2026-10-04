"""Move thumbnails listed in dedup_nsd.py's matches.csv with sim >= --thresh into the quarantine dir (flat, same layout as dedup_nsd.py).

Use this after a --dry-run of dedup_nsd.py, to pick the final threshold from the saved matches.csv without re-embedding anything. Safe to re-run: files already moved are skipped.

    uv run python contrib/bigg_openimages/quarantine_matches.py --matches $OI_DIR/quarantine_train/matches.csv --thresh 0.93 --dry-run
"""

import argparse
import pathlib
import shutil

import beartype
import polars as pl


@beartype.beartype
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matches", type=pathlib.Path, required=True)
    p.add_argument("--thresh", type=float, default=0.93)
    p.add_argument("--dry-run", action="store_true", help="Only report counts; move nothing.")
    a = p.parse_args()
    assert a.matches.exists(), f"Missing '{a.matches}'."

    df = pl.read_csv(a.matches)
    min_sim = df["sim"].min()
    assert min_sim <= a.thresh, f"matches.csv only covers sim >= {min_sim:.4f}, so --thresh {a.thresh} is not fully covered."
    quarantine_dpath = a.matches.parent
    hits = df.filter(pl.col("sim") >= a.thresh)["path"].to_list()

    n_moved = n_already = 0
    for path in hits:
        src_fpath = pathlib.Path(path)
        dst_fpath = quarantine_dpath / src_fpath.name
        if not src_fpath.exists():
            assert dst_fpath.exists(), f"'{src_fpath}' is neither in place nor in quarantine."
            n_already += 1
            continue
        n_moved += 1
        if a.dry_run:
            continue
        shutil.move(src_fpath, dst_fpath)

    verb = "Would move" if a.dry_run else "Moved"
    print(f"{len(hits):,} of {len(df):,} pairs >= {a.thresh}. {verb} {n_moved:,} to {quarantine_dpath} ({n_already:,} already there).")


if __name__ == "__main__":
    main()
