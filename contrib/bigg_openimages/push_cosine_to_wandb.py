"""Copy cosine similarity from saev inference metrics.json files into the matching wandb runs' summaries.

Adds `eval/cosine_uncentered` and `eval/cosine_centered` to each existing run (saev run ids are the wandb run ids), so they sit next to `eval/normalized_mse` and can be compared in the runs table. Nothing else in the run is touched.

As a guard, a run is only updated if the normalized MSE in its metrics.json matches the `eval/normalized_mse` already in wandb (same model, same eval conditions). Dry run by default; pass --apply to write.

    uv run --no-sync python contrib/bigg_openimages/push_cosine_to_wandb.py --runs-root $RUNS_ROOT --val-hash $VAL_HASH
"""

import argparse
import json
import pathlib

import wandb

KEYS = ("cosine_uncentered", "cosine_centered")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=pathlib.Path, required=True)
    p.add_argument(
        "--val-hash",
        required=True,
        help="Hash dir of the val shards under <run>/inference/.",
    )
    p.add_argument(
        "--project",
        default="arnav813-massachusetts-institute-of-technology/bigg-openimages-sae",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Max |nMSE(metrics.json) - nMSE(wandb)| to accept a run.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write to wandb. Without it, only print what would change.",
    )
    a = p.parse_args()

    n_updated = 0
    for run in wandb.Api(timeout=120).runs(a.project):
        fpath = a.runs_root / run.id / "inference" / a.val_hash / "metrics.json"
        if not fpath.exists():
            print(f"{run.id}: skip, no {fpath}")
            continue
        with open(fpath) as fd:
            metrics = json.load(fd)
        if any(metrics.get(key) is None for key in KEYS):
            print(
                f"{run.id}: skip, metrics.json has no cosine fields (rerun inference)"
            )
            continue
        wandb_nmse = run.summary.get("eval/normalized_mse")
        diff = abs(metrics["normalized_mse"] - wandb_nmse)
        if diff > a.tol:
            print(
                f"{run.id}: skip, nMSE differs from wandb by {diff:.2e} ({metrics['normalized_mse']:.5f} vs {wandb_nmse:.5f}); different eval conditions"
            )
            continue
        print(
            f"{run.id}: uncentered {metrics['cosine_uncentered']:.5f}, centered {metrics['cosine_centered']:.5f} (nMSE diff {diff:.1e})"
        )
        if not a.apply:
            continue
        for key in KEYS:
            run.summary[f"eval/{key}"] = metrics[key]
        run.summary.update()
        n_updated += 1
    print(
        f"Updated {n_updated} runs."
        if a.apply
        else "Dry run, nothing written. Pass --apply to write."
    )


if __name__ == "__main__":
    main()
