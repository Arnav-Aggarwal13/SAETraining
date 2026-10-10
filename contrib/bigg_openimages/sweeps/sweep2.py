"""Sweep 2: data scaling. Same recipe as the baseline run 9za00zoc (d_sae 26624, top_k 64, lr 4e-4, AuxK alpha 1/32, dead threshold 10M, n_lr_warmup 500), only n_train varies.

Question: does nMSE keep improving with more tokens? Baseline (125M tokens, ~1 epoch of 127.6M): eval nMSE 0.2868. If 64M is about as good as 125M, collecting more images will not help.

n_train cannot be parallelized across values in saev, so each size is its own job (SWEEP_GROUP, set by slurm/sweep2.sbatch's array index). The CLI must not pass --n-train, since CLI values override the sweep file.
"""


def make_cfgs() -> list[dict]:
    import os

    n_trains = [32_000_000, 64_000_000]
    return [{"n_train": n_trains[int(os.environ["SWEEP_GROUP"])], "tags": ["sweep2"]}]
