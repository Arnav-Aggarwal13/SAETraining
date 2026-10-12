"""Sweep 3: Matryoshka. The four best configs so far (runs 0jhp03qu, 88oufjem, cs0vbj13, s8nya5no), each retrained with n_prefixes 5 and 10 (the earlier runs are n_prefixes=1, a plain SAE). Everything else is identical, including 125M tokens.

Each SWEEP_GROUP (env var, set by slurm/sweep3.sbatch's array index) is one job. The two 26k SAEs share a job; each 53k SAE gets its own, since two 53k SAEs ran out of memory on the 48 GB GPU in sweep 1.

Everything swept is spelled out in full, so none of lr / d_sae / top_k / alpha / dead_threshold_tokens / n_prefixes may be passed on the command line (CLI values override the sweep file).
"""


def make_cfgs() -> list[dict]:
    import os

    def cfg(d_sae, lr, alpha, dead_tokens, n_prefixes) -> dict:
        return {
            "lr": lr,
            "tags": ["matryoshka"],
            "sae": {
                "d_sae": d_sae,
                "activation": {"top_k": 128, "aux": {"alpha": alpha}},
            },
            "objective": {
                "n_prefixes": n_prefixes,
                "dead_threshold_tokens": dead_tokens,
            },
        }

    # (d_sae, lr, alpha, dead_threshold_tokens) of the four reference runs.
    s8nya5no = (26624, 4e-4, 1 / 32, 10_000_000)
    run_88oufjem = (53248, 4e-4, 1 / 32, 10_000_000)
    run_cs0vbj13 = (53248, 4e-4, 1 / 8, 2_000_000)
    run_0jhp03qu = (53248, 1e-3, 1 / 32, 2_000_000)

    groups = [
        [cfg(*s8nya5no, 5), cfg(*s8nya5no, 10)],
        [cfg(*run_88oufjem, 5)],
        [cfg(*run_88oufjem, 10)],
        [cfg(*run_cs0vbj13, 5)],
        [cfg(*run_cs0vbj13, 10)],
        [cfg(*run_0jhp03qu, 5)],
        [cfg(*run_0jhp03qu, 10)],
    ]
    return groups[int(os.environ["SWEEP_GROUP"])]
