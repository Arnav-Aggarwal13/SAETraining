"""Sweep 1: fix reconstruction (nMSE 0.26 train / 0.29 val at k=64, 16x) and latent death (dead/almost dead ~3%, 1,100 dense, aux loss climbing).

Baseline (already trained, run 9za00zoc): d_sae 26624, top_k 64, lr 4e-4, AuxK alpha 1/32, dead_threshold_tokens 10M.

Each SWEEP_GROUP (env var, set by slurm/sweep1.sbatch's array index) is one job whose SAEs train in parallel off a single data stream. Groups are sized to fit one 48 GB GPU, and all SAEs in a group share d_sae (it cannot be parallelized across values).

Everything swept is spelled out in full here, so none of lr / d_sae / top_k / alpha / dead_threshold_tokens may be passed on the command line (CLI values override the sweep file).
"""


def make_cfgs() -> list[dict]:
    import os

    def cfg(
        d_sae=26624, top_k=64, lr=4e-4, alpha=1 / 32, dead_tokens=10_000_000
    ) -> dict:
        return {
            "lr": lr,
            "tags": ["sweep1"],
            "sae": {
                "d_sae": d_sae,
                "activation": {"top_k": top_k, "aux": {"alpha": alpha}},
            },
            "objective": {"dead_threshold_tokens": dead_tokens},
        }

    groups = [
        # 0: capacity at 16x (k) and optimizer (lr) and faster dead-latent revival.
        [cfg(top_k=128), cfg(lr=1e-4), cfg(dead_tokens=2_000_000)],
        # 1: stronger AuxK, alone and combined with faster revival.
        [cfg(alpha=1 / 8), cfg(alpha=1 / 8, dead_tokens=2_000_000)],
        # 2: capacity at 32x, k=64. Alone: two 32x SAEs in one job ran out of memory on the 48 GB GPU.
        [cfg(d_sae=53248)],
        # 3: everything that is expected to help, at 32x.
        [cfg(d_sae=53248, top_k=128, alpha=1 / 8, dead_tokens=2_000_000)],
        # 4: capacity at 32x, k=128, alone for the same reason as group 2.
        [cfg(d_sae=53248, top_k=128)],
    ]
    return groups[int(os.environ["SWEEP_GROUP"])]
