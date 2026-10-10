"""Where does an SAE's reconstruction error live, relative to the structure of the data?

Two passes over the val tokens (patch tokens only, one layer):

1. Data: mean, covariance, PCA spectrum, per-dimension variance, token-norm distribution.
2. SAE: for each token, the residual x - x_hat, which is split by PCA direction, by patch position in the image, by token-norm decile, and by how concentrated it is in a few tokens.

Writes variance_analysis.json and variance_analysis.png under <run>/inference/<shards hash>/variance_analysis/.

    uv run --no-sync python contrib/bigg_openimages/variance_analysis.py --run $RUNS_ROOT/9za00zoc --shards $SHARDS_ROOT/$VAL_HASH
"""

import argparse
import json
import pathlib

import beartype
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saev import disk, helpers, nn
from saev.data import Metadata, OrderedConfig, OrderedDataLoader

PC_CUTS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
VAR_LEVELS = (0.5, 0.9, 0.95, 0.99)
N_NORM_BINS = 10


@beartype.beartype
def make_loader(
    shards_dpath: pathlib.Path, layer: int, batch_size: int
) -> tuple[Metadata, OrderedDataLoader]:
    md = Metadata.load(shards_dpath)
    tokens_per_example = md.content_tokens_per_example
    batch_size = batch_size // tokens_per_example * tokens_per_example
    cfg = OrderedConfig(shards=shards_dpath, layer=layer, batch_size=batch_size)
    return md, OrderedDataLoader(cfg)


@beartype.beartype
@torch.inference_mode()
def get_data_stats(loader: OrderedDataLoader, device: str) -> dict[str, torch.Tensor]:
    """Pass 1: first and second moments in float64, plus every token norm."""
    sum_d = sum_xx_dd = None
    n_tokens = 0
    norms = []
    for batch in helpers.progress(loader, desc="pass 1"):
        x_bd = batch["act"].to(device, torch.float64)
        if sum_d is None:
            d = x_bd.shape[1]
            sum_d = torch.zeros(d, dtype=torch.float64, device=device)
            sum_xx_dd = torch.zeros(d, d, dtype=torch.float64, device=device)
        sum_d += x_bd.sum(0)
        sum_xx_dd += x_bd.T @ x_bd
        n_tokens += len(x_bd)
        norms.append(x_bd.norm(dim=1).float().cpu())

    assert sum_d is not None and sum_xx_dd is not None and n_tokens > 0
    mean_d = sum_d / n_tokens
    cov_dd = sum_xx_dd / n_tokens - torch.outer(mean_d, mean_d)
    # eigh returns ascending eigenvalues, so flip to put the biggest PC first.
    eig_d, vecs_dd = torch.linalg.eigh(cov_dd)
    eig_d = eig_d.flip(0).clamp(min=0)
    vecs_dd = vecs_dd.flip(1)
    assert torch.allclose(eig_d.sum(), cov_dd.diagonal().sum()), (
        "eigenvalues must sum to the trace"
    )
    return {
        "n_tokens": torch.tensor(n_tokens),
        "mean_d": mean_d,
        "var_dim_d": cov_dd.diagonal().clone(),
        "eig_d": eig_d,
        "vecs_dd": vecs_dd,
        "norms": torch.cat(norms),
    }


@beartype.beartype
@torch.inference_mode()
def get_residual_stats(
    loader: OrderedDataLoader,
    sae: nn.SparseAutoencoder,
    stats: dict[str, torch.Tensor],
    tokens_per_example: int,
    device: str,
) -> dict[str, torch.Tensor]:
    """Pass 2: split the SAE residual by PC, patch position, and norm bin."""
    mean_d, vecs_dd = stats["mean_d"], stats["vecs_dd"]
    d = len(mean_d)
    q = torch.linspace(0, 1, N_NORM_BINS + 1)[1:-1]
    edges = torch.quantile(
        stats["norms"][:: max(1, len(stats["norms"]) // 1_000_000)], q
    ).to(device)

    sse_pc_d = torch.zeros(d, dtype=torch.float64, device=device)
    sse_pos, base_pos = (
        torch.zeros(tokens_per_example, dtype=torch.float64, device=device)
        for _ in range(2)
    )
    sse_bin, base_bin = (
        torch.zeros(N_NORM_BINS, dtype=torch.float64, device=device) for _ in range(2)
    )
    n_pos = torch.zeros(tokens_per_example, dtype=torch.float64, device=device)
    n_bin = torch.zeros(N_NORM_BINS, dtype=torch.float64, device=device)
    sse_tok = []
    for batch in helpers.progress(loader, desc="pass 2"):
        x_bd = batch["act"].to(device)
        x_hat_bd = sae(x_bd).x_hats[:, -1, :]
        res_bd = (x_bd - x_hat_bd).to(torch.float64)
        base_b = (x_bd.to(torch.float64) - mean_d).pow(2).sum(1)
        res_b = res_bd.pow(2).sum(1)

        sse_pc_d += (res_bd @ vecs_dd).pow(2).sum(0)
        pos_b = batch["token_idx"].to(device)
        assert pos_b.min() >= 0 and pos_b.max() < tokens_per_example
        sse_pos.index_add_(0, pos_b, res_b)
        base_pos.index_add_(0, pos_b, base_b)
        n_pos.index_add_(0, pos_b, torch.ones_like(res_b))
        bin_b = torch.bucketize(x_bd.norm(dim=1), edges)
        sse_bin.index_add_(0, bin_b, res_b)
        base_bin.index_add_(0, bin_b, base_b)
        n_bin.index_add_(0, bin_b, torch.ones_like(res_b))
        sse_tok.append(res_b.float().cpu())

    return {
        "sse_pc_d": sse_pc_d,
        "sse_pos": sse_pos,
        "base_pos": base_pos,
        "n_pos": n_pos,
        "sse_bin": sse_bin,
        "base_bin": base_bin,
        "n_bin": n_bin,
        "sse_tok": torch.cat(sse_tok),
    }


@beartype.beartype
def summarize(
    stats: dict[str, torch.Tensor],
    res: dict[str, torch.Tensor],
    tokens_per_example: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    n = int(stats["n_tokens"])
    eig = stats["eig_d"].cpu().numpy()
    var_dim = stats["var_dim_d"].cpu().numpy()
    total_var = eig.sum()
    res_pc = res["sse_pc_d"].cpu().numpy() / n
    total_res = res_pc.sum()
    cum_var = np.cumsum(eig) / total_var
    cum_res = np.cumsum(res_pc) / total_res
    norms = stats["norms"].numpy()
    sse_tok = res["sse_tok"].numpy()
    msg = f"{sse_tok.shape[0]} != {n}"
    assert sse_tok.shape[0] == n, msg

    g = int(np.sqrt(tokens_per_example))
    assert g * g == tokens_per_example
    nmse_pos = (res["sse_pos"] / res["base_pos"]).cpu().numpy().reshape(g, g)
    nmse_bin = (res["sse_bin"] / res["base_bin"]).cpu().numpy()
    sse_share_bin = (res["sse_bin"] / res["sse_bin"].sum()).cpu().numpy()
    sorted_sse = np.sort(sse_tok)[::-1]
    cum_tok = np.cumsum(sorted_sse) / sorted_sse.sum()

    per_cut = []
    for m in PC_CUTS:
        per_cut.append({
            "top_pcs": m,
            "data_var_share": float(cum_var[m - 1]),
            "resid_sse_share": float(cum_res[m - 1]),
            "nmse_within_top": float(res_pc[:m].sum() / eig[:m].sum()),
            "nmse_within_rest": float(res_pc[m:].sum() / eig[m:].sum()),
        })

    summary = {
        "n_tokens": n,
        "d_model": len(eig),
        "overall_nmse": float(total_res / total_var),
        "mean_share_of_second_moment": float(
            (
                stats["mean_d"].pow(2).sum()
                / (stats["mean_d"].pow(2).sum() + stats["eig_d"].sum())
            ).cpu()
        ),
        "token_norm_percentiles": {
            str(p): float(np.percentile(norms, p)) for p in (1, 50, 90, 99, 99.9, 100)
        },
        "top_dim_variance_share": {
            str(k): float(np.sort(var_dim)[::-1][:k].sum() / var_dim.sum())
            for k in (1, 5, 10, 50)
        },
        "n_pcs_for_variance": {
            str(v): int(np.searchsorted(cum_var, v) + 1) for v in VAR_LEVELS
        },
        "participation_ratio": float(total_var**2 / (eig**2).sum()),
        "per_pc_cut": per_cut,
        "nmse_by_position": {
            "min": float(nmse_pos.min()),
            "max": float(nmse_pos.max()),
            "mean": float(nmse_pos.mean()),
            "grid": nmse_pos.round(4).tolist(),
        },
        "nmse_by_norm_decile": nmse_bin.round(4).tolist(),
        "sse_share_by_norm_decile": sse_share_bin.round(4).tolist(),
        "sse_share_top_1pct_tokens": float(cum_tok[int(0.01 * n) - 1]),
        "sse_share_top_10pct_tokens": float(cum_tok[int(0.1 * n) - 1]),
    }
    arrays = {
        "cum_var": cum_var,
        "cum_res": cum_res,
        "r2_pc": 1 - res_pc / np.maximum(eig, 1e-12),
        "eig": eig,
        "nmse_pos": nmse_pos,
        "nmse_bin": nmse_bin,
        "sse_share_bin": sse_share_bin,
        "cum_tok": cum_tok,
        "norms": norms,
    }
    return summary, arrays


@beartype.beartype
def plot(summary: dict, a: dict[str, np.ndarray], out_fpath: pathlib.Path):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    pcs = np.arange(1, len(a["cum_var"]) + 1)

    ax[0, 0].plot(pcs, a["cum_var"], label="data variance")
    ax[0, 0].plot(pcs, a["cum_res"], label="SAE residual (SSE)")
    ax[0, 0].plot(pcs, pcs / len(pcs), "k:", lw=0.8, label="uniform")
    ax[0, 0].set_xscale("log")
    ax[0, 0].set_title("Cumulative share by top-m PCs")
    ax[0, 0].set_xlabel("m (PCs, biggest first)")
    ax[0, 0].legend()

    ax[0, 1].plot(pcs, a["r2_pc"], lw=0.7)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_ylim(-0.2, 1.05)
    ax[0, 1].axhline(0, color="k", lw=0.5)
    ax[0, 1].set_title("SAE R^2 along each PC (1 = perfect)")
    ax[0, 1].set_xlabel("PC index")

    im = ax[0, 2].imshow(a["nmse_pos"], cmap="viridis")
    ax[0, 2].set_title("nMSE by patch position (row, col)")
    fig.colorbar(im, ax=ax[0, 2])

    deciles = np.arange(N_NORM_BINS)
    ax[1, 0].bar(deciles, a["nmse_bin"])
    ax[1, 0].set_xlabel("token-norm decile (0 = smallest)")
    ax[1, 0].set_ylabel("nMSE in bin")
    twin = ax[1, 0].twinx()
    twin.plot(deciles, a["sse_share_bin"], "r.-")
    twin.set_ylabel("share of total SSE (red)")
    ax[1, 0].set_title("Error by token norm")

    frac = np.arange(1, len(a["cum_tok"]) + 1) / len(a["cum_tok"])
    ax[1, 1].plot(
        frac[:: max(1, len(frac) // 2000)], a["cum_tok"][:: max(1, len(frac) // 2000)]
    )
    ax[1, 1].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[1, 1].set_xlabel("fraction of tokens (worst first)")
    ax[1, 1].set_ylabel("cumulative share of SSE")
    ax[1, 1].set_title("How concentrated is the error?")

    ax[1, 2].hist(a["norms"], bins=200)
    ax[1, 2].set_yscale("log")
    ax[1, 2].set_title("Token norm distribution")
    ax[1, 2].set_xlabel("||x||")

    fig.suptitle(
        f"overall nMSE {summary['overall_nmse']:.4f} on {summary['n_tokens']:,} val tokens"
    )
    fig.tight_layout()
    fig.savefig(out_fpath, dpi=110)


@beartype.beartype
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run", type=pathlib.Path, required=True, help="Run dir: $RUNS_ROOT/<run id>."
    )
    p.add_argument("--shards", type=pathlib.Path, required=True, help="Val shards dir.")
    p.add_argument("--layer", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=16384)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    run = disk.Run(a.run)
    md, loader = make_loader(a.shards, a.layer, a.batch_size)
    sae = nn.load(run.ckpt).to(a.device).eval()

    stats = get_data_stats(loader, a.device)
    _, loader = make_loader(a.shards, a.layer, a.batch_size)
    res = get_residual_stats(
        loader, sae, stats, md.content_tokens_per_example, a.device
    )
    summary, arrays = summarize(stats, res, md.content_tokens_per_example)

    out_dpath = run.inference / md.hash / "variance_analysis"
    out_dpath.mkdir(parents=True, exist_ok=True)
    with open(out_dpath / "variance_analysis.json", "w") as fd:
        json.dump(summary, fd, indent=2)
    plot(summary, arrays, out_dpath / "variance_analysis.png")

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("nmse_by_position",)},
            indent=2,
        )
    )
    print(
        "nmse_by_position min/mean/max:",
        {k: summary["nmse_by_position"][k] for k in ("min", "mean", "max")},
    )
    print(f"Wrote {out_dpath}")


if __name__ == "__main__":
    main()
