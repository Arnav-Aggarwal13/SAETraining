import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import pathlib

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    return mo, np, os, pathlib, plt, torch


@app.cell
def _(mo):
    mo.md("""
    # SAE latent explorer

    Two jobs:

    1. **Pick thresholds.** After `inference.sbatch` finishes, look at how often (frequency) and how strongly (value) each latent fires, and choose ranges that drop dead, dense and weak latents. The notebook prints the exact command to pass those ranges to `visuals.sbatch`.
    2. **Browse results.** After `visuals.sbatch` finishes, flip through the saved top-activating images for each latent.

    Frequency is the log10 fraction of tokens a latent fires on (-3 means 0.1% of tokens). Value is the log10 mean activation when it fires. You usually want latents that fire rarely and strongly.

    Start with the two dropdowns below.
    """)
    return


@app.cell
def _(mo, os, pathlib):
    runs_dpath = pathlib.Path(
        os.environ.get("RUNS_ROOT", "~/sae_runs/saev/runs")
    ).expanduser()
    run_ids = sorted(os.listdir(runs_dpath)) if runs_dpath.is_dir() else []
    run_dropdown = mo.ui.dropdown(run_ids, label="Run:")
    return run_dropdown, runs_dpath


@app.cell
def _(mo, os, run_dropdown, runs_dpath):
    mo.stop(
        run_dropdown.value is None,
        mo.vstack([run_dropdown, mo.md(f"Pick a run (listing `{runs_dpath}`).")]),
    )
    inference_dpath = runs_dpath / run_dropdown.value / "inference"
    hashes = sorted(os.listdir(inference_dpath)) if inference_dpath.is_dir() else []
    hash_dropdown = mo.ui.dropdown(
        hashes, value=hashes[0] if len(hashes) == 1 else None, label="Shards hash:"
    )
    mo.hstack([run_dropdown, hash_dropdown], justify="start")
    return hash_dropdown, inference_dpath


@app.cell
def _(hash_dropdown, inference_dpath, mo, torch):
    mo.stop(
        hash_dropdown.value is None,
        mo.md("Pick the shards hash (the folder `inference.sbatch` wrote into)."),
    )
    acts_dpath = inference_dpath / hash_dropdown.value
    _missing = [
        n for n in ("sparsity.pt", "mean_values.pt") if not (acts_dpath / n).exists()
    ]
    mo.stop(
        len(_missing) > 0,
        mo.md(f"Missing {_missing} in `{acts_dpath}`. Wait for `inference.sbatch`."),
    )
    sparsity_s = torch.load(
        acts_dpath / "sparsity.pt", weights_only=True, map_location="cpu"
    ).numpy()
    mean_value_s = torch.load(
        acts_dpath / "mean_values.pt", weights_only=True, map_location="cpu"
    ).numpy()
    assert sparsity_s.shape == mean_value_s.shape
    return acts_dpath, mean_value_s, sparsity_s


@app.cell
def _(mean_value_s, np, sparsity_s):
    # Dead latents (never fired) have no defined log, so they are NaN and never pass a range filter.
    alive_s = (sparsity_s > 0) & (mean_value_s > 0)
    log_freq_s = np.full(sparsity_s.shape, np.nan)
    log_freq_s[alive_s] = np.log10(sparsity_s[alive_s])
    log_value_s = np.full(mean_value_s.shape, np.nan)
    log_value_s[alive_s] = np.log10(mean_value_s[alive_s])
    return alive_s, log_freq_s, log_value_s


@app.cell
def _(alive_s, log_freq_s, log_value_s, mo, plt):
    _fig, (_ax_f, _ax_v) = plt.subplots(1, 2, figsize=(10, 3.5))
    _ax_f.hist(log_freq_s[alive_s], bins=100)
    _ax_f.set_xlabel("log10 frequency")
    _ax_f.set_ylabel("n latents")
    _ax_v.hist(log_value_s[alive_s], bins=100)
    _ax_v.set_xlabel("log10 mean value when firing")
    _fig.tight_layout()
    mo.vstack([
        mo.md(
            f"**{len(alive_s)} latents total, {int((~alive_s).sum())} never fired (dead).** Histograms cover the {int(alive_s.sum())} that did."
        ),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    freq_slider = mo.ui.range_slider(
        start=-8,
        stop=0,
        step=0.1,
        value=[-6, -1],
        label="log10 frequency",
        full_width=True,
    )
    value_slider = mo.ui.range_slider(
        start=-4,
        stop=3,
        step=0.1,
        value=[-4, 3],
        label="log10 mean value",
        full_width=True,
    )
    return freq_slider, value_slider


@app.cell
def _(freq_slider, mo, value_slider):
    mo.md(f"""
    ## Choose thresholds

    Drag the ends of each slider. Blue latents are kept, red are filtered out.

    {freq_slider}

    {value_slider}
    """)
    return


@app.cell
def _(
    alive_s,
    freq_slider,
    log_freq_s,
    log_value_s,
    mo,
    plt,
    run_dropdown,
    value_slider,
):
    _f_lo, _f_hi = freq_slider.value
    _v_lo, _v_hi = value_slider.value
    # Same strict inequalities as tdiscovery/visuals.py, so counts match what the job selects from.
    in_range_s = alive_s & (
        (_f_lo < log_freq_s)
        & (log_freq_s < _f_hi)
        & (_v_lo < log_value_s)
        & (log_value_s < _v_hi)
    )

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _out_s = alive_s & ~in_range_s
    _ax.scatter(
        log_freq_s[in_range_s],
        log_value_s[in_range_s],
        marker=".",
        alpha=0.1,
        color="tab:blue",
        label=f"Kept ({int(in_range_s.sum())})",
    )
    _ax.scatter(
        log_freq_s[_out_s],
        log_value_s[_out_s],
        marker=".",
        alpha=0.1,
        color="tab:red",
        label=f"Filtered ({int(_out_s.sum())})",
    )
    for _x in (_f_lo, _f_hi):
        _ax.axvline(_x, linewidth=0.5, color="tab:red")
    for _y in (_v_lo, _v_hi):
        _ax.axhline(_y, linewidth=0.5, color="tab:red")
    _ax.set_xlabel("log10 frequency")
    _ax.set_ylabel("log10 mean value")
    _ax.legend(loc="upper right")

    _cmd = f'LOG_FREQ="{_f_lo} {_f_hi}" LOG_VALUE="{_v_lo} {_v_hi}" RUN={run_dropdown.value} sbatch contrib/bigg_openimages/slurm/visuals.sbatch'
    mo.vstack([
        mo.as_html(_fig),
        mo.md(
            f"Run this from the repo root to render {int(in_range_s.sum())} candidate latents (the job samples N_LATENTS of them at random):\n\n```bash\n{_cmd}\n```"
        ),
    ])
    return (in_range_s,)


@app.cell
def _(acts_dpath, mo, os):
    images_dpath = acts_dpath / "images"
    mo.stop(
        not images_dpath.is_dir(),
        mo.md(
            f"No `{images_dpath}` yet. Run `visuals.sbatch` with the command above, then rerun this cell."
        ),
    )
    saved_latents = sorted(int(n) for n in os.listdir(images_dpath) if n.isdigit())
    sort_dropdown = mo.ui.dropdown(
        ["latent id", "frequency (rare first)", "value (strong first)"],
        value="latent id",
        label="Sort by:",
    )
    only_in_range = mo.ui.checkbox(label="Only latents inside the ranges above")
    show_orig = mo.ui.checkbox(label="Also show un-highlighted images")
    n_cols_slider = mo.ui.slider(1, 8, value=5, label="Columns:")
    mo.vstack([
        mo.md("## Browse saved latents"),
        mo.hstack(
            [sort_dropdown, only_in_range, show_orig, n_cols_slider], justify="start"
        ),
    ])
    return (
        images_dpath,
        n_cols_slider,
        only_in_range,
        saved_latents,
        show_orig,
        sort_dropdown,
    )


@app.cell
def _(
    in_range_s,
    log_freq_s,
    log_value_s,
    mo,
    np,
    only_in_range,
    saved_latents,
    sort_dropdown,
):
    _ids = np.array(saved_latents)
    if only_in_range.value:
        _ids = _ids[in_range_s[_ids]]
    if sort_dropdown.value == "frequency (rare first)":
        _ids = _ids[np.argsort(log_freq_s[_ids])]
    elif sort_dropdown.value == "value (strong first)":
        _ids = _ids[np.argsort(-log_value_s[_ids])]
    browse_ids = _ids.tolist()
    mo.stop(len(browse_ids) == 0, mo.md("No saved latents match the filters."))

    pos_slider = mo.ui.slider(
        start=0,
        stop=max(len(browse_ids) - 1, 1),
        value=0,
        label=f"Position (0 to {len(browse_ids) - 1})",
        full_width=True,
        debounce=True,
    )
    pos_slider
    return browse_ids, pos_slider


@app.cell
def _(
    browse_ids,
    images_dpath,
    log_freq_s,
    log_value_s,
    mo,
    n_cols_slider,
    pos_slider,
    show_orig,
):
    _i = min(pos_slider.value, len(browse_ids) - 1)
    _f = browse_ids[_i]
    _info = mo.md(
        f"**Latent {_f}** ({_i + 1}/{len(browse_ids)}) | fires on {10 ** log_freq_s[_f] * 100:.4f}% of tokens | mean value {10 ** log_value_s[_f]:.3f}"
    )

    _tiles = []
    _j = 0
    while (images_dpath / str(_f) / f"{_j}_sae_img.png").exists():
        _tile = [mo.image(str(images_dpath / str(_f) / f"{_j}_sae_img.png"))]
        _orig_fpath = images_dpath / str(_f) / f"{_j}_img.png"
        if show_orig.value and _orig_fpath.exists():
            _tile.append(mo.image(str(_orig_fpath)))
        _tiles.append(mo.vstack(_tile))
        _j += 1

    _n_cols = n_cols_slider.value
    _rows = [
        mo.hstack(_tiles[_k : _k + _n_cols], justify="start")
        for _k in range(0, len(_tiles), _n_cols)
    ]
    mo.vstack([_info, *_rows])
    return


if __name__ == "__main__":
    app.run()
