"""Score the retrained Fig. 4 surrogates on a stratified bin set of a different shape.

Fig. 4 bins its 100 bubbles on Wadell nonsphericity. That is one choice among
several, and a model can look good on it simply because that is the axis it was
displayed along. This builds an equivalent panel binned on a *different* shape
measure, drawn from bubbles the retrained models never saw, and scores the same
four series on it.

``--bin-by`` picks the stratification axis:

``wadell``    1 - Phi_VA, the axis Fig. 4 itself uses (for a fresh draw along it)
``inertia``   sqrt((I11/I00 - 1)^2 + (I22/I00 - 1)^2), elongation / flattening;
              this is the axis the supplement's ablation panel actually used
``chull``     1 - eta_V, the convex-hull volume deficit, i.e. concavity
``willmore``  1 - 4 pi / W_vertex, bending energy above the sphere's

or any one of the model's **eight input features**, binned on its raw value:
``i11_over_i00``, ``i22_over_i00``, ``non_sph_va``, ``non_sph_vm``,
``non_sph_w``, ``eta_V``, ``eta_A``, ``eta_M``. Those eight are not all oriented
the same way -- the convex-hull ratios approach 1 for a convex bubble, so their
panels read hard-to-easy -- which is what makes them worth plotting separately:
they show which features order the difficulty, and in which direction.
``non_sph_va`` and ``non_sph_w`` are the same scalars as ``wadell`` and
``willmore`` and select exactly the same bubbles. ``eta_V`` is ``chull``
negated, which reverses the bin order and, because the seed of each bin's
farthest-point walk is picked from the bin's own members, shifts the selection
slightly: the two agree on 93 of 100 bubbles rather than all of them.

The pool defaults to the **test split of the retrained models**, so every scored
bubble is out of sample: those models were trained with Fig. 4's 100 bubbles
pinned out and the remaining 9,900 split 70/15/15, and this draws from the 1,485
rows none of them touched.

Within each bin the bubbles are chosen by farthest-point sampling in the
standardized eight-feature space, seeded from the bubble nearest the bin's median
scalar. No RNG: the same dataset and the same axis give the same set every time.

Usage:
    python python/freq_model/NN/eval_alt_bin_sets.py --bin-by inertia
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_ROOT = _THIS_DIR.parents[1]
for _p in (_PYTHON_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from freq_model.NN.fit_shape_freq_model import FEATURE_COLS as FEATURE_COLS_8, load_xy  # noqa: E402
from freq_model.NN.nn_inference import load_nn_artifacts  # noqa: E402

RETRAIN_DIR = Path("python/freq_model/output/fig04_retrain")
FIG04_DIR = Path("results/experiments/fig04_per_bin_model_error")

# Composite axes, each 0 at a sphere and increasing with nonsphericity.
AXES = {
    "wadell": ("Wadell nonsphericity  $1-\\Phi_{VA}$", "non_sph_va"),
    "inertia": ("Inertia anisotropy", "inertia_aniso"),
    "chull": ("Convex-hull deficit  $1-\\eta_V$", "chull_deficit"),
    "willmore": ("Willmore excess  $1-4\\pi/W$", "non_sph_w"),
}

# The eight model features, each binned on its own raw value, ascending. Unlike
# the composite axes these are not all oriented the same way: the three
# ``non_sph_*`` complements and the two inertia ratios grow as a bubble departs
# from a sphere, while the three convex-hull ratios approach 1 for a convex one,
# so a panel binned on eta_V reads hard-to-easy rather than easy-to-hard. That is
# the point of binning on the raw feature: it shows which features actually order
# the difficulty, and in which direction.
FEATURE_AXIS_LABELS = {
    "i11_over_i00": "$I_{11}/I_{00}$",
    "i22_over_i00": "$I_{22}/I_{00}$",
    "non_sph_va": "$1-\\Phi_{VA}$",
    "non_sph_vm": "$1-\\Phi_{VM}$",
    "non_sph_w": "$1-4\\pi/W$",
    "eta_V": "$\\eta_V$",
    "eta_A": "$\\eta_A$",
    "eta_M": "$\\eta_M$",
}
for _f, _lbl in FEATURE_AXIS_LABELS.items():
    AXES[_f] = (f"{_lbl}   ({_f})", _f)


def _scalars(x8: np.ndarray, df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every candidate stratification axis: the eight raw features plus composites."""
    cols = {name: x8[:, i].astype(np.float64) for i, name in enumerate(FEATURE_COLS_8)}
    i11, i22 = cols["i11_over_i00"], cols["i22_over_i00"]
    cols["inertia_aniso"] = np.sqrt((i11 - 1.0) ** 2 + (i22 - 1.0) ** 2)
    cols["chull_deficit"] = 1.0 - cols["eta_V"]
    return cols


def _farthest_point_sample(xz: np.ndarray, k: int, seed_row: int) -> np.ndarray:
    """Indices of ``k`` rows of ``xz`` spread out by greedy farthest-point sampling."""
    if len(xz) <= k:
        return np.arange(len(xz))
    picked = [seed_row]
    d = np.linalg.norm(xz - xz[seed_row], axis=1)
    while len(picked) < k:
        nxt = int(np.argmax(d))
        picked.append(nxt)
        d = np.minimum(d, np.linalg.norm(xz - xz[nxt], axis=1))
    return np.array(picked, dtype=int)


def _predict(artifact_dir: Path, x: np.ndarray, f_strasberg: np.ndarray) -> np.ndarray:
    """Run one retrained artifact over ``x``; adds the baseline back when residual."""
    import torch

    cfg = json.loads((artifact_dir / "train_config.json").read_text(encoding="utf-8"))
    feature_cols = list(cfg["feature_cols"])
    kind = str(cfg["baseline_kind"])
    model, fs, ts, _ = load_nn_artifacts(artifact_dir, feature_cols, kind)

    if x.shape[1] != len(feature_cols):
        # the 6-feature variants drop the two leading inertia ratios
        x = x[:, 8 - len(feature_cols):]
    xn = fs.transform(x.astype(np.float32)).astype(np.float32)
    with torch.no_grad():
        y = model(torch.from_numpy(xn)).numpy().reshape(-1, 1)
    log_pred = ts.inverse_transform(y).flatten().astype(np.float64)
    if kind == "strasberg_log_residual":
        log_pred = log_pred + np.log(f_strasberg)
    elif kind != "none_log_direct":
        raise ValueError(f"unexpected baseline_kind {kind!r} in {artifact_dir}")
    return np.exp(log_pred)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=Path("dataset/bubble_gym/dataset_bubblegym_10k.csv"))
    p.add_argument("--bin-by", choices=tuple(AXES), required=True)
    p.add_argument("--retrain-dir", type=Path, default=RETRAIN_DIR)
    p.add_argument("--features", type=int, choices=(6, 8), default=8)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--per-bin", type=int, default=10)
    p.add_argument("--fig04-dir", type=Path, default=FIG04_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if args.out_dir is None:
        args.out_dir = args.fig04_dir / "alt_bin_sets" / args.bin_by
    args.out_dir.mkdir(parents=True, exist_ok=True)

    axis_label, scalar_name = AXES[args.bin_by]

    # ---- rows the retrained models never saw -----------------------------
    dirs = {o: args.retrain_dir / f"{args.features}feature_{o}" for o in ("residual", "direct")}
    splits = {o: json.loads((d / "split.json").read_text(encoding="utf-8"))
              for o, d in dirs.items()}
    pool_ids = set(splits["residual"]["mesh_id_test"])
    if pool_ids != set(splits["direct"]["mesh_id_test"]):
        raise SystemExit("the two retrains disagree on their test split; cannot share a pool")
    for o, s in splits.items():
        if pool_ids & (set(s["mesh_id_train"]) | set(s["mesh_id_val"])):
            raise SystemExit(f"{o}: test split overlaps train/val")

    x8, y_log, _zero, df = load_xy(args.dataset.resolve())
    mesh_id = df["mesh_id"].astype(str).to_numpy()
    in_pool = np.array([m in pool_ids for m in mesh_id], dtype=bool)
    if int(in_pool.sum()) != len(pool_ids):
        raise SystemExit("pool ids missing from the dataset after load_xy's filter")

    # Fig. 4's own meshes must not sneak back in through the pool.
    fig04_ids = set(pd.read_csv(args.fig04_dir / "selected_rows.csv")["mesh_id_10k"])
    if pool_ids & fig04_ids:
        raise SystemExit("pool overlaps Fig. 4's evaluation meshes")

    idx_pool = np.flatnonzero(in_pool)
    scal = _scalars(x8, df)[scalar_name][idx_pool]
    x_pool = x8[idx_pool]
    if not np.all(np.isfinite(scal)):
        raise SystemExit(f"{scalar_name} has non-finite values in the pool")

    # ---- 10 quantile bins, farthest-point sample within each -------------
    edges = np.quantile(scal, np.linspace(0.0, 1.0, args.bins + 1))
    edges[0], edges[-1] = scal.min(), scal.max()
    if np.any(np.diff(edges) <= 0):
        raise SystemExit(f"{scalar_name} quantile edges are not strictly increasing")

    mu, sd = x_pool.mean(0), x_pool.std(0)
    sd[sd == 0] = 1.0
    xz = (x_pool - mu) / sd

    chosen, chosen_bin = [], []
    for b in range(args.bins):
        lo, hi = edges[b], edges[b + 1]
        m = (scal >= lo) & (scal <= hi if b == args.bins - 1 else scal < hi)
        members = np.flatnonzero(m)
        if len(members) < args.per_bin:
            raise SystemExit(f"bin {b} has only {len(members)} candidates")
        seed_local = int(np.argmin(np.abs(scal[members] - np.median(scal[members]))))
        take = _farthest_point_sample(xz[members], args.per_bin, seed_local)
        chosen.extend(members[take].tolist())
        chosen_bin.extend([b] * args.per_bin)

    sel = idx_pool[np.array(chosen, dtype=int)]
    sel_bin = np.array(chosen_bin, dtype=int)
    if len(set(sel.tolist())) != len(sel):
        raise SystemExit("a bubble was selected twice")

    # ---- score the four series -------------------------------------------
    f_gt = np.exp(y_log[sel].astype(np.float64))
    gt_csv = df["frequency"].to_numpy(dtype=np.float64)[sel]
    if not np.allclose(f_gt, gt_csv, rtol=1e-6):
        raise SystemExit("ground truth mismatch between load_xy and the dataset column")
    f_gt = gt_csv

    f_str = df["f_strasberg"].to_numpy(dtype=np.float64)[sel]
    fig04_rows = pd.read_csv(args.fig04_dir / "selected_rows.csv")
    f_minn = float(fig04_rows["f_minnaert"].median())  # constant at unit volume
    if fig04_rows["f_minnaert"].std() > 1e-9:
        raise SystemExit("f_minnaert is not constant; cannot reuse it as a baseline")

    # One representative per bin for the figure's thumbnail strip: the bubble
    # nearest that bin's median scalar, matching how Fig. 4 picked its ten.
    sel_scalar = scal[np.array(chosen, dtype=int)]
    is_rep = np.zeros(len(sel), dtype=bool)
    for b in range(args.bins):
        where = np.flatnonzero(sel_bin == b)
        is_rep[where[int(np.argmin(np.abs(sel_scalar[where]
                                          - np.median(sel_scalar[where]))))]] = True
    if int(is_rep.sum()) != args.bins:
        raise SystemExit("failed to pick one representative per bin")

    preds = {
        "minnaert": np.full(len(sel), f_minn),
        "strasberg": f_str,
        "residual": _predict(dirs["residual"], x8[sel], f_str),
        "direct": _predict(dirs["direct"], x8[sel], f_str),
    }
    ape = {k: np.abs(v - f_gt) / f_gt * 100.0 for k, v in preds.items()}

    # ---- write ------------------------------------------------------------
    out = pd.DataFrame({
        "bin": sel_bin,
        "is_bin_thumbnail": is_rep,
        "mesh_id_10k": mesh_id[sel],
        "source_10k": df["source"].astype(str).to_numpy()[sel],
        "bin_scalar": sel_scalar,
        "non_sph_va": x8[sel, 2], "non_sph_vm": x8[sel, 3], "non_sph_w": x8[sel, 4],
        "eta_V": x8[sel, 5], "eta_A": x8[sel, 6], "eta_M": x8[sel, 7],
        "i11_over_i00": x8[sel, 0], "i22_over_i00": x8[sel, 1],
        "f_gt": f_gt, "f_minnaert": preds["minnaert"], "f_strasberg": f_str,
        "f_pred_residual": preds["residual"], "f_pred_direct": preds["direct"],
        **{f"ape_{k}": v for k, v in ape.items()},
    }).sort_values(["bin", "bin_scalar"])
    out.to_csv(args.out_dir / "selected_rows.csv", index=False)

    recs = []
    for k, v in ape.items():
        for b in range(args.bins):
            sub = v[sel_bin == b]
            recs.append({"series": k, "bin": b, "n": int(sub.size),
                         "scalar_lo": float(edges[b]), "scalar_hi": float(edges[b + 1]),
                         "mape_pct": float(sub.mean()),
                         "sem_ape_pct": float(sub.std(ddof=1) / np.sqrt(sub.size)),
                         "max_ape_pct": float(sub.max())})
    pd.DataFrame(recs).to_csv(args.out_dir / "per_bin_mape.csv", index=False)

    summary = {
        "bin_by": args.bin_by,
        "axis_label": axis_label,
        "scalar": scalar_name,
        "features": args.features,
        "pool": {"what": f"test split of the {args.features}-feature retrains",
                 "n": int(len(pool_ids)),
                 "disjoint_from_fig04_meshes": True,
                 "disjoint_from_train_and_val": True},
        "bins": args.bins, "per_bin": args.per_bin, "n_selected": int(len(sel)),
        "bin_edges": [float(e) for e in edges],
        "selection": "farthest-point sampling in the standardized 8-feature space, "
                     "seeded from the bin's median-scalar bubble (deterministic)",
        "bin_thumbnails": [
            {"bin": int(b),
             "mesh_id_10k": str(mesh_id[sel][is_rep & (sel_bin == b)][0]),
             "bin_scalar": float(sel_scalar[is_rep & (sel_bin == b)][0])}
            for b in range(args.bins)
        ],
        "overall_mape_pct": {k: float(v.mean()) for k, v in ape.items()},
        "max_ape_pct": {k: float(v.max()) for k, v in ape.items()},
        "source_counts": out["source_10k"].value_counts().to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                               encoding="utf-8")

    print(f"[{args.bin_by}] {len(sel)} bubbles, {args.bins} bins, pool {len(pool_ids)} "
          f"({out['source_10k'].value_counts().to_dict()})")
    for k, v in ape.items():
        print(f"   {k:<10} MAPE {v.mean():7.4f}%   max {v.max():7.4f}%")
    print(f"   -> {args.out_dir}")


if __name__ == "__main__":
    main()
