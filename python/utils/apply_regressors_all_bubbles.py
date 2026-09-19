"""Apply the 10k-trained regressors (MLP NN8 + poly-2 ridge) to every timed
all-bubbles mesh and report frequency MAPE vs BEM plus inference timing.

Post-pass over the ``nn8_vs_bem_all_bubbles_chunked.py`` outputs:

* re-extracts the raw 8 shape features for every row mesh (multiprocess);
* poly-2 ridge (degree 2, the 10k-tuned alpha) fit fresh on the 10k TRAIN
  split (seed-42 70/15/15, asserted against the 10k model's split.json);
* the 10k MLP checkpoint for NN8 predictions -- this also supersedes the
  ``f_hz_nn8_unit`` column of rows collected before run_all_bubbles.ps1
  passed --artifacts (those used the old 9k checkpoint);
* MAPE vs ``f_hz_bem`` over all BEM rows per scene + poly2/MLP inference
  timing (single-query and whole-scene batch).

RUN (repo root, soundlab env, AFTER the BEM run is complete):
    $env:KMP_DUPLICATE_LIB_OK = "TRUE"
    python -u python/utils/apply_regressors_all_bubbles.py
Optional: --labels fruits exhale | --limit N (smoke test) | --workers K
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import nn8_vs_bem_timing as harness  # noqa: E402  (torch-before-numpy fix)

import numpy as np  # noqa: E402

REPO_ROOT = harness.REPO_ROOT
ALL_BUBBLES = (
    REPO_ROOT / "results" / "experiments" / "table01_timing_across_scenes" / "all_bubbles"
)
DATASET_10K = REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv"
MODEL_DIR_10K = (
    REPO_ROOT / "python" / "freq_model" / "output" / "output_8feature_direct_bubblegym_10k"
)
BASELINE_METRICS_10K = (
    REPO_ROOT / "results" / "experiments" / "table02_regressor_comparison" / "baseline_metrics.json"
)

# FEATURE_COLS order of the trainer: i11_over_i00, i22_over_i00, then chull cols.
CHULL_COLS = ("non_sph_va", "non_sph_vm", "non_sph_w", "eta_V", "eta_A", "eta_M")


def _raw8_worker(path_str: str):
    """(obj path) -> raw 8-feature list in FEATURE_COLS order, or error str."""
    try:
        p = Path(path_str)
        v, f = harness.load_obj_mesh(p)
        v = np.ascontiguousarray(v, dtype=np.float64)
        f = np.ascontiguousarray(f, dtype=np.int64)
        v_u, f_u = harness._to_unit_volume(v, f)
        _m, _c, inertia_u = harness.compute_mirtich_moments(v_u, f_u)
        principal = np.linalg.eigvalsh(inertia_u).astype(np.float64)
        principal.sort()
        if principal[0] <= 0.0:
            return "bad principal inertia"
        chull = harness._extract_chull_features(v_u, f_u)
        if chull.get("status") != "ok":
            return f"bad chull: {chull.get('status')}"
        row = [principal[1] / principal[0], principal[2] / principal[0]] + [
            float(chull[c]) for c in CHULL_COLS
        ]
        if not all(math.isfinite(x) for x in row):
            return "non-finite feature"
        return row
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _fit_poly2():
    """Fit poly-2 ridge on the 10k train split; return (scaler, model, hparams)."""
    import types

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch.utils.tensorboard")
        stub.SummaryWriter = object
        sys.modules["torch.utils.tensorboard"] = stub
    fm = REPO_ROOT / "python" / "freq_model"
    for p in (REPO_ROOT / "python", fm):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    from freq_model.regression.baseline_regressors_8feat import check_split_against_paper, metrics_from_log
    from freq_model.NN.bubble_dataset import split_dataset
    from freq_model.NN.fit_shape_freq_model import load_xy

    meta = json.loads(BASELINE_METRICS_10K.read_text(encoding="utf-8"))
    hp = meta["models"]["poly2"]["hparams"]
    x_raw, y_raw, log_fs_raw, df_used = load_xy(DATASET_10K)
    split = split_dataset(x_raw, y_raw, log_fs_raw, val_ratio=0.15, test_ratio=0.15, seed=42)
    check_split_against_paper(df_used, split, split_json=MODEL_DIR_10K / "split.json")
    scaler = StandardScaler().fit(split.x_train)
    model = make_pipeline(
        PolynomialFeatures(degree=int(hp["degree"]), include_bias=True),
        Ridge(alpha=float(hp["ridge_alpha"]), random_state=0),
    )
    model.fit(scaler.transform(split.x_train), split.y_train.astype(np.float64))
    m = metrics_from_log(
        model.predict(scaler.transform(split.x_test)), split.y_test.astype(np.float64)
    )
    print(
        f"[poly2] refit on 10k train: test MAPE {m['mape']:.4f}% "
        f"(recorded {meta['models']['poly2']['metrics']['test']['mape']:.4f}%)",
        flush=True,
    )
    return scaler, model, hp, m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", nargs="+", default=["fruits", "exhale"])
    ap.add_argument("--limit", type=int, default=0, help="only first N rows per scene (smoke test)")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import torch

    scaler_p2, poly2, hp, poly2_test = _fit_poly2()
    nn_bundle = harness._load_nn_chull_inertia_8feat_direct(MODEL_DIR_10K)
    model_nn, fs_nn, ts_nn, cols_nn = nn_bundle
    assert list(cols_nn)[:2] == ["i11_over_i00", "i22_over_i00"], cols_nn

    for label in args.labels:
        rows_csv = ALL_BUBBLES / label / f"{label}_all_rows.csv"
        if not rows_csv.is_file():
            print(f"[{label}] SKIP: {rows_csv} missing", flush=True)
            continue
        with rows_csv.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if args.limit:
            rows = rows[: args.limit]
        print(f"[{label}] {len(rows)} rows; extracting raw features "
              f"({args.workers} workers)...", flush=True)

        t0 = time.perf_counter()
        paths = [r["path"] for r in rows]
        if args.workers <= 1:
            feats = [_raw8_worker(p) for p in paths]
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                feats = list(ex.map(_raw8_worker, paths, chunksize=64))
        t_feat_all = time.perf_counter() - t0
        ok = [i for i, f in enumerate(feats) if isinstance(f, list)]
        print(f"[{label}] features: {len(ok)}/{len(rows)} ok in {t_feat_all:.1f}s", flush=True)

        X = np.array([feats[i] for i in ok], dtype=np.float64)

        # poly2: whole-scene batch + single-query timing.
        Xs_p2 = scaler_p2.transform(X)
        t0 = time.perf_counter()
        log_p2 = poly2.predict(Xs_p2)
        t_p2_batch = time.perf_counter() - t0
        one = Xs_p2[:1]
        for _ in range(20):
            poly2.predict(one)
        t0 = time.perf_counter()
        reps = 200
        for _ in range(reps):
            poly2.predict(one)
        t_p2_single = (time.perf_counter() - t0) / reps
        f_p2 = np.exp(log_p2)

        # 10k MLP: batch (CPU) + GPU batch timing, matching the batched protocol.
        Xs_nn = fs_nn.transform(X).astype(np.float32)
        with torch.no_grad():
            y_norm = model_nn(torch.from_numpy(Xs_nn)).numpy().reshape(-1, 1)
        f_nn = np.exp(ts_nn.inverse_transform(y_norm).flatten().astype(np.float64))

        f_bem = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            if r.get("f_hz_bem"):
                f_bem[i] = float(r["f_hz_bem"])
        idx_ok = np.array(ok, dtype=int)
        f_bem_ok = f_bem[idx_ok]
        has_bem = np.isfinite(f_bem_ok)

        def mape(pred):
            ape = np.abs(pred[has_bem] - f_bem_ok[has_bem]) / f_bem_ok[has_bem] * 100.0
            return {
                "mape_pct": float(np.mean(ape)),
                "median_ape_pct": float(np.median(ape)),
                "p99_ape_pct": float(np.percentile(ape, 99)),
                "max_ape_pct": float(np.max(ape)),
                "n": int(has_bem.sum()),
            }

        out_csv = ALL_BUBBLES / label / f"{label}_regressor_freqs.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "bub_id", "frame", "n_vertices", "f_hz_bem", "f_hz_nn8_10k", "f_hz_poly2"])
            for j, i in enumerate(idx_ok):
                r = rows[i]
                w.writerow(
                    [r["path"], r["bub_id"], r["frame"], r["n_vertices_unit"],
                     r.get("f_hz_bem", ""), f"{f_nn[j]:.9g}", f"{f_p2[j]:.9g}"]
                )

        summary = {
            "label": label,
            "n_rows": len(rows),
            "n_features_ok": len(ok),
            "n_bem_compared": int(has_bem.sum()),
            "poly2_hparams": hp,
            "poly2_test_split_mape_pct": poly2_test["mape"],
            "mape_vs_bem": {"nn8_mlp_10k": mape(f_nn), "poly2_10k": mape(f_p2)},
            "timing": {
                "feature_extraction_total_s": t_feat_all,
                "feature_extraction_workers": args.workers,
                "poly2_batch_s": t_p2_batch,
                "poly2_batch_us_per_bubble": t_p2_batch * 1e6 / len(ok),
                "poly2_single_query_us": t_p2_single * 1e6,
            },
        }
        sp = ALL_BUBBLES / label / f"{label}_regressor_summary.json"
        sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        print(f"[{label}] wrote {out_csv} and {sp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
