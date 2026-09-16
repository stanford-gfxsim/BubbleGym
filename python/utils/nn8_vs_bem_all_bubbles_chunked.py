"""Time NN8 vs BEM on EVERY candidate mesh of an LBM per-bubble tree, chunked.

A single process cannot survive tens of thousands of bempp solves (native
memory grows across solves and the process eventually dies, cf. the Table 1
procedural sweep crashing after ~500 solves), so this driver splits the work
into chunks and runs each chunk in a fresh child process. Fully resumable:
finished chunks are skipped via a ``.done`` sentinel, and a partially written
chunk CSV is continued path-by-path.

Per-mesh rows come from ``nn8_vs_bem_timing._time_one_bubble`` and include the
BEM assemble/solve split AND the BEM + NN8 frequencies (``f_hz_bem``,
``f_hz_nn8_unit`` at unit volume).

Usage (driver = the only mode you normally invoke):

    python python/utils/nn8_vs_bem_all_bubbles_chunked.py \
        --tree dataset/fruit_mc_surface_per_bubble_smoothed_lap3 \
        --label fruits \
        --out-root results/experiments/table01_timing_across_scenes/all_bubbles \
        --artifacts python/freq_model/output/output_8feature_direct_bubblegym_10k \
        --min-vertices 100

Outputs under ``<out-root>/<label>/``:
    vertex_scan.csv          path,n_vertices for every OBJ in the tree
    chunks/chunk_XXXX.csv    per-mesh rows (+ .done sentinel per finished chunk)
    chunks/chunk_XXXX.fail   per-mesh failures (path,error), if any
    <label>_all_rows.csv     merged per-mesh rows
    <label>_all_summary.json mean/median timing summary + hardware
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

# nn8_vs_bem_timing handles the Windows torch-before-numpy import order and
# PATH fix at module import time; import it before anything numeric.
import nn8_vs_bem_timing as harness  # noqa: E402

import numpy as np  # noqa: E402

ROW_FIELDS = [
    "path",
    "bub_id",
    "frame",
    "n_vertices_prepared",
    "n_vertices_unit",
    "n_triangles",
    "t_mesh_load_s",
    "t_feat_s",
    "t_inference_gpu_s",
    "t_inference_gpu_fwd_s",
    "t_inference_cpu_s",
    "t_nn8_total_gpu_s",
    "t_nn8_total_cpu_s",
    "t_bem_assemble_s",
    "t_bem_solve_s",
    "t_bem_s",
    "f_hz_bem",
    "f_hz_nn8_unit",
]


def _bub_frame_from_path(p: Path) -> tuple[str, str]:
    """``.../bub_<id>/frame_<n>.obj`` -> (id, n); blanks if not that layout."""
    bub = p.parent.name
    bub_id = bub[4:] if bub.startswith("bub_") else ""
    stem = p.stem
    frame = stem[6:] if stem.startswith("frame_") else ""
    return bub_id, frame


def _scan_vertices(tree: Path, scan_csv: Path) -> dict[str, int]:
    """Count vertices of every OBJ in the tree (cached in scan_csv)."""
    if scan_csv.is_file():
        with scan_csv.open("r", encoding="utf-8", newline="") as fh:
            nv = {r["path"]: int(r["n_vertices"]) for r in csv.DictReader(fh)}
        print(f"[scan] loaded cached vertex counts for {len(nv)} OBJs", flush=True)
        return nv
    jobs, _, _ = harness.discover_mesh_jobs(tree)
    paths = sorted({Path(p) for _, p in jobs})
    print(f"[scan] counting vertices of {len(paths)} OBJs...", flush=True)
    nv: dict[str, int] = {}
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        try:
            nv[str(p)] = harness._n_vertices_fast(p)
        except Exception as exc:  # noqa: BLE001
            print(f"[scan] unreadable {p}: {exc}", flush=True)
        if (i + 1) % 10000 == 0:
            print(f"[scan]   {i + 1}/{len(paths)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    scan_csv.parent.mkdir(parents=True, exist_ok=True)
    with scan_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "n_vertices"])
        for k in sorted(nv):
            w.writerow([k, nv[k]])
    print(f"[scan] wrote {scan_csv} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return nv


def _nn_only_row(
    obj_path: Path,
    *,
    nn_bundle: tuple,
    model_gpu,
    use_cuda: bool,
    infer_repeats: int,
) -> dict:
    """NN feature+inference timing and NN frequency, no BEM (mesh above cap)."""
    model, feat_scaler, tgt_scaler, feat_cols = nn_bundle
    t0 = time.perf_counter()
    v_raw, f_raw = harness._prepare_mesh_vf(
        obj_path, meshes_root=None, smooth_iters=0, smooth_cache_root=obj_path.parent
    )
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    v_unit, f_unit = harness._to_unit_volume(v_raw, f_raw)
    _m, _c, inertia_u = harness.compute_mirtich_moments(v_unit, f_unit)
    principal = np.linalg.eigvalsh(inertia_u).astype(np.float64)
    principal.sort()
    chull = harness.nonspherical_features_from_unit_mesh(v_unit, f_unit)
    Xn = harness._build_nn8_input(
        chull, principal, feature_scaler=feat_scaler, feature_cols=feat_cols
    )
    t_feat = time.perf_counter() - t0
    infer = harness._time_nn8_inference(
        Xn, model_cpu=model, model_gpu=model_gpu, repeats=infer_repeats, use_cuda=use_cuda
    )
    f_hat = harness._predict_f_unit(Xn, model_cpu=model, target_scaler=tgt_scaler)
    return {
        "path": str(obj_path.resolve()),
        "n_vertices_prepared": int(v_raw.shape[0]),
        "n_vertices_unit": int(v_unit.shape[0]),
        "n_triangles": int(f_unit.shape[0]),
        "t_mesh_load_s": float(t_load),
        "t_feat_s": float(t_feat),
        "t_inference_gpu_s": float(infer["t_inference_gpu_s"]),
        "t_inference_gpu_fwd_s": float(infer["t_inference_gpu_fwd_s"]),
        "t_inference_cpu_s": float(infer["t_inference_cpu_s"]),
        "t_nn8_total_gpu_s": "",
        "t_nn8_total_cpu_s": "",
        "t_bem_assemble_s": "",
        "t_bem_solve_s": "",
        "t_bem_s": "",
        "f_hz_bem": "",
        "f_hz_nn8_unit": float(f_hat),
    }


def _run_child(args: argparse.Namespace) -> int:
    chunk_paths = [
        Path(line.strip())
        for line in Path(args.chunk_file).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    out_csv = Path(args.out_csv)
    done_paths: set[str] = set()
    if out_csv.is_file():
        with out_csv.open("r", encoding="utf-8", newline="") as fh:
            done_paths = {r["path"] for r in csv.DictReader(fh)}
        print(f"[child] resuming: {len(done_paths)} rows already in {out_csv.name}", flush=True)
    todo = [p for p in chunk_paths if str(p.resolve()) not in done_paths]

    # Blacklist meshes that repeatedly killed a child (marker written before
    # each mesh; a mesh whose marker survives 2 child deaths is skipped).
    marker = out_csv.with_suffix(".inprogress")
    abort_json = out_csv.with_suffix(".abortcount")
    abort_counts: dict[str, int] = {}
    if abort_json.is_file():
        abort_counts = json.loads(abort_json.read_text(encoding="utf-8"))
    if marker.is_file():
        stuck = marker.read_text(encoding="utf-8").strip()
        marker.unlink()
        if stuck:
            abort_counts[stuck] = abort_counts.get(stuck, 0) + 1
            abort_json.write_text(json.dumps(abort_counts), encoding="utf-8")
    blacklist = {p for p, n in abort_counts.items() if n >= 2}
    if blacklist:
        fail_path0 = out_csv.with_suffix(".fail")
        for b in sorted(blacklist & {str(p.resolve()) for p in todo}):
            print(f"[child] BLACKLIST (killed 2 children): {b}", flush=True)
            with fail_path0.open("a", encoding="utf-8", newline="") as ff:
                csv.writer(ff).writerow([b, "aborted: repeatedly killed child (timeout)"])
        todo = [p for p in todo if str(p.resolve()) not in blacklist]

    if not todo:
        print("[child] nothing to do", flush=True)
        return 0

    t0 = time.perf_counter()
    nn_bundle = harness.load_nn_chull_inertia_8feat_direct(Path(args.artifacts))
    model_nn, _fs, _ts, cols = nn_bundle
    import copy

    import torch

    with torch.no_grad():
        _ = model_nn(torch.zeros(1, len(cols), dtype=torch.float32))
    use_cuda = torch.cuda.is_available()
    model_gpu = None
    if use_cuda:
        model_gpu = copy.deepcopy(model_nn).to("cuda").eval()
        with torch.no_grad():
            _ = model_gpu(torch.zeros(1, len(cols), dtype=torch.float32, device="cuda"))
        torch.cuda.synchronize()
    print(f"[child] model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    # Warm-up on the driver-chosen small mesh, untimed: full NN feature path
    # (Mirtich + chull + scaler/model first-call costs) plus one BEM solve
    # (bempp/numba JIT), so the first timed mesh sees steady-state costs.
    # A pathological warm-up mesh must not kill the chunk (bempp can raise on
    # degenerate geometry): fall back to the next todo meshes until one works.
    warmup_candidates = ([Path(args.warmup_obj)] if args.warmup_obj else []) + todo[:5]
    warmup_cap = int(args.bem_max_vertices) or 5000
    for wp in warmup_candidates:
        t0 = time.perf_counter()
        try:
            if harness._n_vertices_fast(wp) > warmup_cap:
                print(f"[child] warm-up skip {wp.name} (> {warmup_cap} verts)", flush=True)
                continue
            v_raw, f_raw = harness._prepare_mesh_vf(
                wp, meshes_root=None, smooth_iters=0, smooth_cache_root=wp.parent
            )
            v_u, f_u = harness._to_unit_volume(v_raw, f_raw)
            _m, _c, inertia_u = harness.compute_mirtich_moments(v_u, f_u)
            principal = np.linalg.eigvalsh(inertia_u).astype(np.float64)
            principal.sort()
            chull = harness.nonspherical_features_from_unit_mesh(v_u, f_u)
            Xn = harness._build_nn8_input(
                chull, principal, feature_scaler=_fs, feature_cols=cols
            )
            harness._time_nn8_inference(
                Xn, model_cpu=model_nn, model_gpu=model_gpu, repeats=20, use_cuda=use_cuda
            )
            harness.solve_minnaert_frequency_galerkin(
                (v_u, f_u),
                trial_pair="P1-DP0",
                precond="mass",
                solver="gmres",
                gmres_tol=1e-12,
                quadrature_regular=6,
                quadrature_singular=6,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[child] warm-up FAILED on {wp.name} "
                f"({type(exc).__name__}: {exc}); trying next candidate",
                flush=True,
            )
            continue
        print(f"[child] BEM warm-up ({wp.name}) in {time.perf_counter() - t0:.1f}s", flush=True)
        break
    else:
        print("[child] WARNING: no warm-up candidate solved; first row absorbs JIT", flush=True)

    fail_path = out_csv.with_suffix(".fail")
    new_file = not out_csv.is_file()
    with out_csv.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        bem_cap = int(args.bem_max_vertices)
        for i, p in enumerate(todo):
            marker.write_text(str(p.resolve()), encoding="utf-8")
            try:
                nn_only = bem_cap > 0 and harness._n_vertices_fast(p) > bem_cap
                if nn_only:
                    row = _nn_only_row(
                        p,
                        nn_bundle=nn_bundle,
                        model_gpu=model_gpu,
                        use_cuda=use_cuda,
                        infer_repeats=int(args.infer_repeats),
                    )
                else:
                    row = harness._time_one_bubble(
                        p,
                        meshes_root=None,
                        smooth_iters=0,
                        smooth_cache_root=p.parent,
                        nn_bundle=nn_bundle,
                        model_gpu=model_gpu,
                        use_cuda=use_cuda,
                        infer_repeats=int(args.infer_repeats),
                    )
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                print(f"[child] FAIL {p.name}: {msg}", flush=True)
                with fail_path.open("a", encoding="utf-8", newline="") as ff:
                    csv.writer(ff).writerow([str(p.resolve()), msg])
                continue
            bub_id, frame = _bub_frame_from_path(p)
            row["bub_id"], row["frame"] = bub_id, frame
            w.writerow(row)
            fh.flush()
            if nn_only:
                print(
                    f"[child] [{i + 1}/{len(todo)}] bub_{bub_id}/f{frame} "
                    f"v={row['n_vertices_unit']} NN-ONLY (> {bem_cap} verts) "
                    f"feat={row['t_feat_s'] * 1e3:.1f}ms f_NN8={row['f_hz_nn8_unit']:.4g}Hz",
                    flush=True,
                )
            else:
                print(
                    f"[child] [{i + 1}/{len(todo)}] bub_{bub_id}/f{frame} "
                    f"v={row['n_vertices_unit']} BEM={row['t_bem_s']:.3f}s "
                    f"(asm {row['t_bem_assemble_s']:.3f}/slv {row['t_bem_solve_s']:.3f}) "
                    f"feat={row['t_feat_s'] * 1e3:.1f}ms f_BEM={row['f_hz_bem']:.4g}Hz",
                    flush=True,
                )
    if marker.is_file():
        marker.unlink()
    return 0


def _merge_and_summarize(label: str, out_dir: Path, chunk_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    fail_paths: set[str] = set()
    for cp in sorted(chunk_dir.glob("chunk_*.csv")):
        with cp.open("r", encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh))
        fp = cp.with_suffix(".fail")
        if fp.is_file():
            # A resumed chunk retries failed meshes, appending duplicate fail
            # lines; count unique paths, and drop any that later succeeded.
            with fp.open("r", encoding="utf-8", newline="") as fh:
                for frow in csv.reader(fh):
                    if frow:
                        fail_paths.add(frow[0])
    fail_paths -= {r["path"] for r in rows}
    n_fail = len(fail_paths)
    merged = out_dir / f"{label}_all_rows.csv"
    with merged.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def _arr(k: str) -> np.ndarray:
        return np.array([float(r[k]) for r in rows if r.get(k)], dtype=np.float64)

    bem, asm, slv = _arr("t_bem_s"), _arr("t_bem_assemble_s"), _arr("t_bem_solve_s")
    feat = _arr("t_feat_s")
    inf_g, inf_c = _arr("t_inference_gpu_s"), _arr("t_inference_cpu_s")
    inf_g = inf_g[np.isfinite(inf_g)]
    hw = harness._detect_hardware()
    summary = {
        "label": label,
        "n_rows": len(rows),
        "n_bem_rows": int(bem.size),
        "n_nn_only_rows": len(rows) - int(bem.size),
        "n_failures": n_fail,
        "hardware_cpu": hw["cpu"],
        "hardware_gpu": hw["gpu"],
        "bem_settings": "P1-DP0, mass precond, gmres tol=1e-12, quad 6/6",
        "mean_bem_s": float(np.mean(bem)),
        "median_bem_s": float(np.median(bem)),
        "mean_bem_assemble_s": float(np.mean(asm)),
        "median_bem_assemble_s": float(np.median(asm)),
        "mean_bem_solve_s": float(np.mean(slv)),
        "median_bem_solve_s": float(np.median(slv)),
        "max_bem_s": float(np.max(bem)),
        "total_bem_hours": float(np.sum(bem) / 3600.0),
        "mean_nn_feat_s": float(np.mean(feat)),
        "median_nn_feat_s": float(np.median(feat)),
        "mean_nn_infer_gpu_s": float(np.mean(inf_g)) if inf_g.size else None,
        "mean_nn_infer_cpu_s": float(np.mean(inf_c)),
        "mean_f_hz_bem": float(np.mean(_arr("f_hz_bem"))),
    }
    sp = out_dir / f"{label}_all_summary.json"
    sp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[driver] merged {len(rows)} rows ({n_fail} failures) -> {merged}", flush=True)
    print(f"[driver] summary -> {sp}", flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


def _run_driver(args: argparse.Namespace) -> int:
    tree = Path(args.tree).resolve()
    out_dir = Path(args.out_root) / args.label
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    nv = _scan_vertices(tree, out_dir / "vertex_scan.csv")
    _lo, _hi = int(args.min_vertices), int(args.max_vertices or 0)
    # Smallest first: a chunk then produces rows within seconds instead of
    # stalling behind its largest mesh, so a timeout resumes with progress.
    cands = [p for p, n in sorted(nv.items(), key=lambda kv: kv[1])
             if n > _lo and (not _hi or n <= _hi)]
    print(
        f"[driver] {len(nv)} OBJs total | {len(cands)} candidates "
        f"({args.min_vertices} < V"
        + (f" <= {args.max_vertices}" if args.max_vertices else "")
        + ")",
        flush=True,
    )
    if not cands:
        print("[driver] nothing to do", flush=True)
        return 1

    size = int(args.chunk_size)
    chunks = [cands[i : i + size] for i in range(0, len(cands), size)]
    print(f"[driver] {len(chunks)} chunks of <= {size}", flush=True)

    t_start = time.perf_counter()
    n_done_before = 0
    for ci, chunk in enumerate(chunks):
        done = chunk_dir / f"chunk_{ci:04d}.done"
        if done.is_file():
            n_done_before += 1
            continue
        chunk_file = chunk_dir / f"chunk_{ci:04d}.paths"
        chunk_file.write_text("\n".join(chunk), encoding="utf-8")
        warmup = min(chunk, key=lambda p: nv[p])
        out_csv = chunk_dir / f"chunk_{ci:04d}.csv"
        cmd = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--mode",
            "child",
            "--chunk-file",
            str(chunk_file),
            "--out-csv",
            str(out_csv),
            "--warmup-obj",
            str(warmup),
            "--artifacts",
            str(args.artifacts),
            "--infer-repeats",
            str(args.infer_repeats),
            "--bem-max-vertices",
            str(args.bem_max_vertices),
        ]
        print(f"[driver] === chunk {ci + 1}/{len(chunks)} ({len(chunk)} meshes) ===", flush=True)

        def _rows_in(csv_path: Path) -> int:
            if not csv_path.is_file():
                return 0
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                return sum(1 for _ in fh) - 1

        # Retry while the child makes progress; a chunk only counts as failed
        # after 3 consecutive attempts with zero new rows. A timeout kill is
        # fine: the child resumes path-by-path, and a mesh that kills two
        # children in a row is blacklisted by the child itself.
        no_progress = 0
        while no_progress < 3:
            before = _rows_in(out_csv)
            try:
                rc = subprocess.run(cmd, timeout=float(args.chunk_timeout)).returncode
            except subprocess.TimeoutExpired:
                rc = -1
                print(
                    f"[driver] chunk {ci} child exceeded {args.chunk_timeout}s; killed",
                    flush=True,
                )
            if rc == 0:
                done.touch()
                break
            after = _rows_in(out_csv)
            no_progress = 0 if after > before else no_progress + 1
            print(
                f"[driver] chunk {ci} child exited rc={rc} "
                f"({after - before} new rows); resuming in a fresh child "
                f"(no-progress streak: {no_progress})",
                flush=True,
            )
        else:
            print(f"[driver] chunk {ci}: 3 attempts without progress; moving on", flush=True)
        elapsed = time.perf_counter() - t_start
        fresh = ci + 1 - n_done_before
        if fresh > 0:
            eta_h = elapsed / fresh * (len(chunks) - ci - 1) / 3600.0
            print(
                f"[driver] progress {ci + 1}/{len(chunks)} | elapsed {elapsed / 3600.0:.2f} h "
                f"| ETA ~{eta_h:.1f} h",
                flush=True,
            )

    _merge_and_summarize(args.label, out_dir, chunk_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("drive", "child", "merge"), default="drive")
    ap.add_argument("--tree", type=Path, help="bub_<id>/frame_<n>.obj mesh tree (drive mode)")
    ap.add_argument("--label", type=str, default="scene")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=(harness.REPO_ROOT / "results" / "experiments"
                 / "table01_timing_across_scenes" / "all_bubbles"),
    )
    ap.add_argument(
        "--artifacts",
        type=Path,
        # NOT harness.DEFAULT_ARTIFACTS (the old 9k model): Table 1 uses the
        # 10k retrain.
        default=harness.PYTHON_ROOT
        / "freq_model"
        / "output" / "output_8feature_direct_bubblegym_10k",
    )
    ap.add_argument("--min-vertices", type=int, default=100)
    ap.add_argument(
        "--max-vertices",
        type=int,
        default=0,
        help="Skip meshes above this vertex count entirely (0 = no limit). "
        "Unlike --bem-max-vertices, which only suppresses the BEM solve and "
        "still pays mesh load, hull and feature extraction for an NN-only row.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Meshes per child process. LBM bubble meshes are small (median "
        "~500 verts) so the per-solve native leak that killed the procedural "
        "sweep after ~500 large solves is not limiting here; a crashed child "
        "is retried and resumes path-by-path anyway.",
    )
    ap.add_argument("--infer-repeats", type=int, default=100)
    ap.add_argument(
        "--bem-max-vertices",
        type=int,
        default=5000,
        help="Meshes above this vertex count get an NN-only row (no BEM): a "
        "dense BEM system for a 30k+-vert merged cavity takes hours and up to "
        ">100 GB; 5000 covers ~97.7%% of LBM meshes and matches the largest "
        "procedural (Enright) meshes. 0 disables the cap.",
    )
    ap.add_argument(
        "--chunk-timeout",
        type=float,
        default=5400.0,
        help="Seconds before a child process is killed and resumed (safety "
        "net; legitimate chunks finish well under this with the BEM cap).",
    )
    ap.add_argument("--chunk-file", type=str, help="(child mode)")
    ap.add_argument("--out-csv", type=str, help="(child mode)")
    ap.add_argument("--warmup-obj", type=str, default="", help="(child mode)")
    args = ap.parse_args()

    if args.mode == "child":
        return _run_child(args)
    if args.mode == "merge":
        out_dir = Path(args.out_root) / args.label
        _merge_and_summarize(args.label, out_dir, out_dir / "chunks")
        return 0
    if args.tree is None:
        ap.error("--tree is required in drive mode")
    return _run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
