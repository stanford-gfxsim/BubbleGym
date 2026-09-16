"""Time NN8 (features + inference) vs P1-DP0 Galerkin BEM per bubble mesh.

Source of the per-mesh timings in the paper's Table 1. For each scene it finds
bubble OBJs above ``--min-vertices``, randomly picks ``--num-bubbles`` of them
(or every one, with ``--all-bubbles``), and reports the mean of:

* BEM wall time (operator assembly + linear solve from the Galerkin solver)
* NN8 feature time (load / optional Laplacian smooth, unit-volume rescale,
  chull features + Mirtich inertia)
* NN8 inference time (feature scaling + one forward pass through BubbleFreqNet)

Model weights load once; that time is printed but excluded from the per-bubble
inference average. Speedup is ``mean(BEM) / mean(features + inference)``.

A scene argument is either a flat folder of ``*.obj``, an LBM scene with meshes
under ``<scene>/ppm_ve_home_test_phi_iter/mc_surface_per_bubble``, or
``<scene>@<meshes_root>`` where the part after ``@`` is an external tree walked
as ``bub_<id>/frame_<n>.obj`` and the part before it only labels the scene and
holds the smoothing cache. A ``*smoothed*`` meshes_root defaults to
``--smooth-iters 0`` so those meshes are not Laplacian-smoothed twice.

    python python/utils/nn8_vs_bem_timing.py \\
        --scenes dataset/bubble_theater/curl_noise/mesh \\
            dataset/exhalation@<lbm-output>/mc_surface_per_bubble_smoothed_lap3
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

# torch before scipy / igl on Windows in some envs (same as other scene scripts).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Drop %CONDA_PREFIX%\Library\mingw-w64\bin from PATH before importing torch:
# under `conda run` those MinGW DLLs shadow torch/lib/shm.dll and raise
# "OSError: [WinError 127]" (see python/win_torch_dll_path.py).
if os.name == "nt":
    _cp = os.environ.get("CONDA_PREFIX")
    if _cp:
        _mingw = os.path.normcase(
            os.path.normpath(os.path.join(_cp, "Library", "mingw-w64", "bin"))
        )
        os.environ["PATH"] = os.pathsep.join(
            e
            for e in os.environ.get("PATH", "").split(os.pathsep)
            if e and os.path.normcase(os.path.normpath(e)) != _mingw
        )

# Import torch BEFORE numpy: on this Windows stack importing NumPy (MKL) first
# also triggers the shm.dll WinError 127 (see python/win_torch_dll_path.py).
import torch as _torch  # noqa: E402, F401
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for _p in (
    REPO_ROOT,
    PYTHON_ROOT,
    PYTHON_ROOT / "freq_model",
    PYTHON_ROOT / "baseline",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from python.bem.compute_freq_bempp_galerkin import (  # noqa: E402
    solve_minnaert_frequency_galerkin,
)
from tracked_bubinfo import discover_mesh_jobs
from shape_feature.mesh_utils import load_or_smooth_obj
from freq_model.NN.nn_inference import load_nn_chull_inertia_8feat_direct
from shape_feature.nonspherical_features import nonspherical_features_from_unit_mesh  # noqa: E402
from shape_feature.mesh_utils import load_obj_mesh  # noqa: E402
from shape_feature.mesh_utils import (  # noqa: E402
    compute_mirtich_moments,
    mesh_signed_volume,
    vertices_scaled_to_target_volume,
)

DEFAULT_ARTIFACTS = (
    PYTHON_ROOT / "freq_model" / "output" / "output_8feature_direct_bubblegym_10k"
)
DEFAULT_MESH_SUBDIR = Path("ppm_ve_home_test_phi_iter") / "mc_surface_per_bubble"

# The per-bubble mesh trees for the two LBM scenes are produced by the separate
# LBM simulator and are not shipped here. Point BUBBLEGYM_LBM_ROOT at that
# simulator's output directory, or name the trees explicitly with --scene.
_LBM_ROOT = Path(os.environ.get("BUBBLEGYM_LBM_ROOT", str(REPO_ROOT / "external" / "lbm_output")))
_DEFAULT_EXT_EXHALE = (
    _LBM_ROOT / "dataMR3D_exhale"
    / "ppm_ve_home_test_phi_iter"
    / "mc_surface_per_bubble_smoothed_lap3"
)
_DEFAULT_EXT_FRUIT = (
    _LBM_ROOT / "dataMR3D_fruit"
    / "ppm_ve_home_test_phi_iter"
    / "mc_surface_per_bubble_smoothed_lap3"
)

DEFAULT_SCENE_SPECS: tuple[str, ...] = (
    str(REPO_ROOT / "dataset/bubble_theater/curl_noise/mesh"),
    f"{REPO_ROOT / 'dataset/exhalation'}@{_DEFAULT_EXT_EXHALE}",
    f"{REPO_ROOT / 'dataset/fruit_splash'}@{_DEFAULT_EXT_FRUIT}",
)

# When an LBM tree lists more OBJs than this, default vertex filtering samples a subset.
VERTEX_SCAN_AUTO_THRESHOLD = 25000
VERTEX_SCAN_AUTO_CAP = 25000


def _parse_scene_spec(raw: str) -> tuple[Path, Path | None]:
    """``scene_dir`` or ``scene_dir@meshes_root`` -> (scene_dir, override or None)."""
    s = raw.strip()
    if "@" in s:
        left, right = s.split("@", 1)
        scene = Path(left.strip()).expanduser()
        override = Path(right.strip()).expanduser()
        return scene, override
    return Path(s).expanduser(), None


def _iter_scene_obj_paths(
    scene_dir: Path,
    meshes_root_override: Path | None = None,
) -> tuple[list[Path], Path | None]:
    """Return (obj paths, meshes_root or None if flat *.obj layout)."""
    if meshes_root_override is not None:
        mr = meshes_root_override.resolve()
        if not mr.is_dir():
            return [], mr
        jobs, _, _ = discover_mesh_jobs(mr)
        paths = sorted({Path(p) for _, p in jobs})
        return paths, mr

    meshes_root = scene_dir / DEFAULT_MESH_SUBDIR
    if meshes_root.is_dir():
        jobs, _, _ = discover_mesh_jobs(meshes_root)
        paths = sorted({Path(p) for _, p in jobs})
        return paths, meshes_root
    flat = sorted(scene_dir.glob("*.obj"))
    return flat, None


def _n_vertices_fast(path: Path) -> int:
    """Count geometric vertices as in ``load_obj_mesh`` (lines ``v x y z`` only)."""
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    n += 1
    return n


def _n_vertices(path: Path) -> int:
    """Full parse fallback (same count as fast path for well-formed OBJ)."""
    return _n_vertices_fast(path)


def _prepare_mesh_vf(
    obj_path: Path,
    *,
    meshes_root: Path | None,
    smooth_iters: int,
    smooth_cache_root: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (V, F) in raw scale (not yet unit volume)."""
    if smooth_iters > 0 and meshes_root is not None:
        rel = obj_path.relative_to(meshes_root)
        cached = smooth_cache_root / rel
        v, f = load_or_smooth_obj(obj_path, cached, iters=int(smooth_iters))
        return v, f
    v, f = load_obj_mesh(obj_path)
    return np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(
        f, dtype=np.int64
    )


def _to_unit_volume(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mesh_signed_volume(v, f) < 0.0:
        f = np.ascontiguousarray(f[:, [0, 2, 1]], dtype=np.int64)
    v_unit = vertices_scaled_to_target_volume(v, f, target_volume=1.0)
    return v_unit, f


def _detect_hardware() -> dict[str, str]:
    """Best-effort CPU + GPU description strings for the timing CSV.

    CPU name comes from the Windows registry ``ProcessorNameString`` (falls
    back to ``platform.processor()`` / ``PROCESSOR_IDENTIFIER``); GPU name from
    ``torch.cuda.get_device_name`` when a CUDA device is visible.
    """
    cpu_name = ""
    if os.name == "nt":
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                cpu_name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except Exception:  # noqa: BLE001
            cpu_name = ""
    if not cpu_name:
        cpu_name = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
    cpu_name = cpu_name.strip() or "unknown CPU"
    n_logical = os.cpu_count() or 0
    cpu = f"{cpu_name} ({n_logical} logical cores)" if n_logical else cpu_name

    gpu = "CPU-only (no CUDA device)"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            gpu = str(torch.cuda.get_device_name(0))
    except Exception:  # noqa: BLE001
        pass
    return {"cpu": cpu, "gpu": gpu}


def _build_nn8_input(
    chull: dict[str, Any],
    principal_unit: np.ndarray,
    *,
    feature_scaler: Any,
    feature_cols: list[str],
) -> np.ndarray:
    """Build the scaled (1, N) float32 network input ``Xn`` (all CPU work)."""
    if chull.get("status") != "ok":
        raise ValueError(f"bad chull features: {chull.get('status')}")
    chull_cols = [c for c in feature_cols if c not in ("i11_over_i00", "i22_over_i00")]
    row = [float(chull[c]) for c in chull_cols]
    if not all(math.isfinite(x) for x in row):
        raise ValueError("non-finite chull row")
    Ip = np.asarray(principal_unit, dtype=np.float64)
    if Ip.shape != (3,) or float(Ip[0]) <= 0.0:
        raise ValueError("bad principal inertia")
    i00 = float(Ip[0])
    i11 = float(Ip[1]) / i00
    i22 = float(Ip[2]) / i00
    vec: list[float] = []
    for c in feature_cols:
        if c == "i11_over_i00":
            vec.append(i11)
        elif c == "i22_over_i00":
            vec.append(i22)
        else:
            vec.append(float(chull[c]))
    X = np.array([vec], dtype=np.float32)
    return feature_scaler.transform(X).astype(np.float32)


def _predict_f_unit(
    Xn: np.ndarray, *, model_cpu: Any, target_scaler: Any
) -> float:
    """Predicted frequency at unit volume from the scaled input (CPU forward)."""
    import torch

    with torch.no_grad():
        y_norm = model_cpu(torch.from_numpy(Xn)).numpy().reshape(-1, 1)
    log_pred = target_scaler.inverse_transform(y_norm).flatten().astype(np.float64)
    return float(np.exp(log_pred[0]))


def _time_nn8_inference(
    Xn: np.ndarray,
    *,
    model_cpu: Any,
    model_gpu: Any | None,
    repeats: int,
    use_cuda: bool,
) -> dict[str, float]:
    """Mean per-call forward times (seconds) for the scaled input ``Xn``.

    Returns ``t_inference_gpu_s`` (host->device + forward + device->host, with
    ``cuda.synchronize``), ``t_inference_gpu_fwd_s`` (forward only, input
    pre-placed on the GPU), and ``t_inference_cpu_s`` (CPU forward, no
    transfer). The first ``warmup`` iterations on each device are discarded.
    """
    import torch

    reps = max(1, int(repeats))
    warmup = min(reps, max(1, reps // 10))

    t_gpu_full = float("nan")
    t_gpu_fwd = float("nan")
    if use_cuda and model_gpu is not None:
        with torch.no_grad():
            for _ in range(warmup):
                x = torch.from_numpy(Xn).to("cuda")
                _ = model_gpu(x).cpu().numpy()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                x = torch.from_numpy(Xn).to("cuda")
                _ = model_gpu(x).cpu().numpy()
            torch.cuda.synchronize()
            t_gpu_full = (time.perf_counter() - t0) / reps

            x_dev = torch.from_numpy(Xn).to("cuda")
            for _ in range(warmup):
                _ = model_gpu(x_dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                _ = model_gpu(x_dev)
            torch.cuda.synchronize()
            t_gpu_fwd = (time.perf_counter() - t0) / reps

    with torch.no_grad():
        x_cpu = torch.from_numpy(Xn)
        for _ in range(warmup):
            _ = model_cpu(x_cpu).numpy()
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = model_cpu(x_cpu).numpy()
        t_cpu = (time.perf_counter() - t0) / reps

    return {
        "t_inference_gpu_s": t_gpu_full,
        "t_inference_gpu_fwd_s": t_gpu_fwd,
        "t_inference_cpu_s": t_cpu,
    }


def _time_one_bubble(
    obj_path: Path,
    *,
    meshes_root: Path | None,
    smooth_iters: int,
    smooth_cache_root: Path,
    nn_bundle: tuple[Any, Any, Any, list[str]],
    model_gpu: Any | None = None,
    use_cuda: bool = False,
    infer_repeats: int = 100,
    bem_defaults: bool = False,
) -> dict[str, Any]:
    """Return timing dict for one mesh.

    NN inference is averaged over ``infer_repeats`` forwards per device (see
    ``_time_nn8_inference``); ``t_inference_cpu_s`` is always recorded and the
    GPU fields are NaN unless ``use_cuda`` and ``model_gpu`` are given.

    ``bem_defaults=True`` calls the Galerkin solver with the current solver's native
    defaults (relaxed gmres tol / lower quadrature) instead of the benchmark's historical
    explicit ``tol=1e-12, quad 6/6`` overrides.
    """
    model, feat_scler, tgt_scler, feat_cols = nn_bundle

    t0 = time.perf_counter()
    v_raw, f_raw = _prepare_mesh_vf(
        obj_path,
        meshes_root=meshes_root,
        smooth_iters=smooth_iters,
        smooth_cache_root=smooth_cache_root,
    )
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    v_unit, f_unit = _to_unit_volume(v_raw, f_raw)
    _mass, _com, inertia_u = compute_mirtich_moments(v_unit, f_unit)
    principal = np.linalg.eigvalsh(inertia_u).astype(np.float64)
    principal.sort()
    chull = nonspherical_features_from_unit_mesh(v_unit, f_unit)
    Xn = _build_nn8_input(
        chull, principal, feature_scaler=feat_scler, feature_cols=feat_cols
    )
    t_feat = time.perf_counter() - t0

    infer = _time_nn8_inference(
        Xn,
        model_cpu=model,
        model_gpu=model_gpu,
        repeats=infer_repeats,
        use_cuda=use_cuda,
    )
    f_hat = _predict_f_unit(Xn, model_cpu=model, target_scaler=tgt_scler)

    bem_kwargs: dict[str, Any] = dict(
        trial_pair="P1-DP0", precond="mass", solver="gmres", verbose=False
    )
    if not bem_defaults:
        bem_kwargs.update(gmres_tol=1e-12, quadrature_regular=6, quadrature_singular=6)
    out = solve_minnaert_frequency_galerkin((v_unit, f_unit), **bem_kwargs)
    t_bem_assemble = float(out["wall_time_assemble_s"])
    t_bem_solve = float(out["wall_time_solve_s"])
    t_bem = t_bem_assemble + t_bem_solve

    t_infer_gpu = float(infer["t_inference_gpu_s"])
    t_infer_cpu = float(infer["t_inference_cpu_s"])
    # Primary NN8 total uses the GPU forward when CUDA is the active device,
    # otherwise falls back to the CPU forward.
    t_infer_primary = t_infer_gpu if (use_cuda and math.isfinite(t_infer_gpu)) else t_infer_cpu
    t_nn8_total_gpu = float(t_feat + t_infer_primary)
    t_nn8_total_cpu = float(t_feat + t_infer_cpu)

    return {
        "path": str(obj_path.resolve()),
        "n_vertices_prepared": int(v_raw.shape[0]),
        "n_vertices_unit": int(v_unit.shape[0]),
        "n_triangles": int(f_unit.shape[0]),
        "t_mesh_load_s": float(t_load),
        "t_feat_s": float(t_feat),
        "t_inference_gpu_s": t_infer_gpu,
        "t_inference_gpu_fwd_s": float(infer["t_inference_gpu_fwd_s"]),
        "t_inference_cpu_s": t_infer_cpu,
        "t_nn8_total_gpu_s": t_nn8_total_gpu,
        "t_nn8_total_cpu_s": t_nn8_total_cpu,
        "t_bem_s": float(t_bem),
        "t_bem_assemble_s": t_bem_assemble,
        "t_bem_solve_s": t_bem_solve,
        "speedup_bem_over_nn8_gpu": float(t_bem / max(1e-30, t_nn8_total_gpu)),
        "speedup_bem_over_nn8_cpu": float(t_bem / max(1e-30, t_nn8_total_cpu)),
        "f_hz_bem": float(out["frequency"]),
        "f_hz_nn8_unit": float(f_hat),
    }


def _scene_auto_smooth_iters(meshes_root: Path | None) -> int:
    if meshes_root is None:
        return 0
    p = str(meshes_root).lower()
    if "smoothed" in p:
        return 0
    return 3


CSV_COLUMNS: tuple[str, ...] = (
    "scene",
    "bubble",
    "n_vertices",
    "n_triangles",
    "t_mesh_load_s",
    "t_feature_s",
    "t_inference_gpu_s",
    "t_inference_gpu_fwd_s",
    "t_inference_cpu_s",
    "t_nn8_total_gpu_s",
    "t_nn8_total_cpu_s",
    "t_bem_assemble_s",
    "t_bem_solve_s",
    "t_bem_total_s",
    "speedup_bem_over_nn8_gpu",
    "speedup_bem_over_nn8_cpu",
    "f_hz_bem",
    "f_hz_nn8_unit",
    "nn_device",
    "cpu",
    "gpu",
    "hardware_note",
)


def _write_scene_csv(
    out_dir: Path,
    *,
    scene_dir: Path,
    scene_rows: list[dict[str, Any]],
    hardware: dict[str, str],
    nn_device: str,
    hardware_note: str,
) -> Path:
    """Write one per-bubble CSV for a scene and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"nn8_vs_bem_timing_{scene_dir.name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in scene_rows:
            w.writerow(
                {
                    "scene": scene_dir.name,
                    "bubble": Path(r["path"]).name,
                    "n_vertices": r["n_vertices_unit"],
                    "n_triangles": r["n_triangles"],
                    "t_mesh_load_s": r["t_mesh_load_s"],
                    "t_feature_s": r["t_feat_s"],
                    "t_inference_gpu_s": r["t_inference_gpu_s"],
                    "t_inference_gpu_fwd_s": r["t_inference_gpu_fwd_s"],
                    "t_inference_cpu_s": r["t_inference_cpu_s"],
                    "t_nn8_total_gpu_s": r["t_nn8_total_gpu_s"],
                    "t_nn8_total_cpu_s": r["t_nn8_total_cpu_s"],
                    "t_bem_assemble_s": r["t_bem_assemble_s"],
                    "t_bem_solve_s": r["t_bem_solve_s"],
                    "t_bem_total_s": r["t_bem_s"],
                    "speedup_bem_over_nn8_gpu": r["speedup_bem_over_nn8_gpu"],
                    "speedup_bem_over_nn8_cpu": r["speedup_bem_over_nn8_cpu"],
                    "f_hz_bem": r["f_hz_bem"],
                    "f_hz_nn8_unit": r["f_hz_nn8_unit"],
                    "nn_device": nn_device,
                    "cpu": hardware["cpu"],
                    "gpu": hardware["gpu"],
                    "hardware_note": hardware_note,
                }
            )
    return csv_path


CSV_HEADER = [
    "frame_name",
    "n_vertices",
    "n_triangles",
    "t_feat_s",
    "t_infer_cpu_s",
    "t_infer_gpu_s",
    "t_nn8_total_cpu_s",
    "t_nn8_total_gpu_s",
    "t_bem_assemble_s",
    "t_bem_solve_s",
    "t_bem_total_s",
]


def _csv_num(x: float) -> Any:
    """Blank cell for non-finite (NaN) values, else the value itself."""
    return "" if (isinstance(x, float) and not math.isfinite(x)) else x


def _run_csv_all_for_scene(
    scene_dir: Path,
    mesh_override: Path | None,
    *,
    nn_bundle: tuple[Any, Any, Any, list[str]],
    model_gpu: Any | None,
    use_cuda: bool,
    infer_repeats: int,
    smooth_iters_arg: int,
    csv_out_dir: Path,
    skip_first_set: set[str],
    flush_every: int,
    hw: dict[str, str],
    bem_warmup: bool = False,
    bem_defaults: bool = False,
) -> dict[str, Any] | None:
    """Deterministic per-frame sweep for one folder -> one CSV. Returns summary or None."""
    scene_dir = scene_dir.resolve()
    print(f"=== Scene: {scene_dir}", flush=True)
    paths, meshes_root = _iter_scene_obj_paths(scene_dir, mesh_override)
    smooth_iters = int(smooth_iters_arg)
    if smooth_iters < 0:
        smooth_iters = _scene_auto_smooth_iters(meshes_root)

    if not paths:
        print(f"  ERROR: no OBJs found under {scene_dir}", flush=True)
        return None

    if scene_dir.name in skip_first_set:
        skipped = paths[0].name
        paths = paths[1:]
        print(f"  skipping first frame: {skipped}", flush=True)

    print(
        f"  layout: {'lbm_tree ' + str(meshes_root) if meshes_root else 'flat *.obj'} | "
        f"{len(paths)} frame(s) | smooth_iters={smooth_iters}",
        flush=True,
    )

    smooth_cache_root = scene_dir / ".benchmark_nn8_bem_smooth_cache"

    # One throwaway BEM solve so bempp's first-call JIT/operator compilation (~tens of
    # seconds) is not charged to the first timed frame. Excluded from the CSV/averages.
    if bem_warmup and paths:
        wp = paths[0]
        print(f"  BEM warm-up (JIT) on {wp.name} — excluded from CSV...", flush=True)
        try:
            v_raw, f_raw = _prepare_mesh_vf(
                wp,
                meshes_root=meshes_root,
                smooth_iters=smooth_iters,
                smooth_cache_root=smooth_cache_root,
            )
            v_unit, f_unit = _to_unit_volume(v_raw, f_raw)
            warm_kwargs: dict[str, Any] = dict(
                trial_pair="P1-DP0", precond="mass", solver="gmres", verbose=False
            )
            if not bem_defaults:
                warm_kwargs.update(gmres_tol=1e-12, quadrature_regular=6, quadrature_singular=6)
            solve_minnaert_frequency_galerkin((v_unit, f_unit), **warm_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"  (BEM warm-up failed, continuing: {type(exc).__name__}: {exc})", flush=True)
    csv_out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_out_dir / f"nn8_vs_bem_{scene_dir.name}.csv"

    rows_acc: list[dict[str, Any]] = []
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        fh.flush()
        for i, p in enumerate(paths):
            try:
                row = _time_one_bubble(
                    p,
                    meshes_root=meshes_root,
                    smooth_iters=smooth_iters,
                    smooth_cache_root=smooth_cache_root,
                    nn_bundle=nn_bundle,
                    model_gpu=model_gpu,
                    use_cuda=use_cuda,
                    infer_repeats=infer_repeats,
                    bem_defaults=bem_defaults,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)
                continue

            gpu_infer = float(row.get("t_inference_gpu_s", float("nan")))
            t_nn_cpu = row["t_feat_s"] + row["t_inference_cpu_s"]
            t_nn_gpu = (
                row["t_feat_s"] + gpu_infer if math.isfinite(gpu_infer) else float("nan")
            )
            w.writerow(
                [
                    p.name,
                    row["n_vertices_unit"],
                    row["n_triangles"],
                    row["t_feat_s"],
                    row["t_inference_cpu_s"],
                    _csv_num(gpu_infer),
                    t_nn_cpu,
                    _csv_num(t_nn_gpu),
                    row["t_bem_assemble_s"],
                    row["t_bem_solve_s"],
                    row["t_bem_s"],
                ]
            )
            rows_acc.append(row)
            if flush_every > 0 and ((i + 1) % flush_every == 0):
                fh.flush()

            print(
                f"  [{i + 1}/{len(paths)}] {p.name}: v={row['n_vertices_unit']} "
                f"f={row['n_triangles']} feat={row['t_feat_s']:.4f}s "
                f"inferCPU={row['t_inference_cpu_s']:.5f}s inferGPU={gpu_infer:.5f}s "
                f"BEMsolve={row['t_bem_solve_s']:.4f}s BEMtot={row['t_bem_s']:.4f}s",
                flush=True,
            )

        if not rows_acc:
            print("  ERROR: no frames succeeded; CSV has header only.", flush=True)
            return None

        def _mean(key: str) -> float:
            return float(np.mean([r[key] for r in rows_acc]))

        gpu_finite = [
            float(r["t_inference_gpu_s"])
            for r in rows_acc
            if math.isfinite(float(r.get("t_inference_gpu_s", float("nan"))))
        ]
        avg_feat = _mean("t_feat_s")
        avg_infer_cpu = _mean("t_inference_cpu_s")
        avg_infer_gpu = float(np.mean(gpu_finite)) if gpu_finite else float("nan")
        avg_bem_asm = _mean("t_bem_assemble_s")
        avg_bem_solve = _mean("t_bem_solve_s")
        avg_bem_tot = _mean("t_bem_s")
        avg_nn_cpu = avg_feat + avg_infer_cpu
        avg_nn_gpu = avg_feat + avg_infer_gpu if math.isfinite(avg_infer_gpu) else float("nan")

        w.writerow([])
        w.writerow(
            [
                "AVERAGE",
                "",
                "",
                avg_feat,
                avg_infer_cpu,
                _csv_num(avg_infer_gpu),
                avg_nn_cpu,
                _csv_num(avg_nn_gpu),
                avg_bem_asm,
                avg_bem_solve,
                avg_bem_tot,
            ]
        )
        w.writerow(["N_FRAMES", len(rows_acc)])
        w.writerow(["HARDWARE_CPU", hw["cpu"]])
        w.writerow(["HARDWARE_GPU", hw["gpu"]])
        w.writerow(["NN8_INFER_DEVICES", hw["nn8_infer_devices"]])
        fh.flush()

    print(
        f"  ----\n"
        f"  Wrote {csv_path} ({len(rows_acc)} frame rows)\n"
        f"    avg NN8 feature-extraction: {avg_feat:.5f} s\n"
        f"    avg NN8 inference (CPU):    {avg_infer_cpu:.6f} s\n"
        f"    avg NN8 inference (GPU):    {avg_infer_gpu:.6f} s\n"
        f"    avg NN8 total  CPU/GPU:     {avg_nn_cpu:.5f} / {avg_nn_gpu:.5f} s\n"
        f"    avg BEM solve:              {avg_bem_solve:.5f} s\n"
        f"    avg BEM assemble/total:     {avg_bem_asm:.5f} / {avg_bem_tot:.5f} s\n"
        f"    hardware: CPU={hw['cpu']} | GPU={hw['gpu']} | infer={hw['nn8_infer_devices']}\n",
        flush=True,
    )

    return {
        "scene": str(scene_dir),
        "csv_path": str(csv_path),
        "n_frames": len(rows_acc),
        "avg_feat_s": avg_feat,
        "avg_infer_cpu_s": avg_infer_cpu,
        "avg_infer_gpu_s": avg_infer_gpu,
        "avg_nn_total_cpu_s": avg_nn_cpu,
        "avg_nn_total_gpu_s": avg_nn_gpu,
        "avg_bem_assemble_s": avg_bem_asm,
        "avg_bem_solve_s": avg_bem_solve,
        "avg_bem_total_s": avg_bem_tot,
        "hardware": hw,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=list(DEFAULT_SCENE_SPECS),
        help=(
            "Scene dirs, optionally 'scene@/path/to/mc_surface_...' for an external "
            "bub_*/frame_*.obj tree. The exhale/fruits defaults look under "
            "$BUBBLEGYM_LBM_ROOT, which is where the separate LBM simulator writes."
        ),
    )
    ap.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="NN8 artifact directory (train_config.json + checkpoint + scalers).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--num-bubbles",
        type=int,
        default=3,
        help="Randomly pick this many meshes per scene (default: 3).",
    )
    ap.add_argument(
        "--min-vertices",
        type=int,
        default=3000,
        help="Only consider OBJs whose vertex count exceeds this (default: 3000).",
    )
    ap.add_argument(
        "--smooth-iters",
        type=int,
        default=-1,
        help="Laplacian smoothing iters for LBM tree layout; -1 = auto "
        "(0 if mesh path contains 'smoothed' or flat layout, else 3 for raw LBM tree).",
    )
    ap.add_argument(
        "--bem-warmup",
        action="store_true",
        help="Run an extra P1-DP0 solve on the first selected mesh and exclude it from averages.",
    )
    ap.add_argument(
        "--all-bubbles",
        action="store_true",
        help="Time every candidate mesh in each scene (ignores --num-bubbles).",
    )
    ap.add_argument(
        "--infer-repeats",
        type=int,
        default=100,
        help="Repeats for averaging each NN forward pass (default: 100).",
    )
    ap.add_argument(
        "--nn-device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Primary device for NN8 inference timing (auto -> cuda if available). "
        "The CPU forward time is always recorded too.",
    )
    ap.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one discarded NN+BEM pass on the first selected mesh per scene "
        "(steady-state timings) WITHOUT dropping it from the CSV. Default: on.",
    )
    ap.add_argument(
        "--csv-out-dir",
        type=Path,
        default=None,
        help="If set, write one per-bubble CSV per scene into this directory "
        "(--csv-all mode defaults to results/experiments when unset).",
    )
    ap.add_argument(
        "--csv-flush-every",
        type=int,
        default=5,
        help="Checkpoint the scene CSV after every N timed bubbles so progress "
        "survives an interruption (default: 5; also the flush cadence in --csv-all mode).",
    )
    ap.add_argument(
        "--hardware-note",
        type=str,
        default="",
        help="Optional free-text note recorded in the CSV 'hardware_note' column.",
    )
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="If set, write a machine-readable summary to this path.",
    )
    ap.add_argument(
        "--vertex-scan-limit",
        type=int,
        default=-1,
        help=(
            "Max OBJ files to scan for --min-vertices filtering. "
            "-1 (default) = if more than 25000 paths exist, randomly sample 25000; "
            "0 = scan every path (slow for huge LBM trees; can take tens of minutes)."
        ),
    )
    ap.add_argument(
        "--csv-all",
        action="store_true",
        help="Deterministic mode: time EVERY frame per scene folder in filename order "
        "(no random sampling / --min-vertices filter) and write one CSV per folder.",
    )
    ap.add_argument(
        "--skip-first-frames",
        type=str,
        default="enright",
        help="Comma-separated scene folder names whose first frame is skipped "
        "(--csv-all mode). Default: 'enright'.",
    )
    ap.add_argument(
        "--bem-defaults",
        action="store_true",
        help="Call the Galerkin solver with its native defaults (relaxed gmres tol, "
        "lower quadrature) instead of the benchmark's explicit tol=1e-12, quad 6/6.",
    )
    args = ap.parse_args()

    rng = random.Random(int(args.seed))

    print(
        "NN8 vs BEM timing\n"
        f"  artifacts: {args.artifacts}\n"
        f"  min_vertices: {args.min_vertices} | num_bubbles: {args.num_bubbles} | seed: {args.seed}\n",
        flush=True,
    )

    t_load0 = time.perf_counter()
    nn_bundle = load_nn_chull_inertia_8feat_direct(Path(args.artifacts))
    model_nn, _fs_w, _ts_w, cols_w = nn_bundle
    with _torch.no_grad():
        _ = model_nn(_torch.zeros(1, len(cols_w), dtype=_torch.float32))

    cuda_available = bool(_torch.cuda.is_available())
    if args.nn_device == "cuda" and not cuda_available:
        print("  WARNING: --nn-device cuda requested but no CUDA device; using CPU.", flush=True)
    use_cuda = (args.nn_device == "cuda" or args.nn_device == "auto") and cuda_available
    model_gpu = None
    if use_cuda:
        model_gpu = copy.deepcopy(model_nn).to("cuda").eval()
        with _torch.no_grad():  # upload weights + init CUDA context once
            _ = model_gpu(_torch.zeros(1, len(cols_w), dtype=_torch.float32, device="cuda"))
        _torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load0

    hardware = _detect_hardware()
    print(
        f"  hardware: CPU={hardware['cpu']} | GPU={hardware['gpu']}\n"
        f"  nn_inference_device: {'cuda' if use_cuda else 'cpu'} | "
        f"infer_repeats: {args.infer_repeats}",
        flush=True,
    )

    if args.csv_all:
        hw = dict(hardware)
        hw["nn8_infer_devices"] = "cpu+cuda" if use_cuda else "cpu"
        skip_set = {s.strip() for s in str(args.skip_first_frames).split(",") if s.strip()}
        scene_specs = args.scenes
        # Bare `--csv-all` (default scene list): use the two procedural target folders.
        if scene_specs == list(DEFAULT_SCENE_SPECS):
            scene_specs = [
                str(REPO_ROOT / "dataset/bubble_theater/enright_test/mesh"),
                str(REPO_ROOT / "dataset/bubble_theater/curl_noise/mesh"),
            ]
        csv_out_dir = (
            Path(args.csv_out_dir)
            if args.csv_out_dir is not None
            else PYTHON_ROOT / "utils" / "results"
        )
        print(
            f"CSV-all mode | out_dir={csv_out_dir} | skip_first={sorted(skip_set)} | "
            f"flush_every={args.csv_flush_every} | "
            f"bem_settings={'solver-native-defaults' if args.bem_defaults else 'tol=1e-12,quad=6/6'}\n"
            f"  (NN8 model load + warm-up: {t_load:.4f} s)\n",
            flush=True,
        )
        csv_summaries: list[dict[str, Any]] = []
        for scene_idx, scene_spec in enumerate(scene_specs):
            scene_dir, mesh_override = _parse_scene_spec(scene_spec)
            summary = _run_csv_all_for_scene(
                scene_dir,
                mesh_override,
                nn_bundle=nn_bundle,
                model_gpu=model_gpu,
                use_cuda=use_cuda,
                infer_repeats=int(args.infer_repeats),
                smooth_iters_arg=int(args.smooth_iters),
                csv_out_dir=csv_out_dir,
                skip_first_set=skip_set,
                flush_every=int(args.csv_flush_every),
                hw=hw,
                bem_warmup=(scene_idx == 0),
                bem_defaults=bool(args.bem_defaults),
            )
            if summary is not None:
                csv_summaries.append(summary)
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps({"mode": "csv_all", "hardware": hw, "scenes": csv_summaries}, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {args.json_out}", flush=True)
        return 0

    all_scenes_summary: list[dict[str, Any]] = []

    for scene_spec in args.scenes:
        scene_dir, mesh_override = _parse_scene_spec(scene_spec)
        scene_dir = scene_dir.resolve()
        print(f"=== Scene: {scene_dir}", flush=True)
        if mesh_override is not None:
            print(f"  meshes_root (override): {mesh_override}", flush=True)
        paths, meshes_root = _iter_scene_obj_paths(scene_dir, mesh_override)
        smooth_iters = int(args.smooth_iters)
        if smooth_iters < 0:
            smooth_iters = _scene_auto_smooth_iters(meshes_root)

        if not paths:
            loc = scene_dir / DEFAULT_MESH_SUBDIR
            if mesh_override is not None:
                msg = (
                    f"No OBJs under meshes_root {mesh_override.resolve()!s} "
                    f"(override for scene {scene_dir})."
                )
            else:
                msg = (
                    f"No OBJs found under {scene_dir}.\n"
                    f"  For LBM scenes, export meshes to {loc} (bub_<id>/frame_<n>.obj) "
                    f"or pass scene@{DEFAULT_MESH_SUBDIR.as_posix()}."
                )
            print(f"  ERROR: {msg}", flush=True)
            all_scenes_summary.append(
                {
                    "scene": str(scene_dir),
                    "error": msg,
                    "mesh_layout": "lbm_tree" if meshes_root else "flat",
                }
            )
            continue

        print(
            f"  layout: {'lbm_tree ' + str(meshes_root) if meshes_root else 'flat *.obj'} | "
            f"discovered {len(paths)} OBJ(s) | smooth_iters={smooth_iters}",
            flush=True,
        )

        limit = int(args.vertex_scan_limit)
        if limit == 0:
            paths_for_scan = paths
        elif limit > 0:
            if len(paths) > limit:
                paths_for_scan = rng.sample(paths, k=limit)
                print(
                    f"  vertex filter: random sample {len(paths_for_scan)} of {len(paths)} OBJs "
                    f"(--vertex-scan-limit {limit})",
                    flush=True,
                )
            else:
                paths_for_scan = paths
        else:
            # auto (-1)
            if len(paths) > VERTEX_SCAN_AUTO_THRESHOLD:
                paths_for_scan = rng.sample(paths, k=VERTEX_SCAN_AUTO_CAP)
                print(
                    f"  vertex filter: random sample {len(paths_for_scan)} of {len(paths)} OBJs "
                    f"(auto cap; use --vertex-scan-limit 0 to scan all)",
                    flush=True,
                )
            else:
                paths_for_scan = paths

        candidates: list[Path] = []
        for p in paths_for_scan:
            try:
                nv = _n_vertices(p)
            except Exception as exc:  # noqa: BLE001
                print(f"  skip (load error) {p.name}: {exc}", flush=True)
                continue
            if nv > int(args.min_vertices):
                candidates.append(p)

        if not candidates:
            msg = f"No meshes with >{args.min_vertices} vertices."
            print(f"  ERROR: {msg}", flush=True)
            all_scenes_summary.append(
                {
                    "scene": str(scene_dir),
                    "error": msg,
                    "mesh_layout": "lbm_tree" if meshes_root else "flat",
                    "candidates": 0,
                }
            )
            continue

        if args.all_bubbles:
            picked = sorted(candidates, key=lambda p: p.name)
            print(
                f"  candidates: {len(candidates)} | timing ALL {len(picked)} (--all-bubbles)",
                flush=True,
            )
        else:
            k = min(int(args.num_bubbles), len(candidates))
            picked = rng.sample(candidates, k=k)
            print(
                f"  candidates: {len(candidates)} | picked {k}: "
                + ", ".join(f"{p.name}({_n_vertices(p)}v)" for p in picked),
                flush=True,
            )

        smooth_cache_root = scene_dir / ".benchmark_nn8_bem_smooth_cache"

        scene_rows: list[dict[str, Any]] = []

        if (args.bem_warmup or args.warmup) and picked:
            wpath = picked[0]
            excluded = bool(args.bem_warmup)
            tag = "excluded from averages" if excluded else "still timed below"
            print(f"  warm-up (NN+BEM) on {wpath.name} ({tag})...", flush=True)
            v_raw, f_raw = _prepare_mesh_vf(
                wpath,
                meshes_root=meshes_root,
                smooth_iters=smooth_iters,
                smooth_cache_root=smooth_cache_root,
            )
            v_unit, f_unit = _to_unit_volume(v_raw, f_raw)
            _mass, _com, inertia_u = compute_mirtich_moments(v_unit, f_unit)
            principal = np.linalg.eigvalsh(inertia_u).astype(np.float64)
            principal.sort()
            chull = nonspherical_features_from_unit_mesh(v_unit, f_unit)
            Xn = _build_nn8_input(
                chull, principal, feature_scaler=_fs_w, feature_cols=cols_w
            )
            _time_nn8_inference(
                Xn,
                model_cpu=model_nn,
                model_gpu=model_gpu,
                repeats=max(1, int(args.infer_repeats) // 5),
                use_cuda=use_cuda,
            )
            solve_minnaert_frequency_galerkin(
                (v_unit, f_unit),
                trial_pair="P1-DP0",
                precond="mass",
                solver="gmres",
                gmres_tol=1e-12,
                quadrature_regular=6,
                quadrature_singular=6,
                verbose=False,
            )
            picked_eff = picked[1:] if excluded else picked
        else:
            picked_eff = picked

        if not picked_eff:
            print("  No meshes left after warm-up; nothing to average.", flush=True)
            continue

        for p in picked_eff:
            try:
                row = _time_one_bubble(
                    p,
                    meshes_root=meshes_root,
                    smooth_iters=smooth_iters,
                    smooth_cache_root=smooth_cache_root,
                    nn_bundle=nn_bundle,
                    model_gpu=model_gpu,
                    use_cuda=use_cuda,
                    infer_repeats=int(args.infer_repeats),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAIL {p.name}: {type(exc).__name__}: {exc}", flush=True)
                continue
            scene_rows.append(row)
            print(
                f"  {p.name}: BEM={row['t_bem_s']:.4f}s  "
                f"feat={row['t_feat_s']:.4f}s  "
                f"infer_gpu={row['t_inference_gpu_s']*1e3:.4f}ms  "
                f"infer_cpu={row['t_inference_cpu_s']*1e3:.4f}ms  "
                f"f_BEM={row['f_hz_bem']:.4g} Hz  f_NN8@V=1={row['f_hz_nn8_unit']:.4g} Hz",
                flush=True,
            )
            if (
                args.csv_out_dir is not None
                and int(args.csv_flush_every) > 0
                and len(scene_rows) % int(args.csv_flush_every) == 0
            ):
                cp = _write_scene_csv(
                    Path(args.csv_out_dir),
                    scene_dir=scene_dir,
                    scene_rows=scene_rows,
                    hardware=hardware,
                    nn_device="cuda" if use_cuda else "cpu",
                    hardware_note=str(args.hardware_note),
                )
                print(
                    f"    [checkpoint] wrote {len(scene_rows)} rows -> {cp}",
                    flush=True,
                )

        if not scene_rows:
            all_scenes_summary.append(
                {
                    "scene": str(scene_dir),
                    "error": "all selected meshes failed",
                    "mesh_layout": "lbm_tree" if meshes_root else "flat",
                }
            )
            continue

        arr_bem = np.array([r["t_bem_s"] for r in scene_rows], dtype=np.float64)
        arr_feat = np.array([r["t_feat_s"] for r in scene_rows], dtype=np.float64)
        arr_inf_gpu = np.array([r["t_inference_gpu_s"] for r in scene_rows], dtype=np.float64)
        arr_inf_cpu = np.array([r["t_inference_cpu_s"] for r in scene_rows], dtype=np.float64)
        nn_total_gpu = np.array([r["t_nn8_total_gpu_s"] for r in scene_rows], dtype=np.float64)
        nn_total_cpu = np.array([r["t_nn8_total_cpu_s"] for r in scene_rows], dtype=np.float64)
        speedup_gpu = float(np.mean(arr_bem) / max(1e-30, np.mean(nn_total_gpu)))
        speedup_cpu = float(np.mean(arr_bem) / max(1e-30, np.mean(nn_total_cpu)))

        summary = {
            "scene": str(scene_dir),
            "meshes_root": str(meshes_root.resolve()) if meshes_root else None,
            "mesh_layout": "lbm_tree" if meshes_root else "flat",
            "smooth_iters": smooth_iters,
            "n_averaged": len(scene_rows),
            "nn_inference_device": "cuda" if use_cuda else "cpu",
            "infer_repeats": int(args.infer_repeats),
            "mean_bem_s": float(np.mean(arr_bem)),
            "mean_nn_feat_s": float(np.mean(arr_feat)),
            "mean_nn_infer_gpu_s": float(np.mean(arr_inf_gpu)),
            "mean_nn_infer_cpu_s": float(np.mean(arr_inf_cpu)),
            "mean_nn_total_gpu_s": float(np.mean(nn_total_gpu)),
            "mean_nn_total_cpu_s": float(np.mean(nn_total_cpu)),
            "nn_speedup_bem_over_nn_total_gpu": speedup_gpu,
            "nn_speedup_bem_over_nn_total_cpu": speedup_cpu,
            "model_load_s": float(t_load),
            "hardware_cpu": hardware["cpu"],
            "hardware_gpu": hardware["gpu"],
            "rows": scene_rows,
        }
        all_scenes_summary.append(summary)

        if args.csv_out_dir is not None:
            csv_path = _write_scene_csv(
                Path(args.csv_out_dir),
                scene_dir=scene_dir,
                scene_rows=scene_rows,
                hardware=hardware,
                nn_device="cuda" if use_cuda else "cpu",
                hardware_note=str(args.hardware_note),
            )
            print(f"  wrote CSV: {csv_path}", flush=True)

        print(
            f"  ----\n"
            f"  Averages over {len(scene_rows)} bubble(s):\n"
            f"    BEM (assemble+solve):     {summary['mean_bem_s']:.4f} s\n"
            f"    NN8 features:             {summary['mean_nn_feat_s']:.4f} s\n"
            f"    NN8 inference (GPU full): {summary['mean_nn_infer_gpu_s']*1e3:.4f} ms\n"
            f"    NN8 inference (CPU):      {summary['mean_nn_infer_cpu_s']*1e3:.4f} ms\n"
            f"    Speedup BEM/NN8(GPU):     {speedup_gpu:.2f}x\n"
            f"    Speedup BEM/NN8(CPU):     {speedup_cpu:.2f}x\n",
            flush=True,
        )

    print(
        f"(NN8 model load + torch warm-up once: {t_load:.4f} s "
        f"- not included in per-bubble inference means)",
        flush=True,
    )

    if args.json_out is not None:
        payload = {
            "artifacts": str(args.artifacts),
            "nn8_model_load_s": t_load,
            "scenes": all_scenes_summary,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
