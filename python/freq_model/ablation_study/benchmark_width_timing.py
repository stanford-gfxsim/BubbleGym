"""Time the MLP width ablation under the paper's Table 1 inference protocol.

Table 1 times inference at batch 512 on the GPU and quotes both the forward pass
alone and the whole path including host->device transfer and readback. This
measures both for every width variant in ``results/experiments/supp_table02_width_ablation``.

Two measurement traps:

* **Interleaving.** Warm-GPU run-to-run scatter (10-20%) exceeds the spread
  across the five widths, so timing in per-variant blocks lets one slow period
  manufacture a trend. One pass of every variant, then repeat, spreads drift.
* **TorchScript.** The model is small enough that Python dispatch dominates
  kernel time, so an eager-mode measurement reads ~20% high. Production exports
  to TorchScript, so the models are scripted here to match.

Usage (from the repository root; see ``README_timing.md`` for the procedure):

    python -m python.freq_model.ablation_study.benchmark_width_timing \
        [--batch 512 --passes 7 --out my_machine_timing.json]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from python.freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402

EXP2 = _HERE.parents[2] / "results" / "experiments" / "supp_table02_width_ablation"
DEFAULT_VARIANTS = ("h16_h8", "h32_h16", "h64_h32", "h128_h64", "h256_h128")
INPUT_DIM = 8


# --------------------------------------------------------------------------- #
# environment capture
# --------------------------------------------------------------------------- #
def _cpu_name() -> str:
    """Best-effort CPU model string; the call is dispatch-bound, so this matters."""
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["wmic", "cpu", "get", "name"],
                                 capture_output=True, text=True, timeout=10).stdout
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def environment(device: str) -> dict:
    env = {
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "cpu": _cpu_name(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda"):
        try:
            env["cuda"] = torch.version.cuda
            env["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            pass
    return env


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_variant(variant: str, device: str) -> tuple[torch.nn.Module, int]:
    """Load one width variant's released checkpoint and TorchScript it."""
    vdir = EXP2 / variant
    cfg = json.loads((vdir / "train_config.json").read_text())
    net = BubbleFreqNet(
        input_dim=int(cfg.get("input_dim", INPUT_DIM)),
        hidden_dim=int(cfg["hidden_dim"]),
        hidden_dim2=int(cfg["hidden_dim2"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )
    obj = torch.load(vdir / "bubble_freq_net_best.pt", map_location="cpu",
                     weights_only=False)
    for key in ("model_state_dict", "state_dict"):
        if isinstance(obj, dict) and key in obj:
            obj = obj[key]
            break
    net.load_state_dict(obj)
    n_params = sum(p.numel() for p in net.parameters())

    net = net.to(device).eval()
    with torch.no_grad():
        scripted = torch.jit.freeze(torch.jit.script(net))
        scripted(torch.randn(8, INPUT_DIM, device=device))   # trigger fusion passes
    return scripted, n_params


# --------------------------------------------------------------------------- #
# the two measurements Table 1 reports
# --------------------------------------------------------------------------- #
def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_forward(model, x, device: str, warmup: int, repeats: int) -> float:
    """Forward pass only, batch already resident on the device. Returns ms."""
    with torch.no_grad():
        _sync(device)
        for _ in range(warmup):
            model(x)
        _sync(device)
        if device.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            ts = []
            for _ in range(repeats):
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                ts.append(float(start.elapsed_time(end)))
        else:
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts))


def time_full_path(model, x_cpu, device: str, warmup: int, repeats: int) -> float:
    """Host->device transfer + forward + readback, synchronized. Returns ms."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x_cpu.to(device, non_blocking=True)).cpu()
        _sync(device)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x_cpu.to(device, non_blocking=True)).cpu()
            _sync(device)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(ts))


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Width-ablation inference timing at the paper's Table 1 protocol.")
    ap.add_argument("--batch", type=int, default=512,
                    help="batch size; 512 is the main paper's Table 1 protocol")
    ap.add_argument("--passes", type=int, default=7,
                    help="interleaved passes over all variants (>=5 recommended)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=150)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    ap.add_argument(
        "--out",
        type=Path,
        default=EXP2 / "width_timing.json",
    )
    args = ap.parse_args()

    if not EXP2.is_dir():
        print(f"ERROR: {EXP2} not found. The width checkpoints ship with the repo; "
              "check that this is a full clone.", file=sys.stderr)
        return 2

    env = environment(args.device)
    print("machine")
    for k, v in env.items():
        if v:
            print(f"  {k:8s} {v}")
    print(f"\nprotocol: batch {args.batch}, TorchScript forward, "
          f"{args.warmup} warmup + {args.repeats} repeats, "
          f"{args.passes} interleaved passes\n")

    models, params = {}, {}
    for v in args.variants:
        models[v], params[v] = load_variant(v, args.device)

    x = torch.randn(args.batch, INPUT_DIM, device=args.device)
    x_cpu = torch.randn(args.batch, INPUT_DIM)

    fwd = {v: [] for v in args.variants}
    full = {v: [] for v in args.variants}
    for p in range(args.passes):
        for v in args.variants:            # interleaved: drift hits every variant
            fwd[v].append(time_forward(models[v], x, args.device,
                                       args.warmup, args.repeats) * 1000.0 / args.batch)
            full[v].append(time_full_path(models[v], x_cpu, args.device,
                                          args.warmup, args.repeats) * 1000.0 / args.batch)
        print(f"  pass {p + 1}/{args.passes} done")

    hdr = (f"\n{'variant':>11} {'#params':>9} | {'forward us/bubble':>21} | "
           f"{'full path us/bubble':>21}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    results = {}
    for v in args.variants:
        f, t = np.array(fwd[v]), np.array(full[v])
        results[v] = {
            "n_params": params[v],
            "forward_us_per_bubble_mean": float(f.mean()),
            "forward_us_per_bubble_std": float(f.std()),
            "fullpath_us_per_bubble_mean": float(t.mean()),
            "fullpath_us_per_bubble_std": float(t.std()),
            "forward_us_per_bubble_passes": [float(a) for a in f],
            "fullpath_us_per_bubble_passes": [float(a) for a in t],
        }
        print(f"{v:>11} {params[v]:>9,} | {f.mean():10.3f} +/- {f.std():<8.3f} | "
              f"{t.mean():10.3f} +/- {t.std():<8.3f}")

    # The claim under test is "cost does not grow with width", so compare the
    # widest variant against the narrowest. Max-minus-min across all five would
    # trip on any variant that happens to run fast, which is a different question.
    order = sorted(args.variants, key=lambda v: results[v]["n_params"])
    lo, hi = order[0], order[-1]
    d_lo = results[lo]["forward_us_per_bubble_mean"]
    d_hi = results[hi]["forward_us_per_bubble_mean"]
    pooled = float(np.hypot(results[lo]["forward_us_per_bubble_std"],
                            results[hi]["forward_us_per_bubble_std"]))
    delta = d_hi - d_lo
    ratio = results[hi]["n_params"] / results[lo]["n_params"]
    print(f"\nwidest vs narrowest ({hi} vs {lo}, {ratio:.0f}x the parameters):")
    print(f"  {d_hi:.3f} vs {d_lo:.3f} us/bubble -> {delta:+.3f} us "
          f"({delta / d_lo * 100:+.0f}%), pooled scatter {pooled:.3f} us")
    if abs(delta) <= 2 * pooled:
        print("-> no width penalty: the difference is within run-to-run scatter, "
              "so width is effectively free at this batch size.")
    else:
        print("-> WIDTH PENALTY: the difference exceeds the scatter. Report this; "
              "it contradicts the current supplement text.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"environment": env,
         "protocol": {"batch": args.batch, "passes": args.passes,
                      "warmup": args.warmup, "repeats": args.repeats,
                      "scripted": True},
         "results": results}, indent=2))
    print(f"\nwrote {args.out}")
    print("Send this file back -- it carries the machine identification with it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
