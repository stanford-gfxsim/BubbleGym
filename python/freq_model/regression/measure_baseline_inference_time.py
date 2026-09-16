"""Measure inference time of the baseline regressors (+ the 8-feature MLP).

Companion to ``baseline_regressors_8feat.py``: re-fits the four regressors at
the recorded hyperparameters (asserting the refit test MAPE matches), then times
the held-out test split three ways. All three are verified to agree numerically
before being timed.

* **framework**: ``StandardScaler.transform -> model.predict`` (for the MLP:
  transform -> tensor -> forward -> inverse target scaling). At batch 1 this is
  dominated by fixed per-call overhead, not arithmetic.
* **raw**: the same math in plain numpy with the scalers folded into the
  coefficients. Still pays ~1-2 us of numpy dispatch per op at batch 1.
* **compiled**: the folded math as explicit ``numba.njit`` loops, timed inside a
  compiled driver that cycles through the test set so the work cannot be
  hoisted. This is what a C++ caller would see. Skipped without numba.

Batch sizes 1 and the full test set are reported. The CUDA MLP row keeps the
framework path only (launch overhead is intrinsic there). ``--clamp-to-minnaert``
(OFF by default) adds the [f_Minnaert, 1.35 x] clamp to every timed path so the
cost is measured, not estimated.

RUN (repo root; $env:KMP_DUPLICATE_LIB_OK = "TRUE" on Windows):
    python -u python/freq_model/regression/measure_baseline_inference_time.py
"""

from __future__ import annotations

import torch  # noqa: F401  (torch first: Windows OpenMP DLL ordering)

import argparse
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import erf

_THIS = Path(__file__).resolve()
_FREQ_MODEL_DIR = _THIS.parents[1]
_PYTHON_ROOT = _FREQ_MODEL_DIR.parent
_REPO_ROOT = _PYTHON_ROOT.parent
for _p in (_PYTHON_ROOT, _FREQ_MODEL_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sklearn.kernel_ridge import KernelRidge  # noqa: E402
from sklearn.linear_model import LinearRegression, Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

from freq_model.regression.baseline_regressors_8feat import (  # noqa: E402
    CLAMP_FLAG_HELP,
    MINNAERT_CLAMP_FACTOR,
    check_split_against_paper,
    clamp_log_predictions,
    load_xy,
    metrics_from_log,
    minnaert_log_clamp_bounds,
)
from freq_model.NN.bub_freq_net import BubbleFreqNet  # noqa: E402
from freq_model.NN.bubble_dataset import split_dataset  # noqa: E402
from freq_model.NN.fit_shape_freq_model import (  # noqa: E402
    FEATURE_COLS,
)

DEFAULT_METRICS_JSON = (
    _REPO_ROOT / "results" / "baseline_regressors_8feat_bubblegym_10k" / "baseline_metrics.json"
)
DEFAULT_MODEL_DIR = (
    _FREQ_MODEL_DIR / "output" / "output_8feature_direct_bubblegym_10k"
)
DEFAULT_DATASET_10K = _REPO_ROOT / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv"


def time_call(fn, *, warmup: int, repeats: int) -> dict[str, float]:
    """Warm up, then time ``fn()`` ``repeats`` times; return per-call stats [ms]."""
    for _ in range(warmup):
        fn()
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples[i] = (time.perf_counter() - t0) * 1e3
    return {
        "mean_ms": float(samples.mean()),
        "std_ms": float(samples.std()),
        "median_ms": float(np.median(samples)),
        "min_ms": float(samples.min()),
    }


# ---------------------------------------------------------------------------
# Raw (framework-free) predictors: feature scaler folded into coefficients.
# All take/return float64 numpy arrays of shape (n, 8) -> (n,).
# ---------------------------------------------------------------------------

def make_raw_linear(model: LinearRegression, mu: np.ndarray, sig: np.ndarray):
    w = model.coef_ / sig
    b = float(model.intercept_ - (mu / sig) @ model.coef_)
    return lambda x: x @ w + b


def _monomial_index_table(powers: np.ndarray, n_features: int) -> np.ndarray:
    """(n_out, degree) column indices into z padded with a ones column.

    Each monomial ``prod_j z_j^p_j`` becomes ``degree`` factors: feature j
    repeated ``p_j`` times, padded with index ``n_features`` (the ones column).
    """
    degree = int(powers.sum(axis=1).max())
    table = np.full((powers.shape[0], degree), n_features, dtype=np.int64)
    for r, row in enumerate(powers):
        k = 0
        for j, p in enumerate(row):
            for _ in range(int(p)):
                table[r, k] = j
                k += 1
    return table


def make_raw_poly(pipeline, mu: np.ndarray, sig: np.ndarray):
    pf: PolynomialFeatures = pipeline.steps[0][1]
    reg = pipeline.steps[-1][1]
    table = _monomial_index_table(pf.powers_, len(mu))
    coef = reg.coef_.astype(np.float64)
    b = float(reg.intercept_)

    def predict(x: np.ndarray) -> np.ndarray:
        z = (x - mu) / sig
        z_aug = np.concatenate([z, np.ones((z.shape[0], 1))], axis=1)
        feats = np.prod(z_aug[:, table], axis=2)
        return feats @ coef + b

    return predict


def make_raw_rbf(model: KernelRidge, mu: np.ndarray, sig: np.ndarray):
    z_fit = model.X_fit_.astype(np.float64)          # already standardized
    dual = model.dual_coef_.astype(np.float64)
    gamma = float(model.gamma)
    fit_sq = np.einsum("ij,ij->i", z_fit, z_fit)

    def predict(x: np.ndarray) -> np.ndarray:
        z = (x - mu) / sig
        sq = (
            np.einsum("ij,ij->i", z, z)[:, None]
            + fit_sq[None, :]
            - 2.0 * (z @ z_fit.T)
        )
        return np.exp(-gamma * sq) @ dual

    return predict


def make_raw_mlp(net: BubbleFreqNet, mu, sig, y_mean: float, y_scale: float):
    """Numpy forward of the eval-mode BubbleFreqNet in float32.

    The feature scaler is folded into the first Linear and the target scaler
    into the head; LayerNorm and exact-erf GELU match torch eval semantics.
    """
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    w1, b1 = sd["input_proj.0.weight"], sd["input_proj.0.bias"]
    w1f = (w1 / sig[None, :]).astype(np.float32)
    b1f = (b1 - w1 @ (mu / sig)).astype(np.float32)
    g1, be1 = sd["input_proj.1.weight"].astype(np.float32), sd["input_proj.1.bias"].astype(np.float32)
    w2, b2 = sd["block1_linear.weight"].astype(np.float32), sd["block1_linear.bias"].astype(np.float32)
    g2, be2 = sd["block1_norm.weight"].astype(np.float32), sd["block1_norm.bias"].astype(np.float32)
    w3, b3 = sd["block2.0.weight"].astype(np.float32), sd["block2.0.bias"].astype(np.float32)
    g3, be3 = sd["block2.1.weight"].astype(np.float32), sd["block2.1.bias"].astype(np.float32)
    wh = (sd["head.weight"] * y_scale).astype(np.float32)
    bh = np.float32(sd["head.bias"][0] * y_scale + y_mean)
    inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
    eps = np.float32(1e-5)

    def _ln_gelu(h, gamma, beta):
        m = h.mean(axis=1, keepdims=True)
        v = h.var(axis=1, keepdims=True)
        h = (h - m) / np.sqrt(v + eps) * gamma + beta
        return np.float32(0.5) * h * (np.float32(1.0) + erf(h * inv_sqrt2))

    def predict(x: np.ndarray) -> np.ndarray:
        h = _ln_gelu(x.astype(np.float32) @ w1f.T + b1f, g1, be1)
        h = _ln_gelu(h @ w2.T + b2, g2, be2) + h
        h = _ln_gelu(h @ w3.T + b3, g3, be3)
        return (h @ wh.T + bh).ravel().astype(np.float64)

    return predict


# ---------------------------------------------------------------------------
# Compiled (numba) predictors: identical folded math as explicit loops.
# Each factory returns a scalar single-query function ``pred(x_row) -> float``.
# ---------------------------------------------------------------------------

def make_compiled_linear(model: LinearRegression, mu: np.ndarray, sig: np.ndarray):
    w = np.ascontiguousarray(model.coef_ / sig)
    b = float(model.intercept_ - (mu / sig) @ model.coef_)

    @njit(fastmath=False)
    def pred(x):
        acc = b
        for j in range(w.shape[0]):
            acc += w[j] * x[j]
        return acc

    return pred


def make_compiled_poly(pipeline, mu: np.ndarray, sig: np.ndarray):
    pf: PolynomialFeatures = pipeline.steps[0][1]
    reg = pipeline.steps[-1][1]
    table = _monomial_index_table(pf.powers_, len(mu))
    coef = np.ascontiguousarray(reg.coef_, dtype=np.float64)
    b = float(reg.intercept_)
    mu_ = np.ascontiguousarray(mu)
    sig_ = np.ascontiguousarray(sig)
    nf = len(mu)
    degree = table.shape[1]

    @njit(fastmath=False)
    def pred(x):
        z = np.empty(nf + 1)
        for j in range(nf):
            z[j] = (x[j] - mu_[j]) / sig_[j]
        z[nf] = 1.0
        acc = b
        for f in range(coef.shape[0]):
            p = 1.0
            for k in range(degree):
                p *= z[table[f, k]]
            acc += coef[f] * p
        return acc

    return pred


def make_compiled_rbf(model: KernelRidge, mu: np.ndarray, sig: np.ndarray):
    z_fit = np.ascontiguousarray(model.X_fit_, dtype=np.float64)
    dual = np.ascontiguousarray(model.dual_coef_, dtype=np.float64)
    gamma = float(model.gamma)
    mu_ = np.ascontiguousarray(mu)
    sig_ = np.ascontiguousarray(sig)
    nf = len(mu)

    @njit(fastmath=False)
    def pred(x):
        z = np.empty(nf)
        for j in range(nf):
            z[j] = (x[j] - mu_[j]) / sig_[j]
        acc = 0.0
        for i in range(z_fit.shape[0]):
            sq = 0.0
            for j in range(nf):
                d = z[j] - z_fit[i, j]
                sq += d * d
            acc += dual[i] * math.exp(-gamma * sq)
        return acc

    return pred


def make_compiled_mlp(net: BubbleFreqNet, mu, sig, y_mean: float, y_scale: float):
    """Compiled float64 forward of the eval-mode BubbleFreqNet (scalers folded)."""
    sd = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    w1, b1 = sd["input_proj.0.weight"], sd["input_proj.0.bias"]
    w1f = np.ascontiguousarray(w1 / sig[None, :])
    b1f = np.ascontiguousarray(b1 - w1 @ (mu / sig))
    g1, be1 = np.ascontiguousarray(sd["input_proj.1.weight"]), np.ascontiguousarray(sd["input_proj.1.bias"])
    w2, b2 = np.ascontiguousarray(sd["block1_linear.weight"]), np.ascontiguousarray(sd["block1_linear.bias"])
    g2, be2 = np.ascontiguousarray(sd["block1_norm.weight"]), np.ascontiguousarray(sd["block1_norm.bias"])
    w3, b3 = np.ascontiguousarray(sd["block2.0.weight"]), np.ascontiguousarray(sd["block2.0.bias"])
    g3, be3 = np.ascontiguousarray(sd["block2.1.weight"]), np.ascontiguousarray(sd["block2.1.bias"])
    wh = np.ascontiguousarray(sd["head.weight"][0] * y_scale)
    bh = float(sd["head.bias"][0] * y_scale + y_mean)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    @njit(fastmath=False)
    def _ln_gelu_inplace(h, g, be):
        n = h.shape[0]
        m = 0.0
        for o in range(n):
            m += h[o]
        m /= n
        v = 0.0
        for o in range(n):
            d = h[o] - m
            v += d * d
        v /= n
        inv = 1.0 / math.sqrt(v + 1e-5)
        for o in range(n):
            t = (h[o] - m) * inv * g[o] + be[o]
            h[o] = 0.5 * t * (1.0 + math.erf(t * inv_sqrt2))

    @njit(fastmath=False)
    def pred(x):
        h = np.empty(w1f.shape[0])
        for o in range(w1f.shape[0]):
            acc = b1f[o]
            for j in range(w1f.shape[1]):
                acc += w1f[o, j] * x[j]
            h[o] = acc
        _ln_gelu_inplace(h, g1, be1)
        h2 = np.empty(w2.shape[0])
        for o in range(w2.shape[0]):
            acc = b2[o]
            for j in range(w2.shape[1]):
                acc += w2[o, j] * h[j]
            h2[o] = acc
        _ln_gelu_inplace(h2, g2, be2)
        for o in range(h2.shape[0]):
            h2[o] += h[o]
        h3 = np.empty(w3.shape[0])
        for o in range(w3.shape[0]):
            acc = b3[o]
            for j in range(w3.shape[1]):
                acc += w3[o, j] * h2[j]
            h3[o] = acc
        _ln_gelu_inplace(h3, g3, be3)
        acc = bh
        for j in range(wh.shape[0]):
            acc += wh[j] * h3[j]
        return acc

    return pred


def make_compiled_clamp(pred, lo: float, hi: float):
    """Wrap a jitted single-query predictor with the Minnaert log-space clamp."""

    @njit(fastmath=False)
    def clamped(x):
        v = pred(x)
        if v < lo:
            return lo
        if v > hi:
            return hi
        return v

    return clamped


def make_compiled_bench(pred):
    """Driver loop: cycles through test rows so the work cannot be hoisted."""

    @njit(fastmath=False)
    def bench(x_all, n):
        s = 0.0
        m = x_all.shape[0]
        for i in range(n):
            s += pred(x_all[i % m])
        return s

    return bench


def time_compiled(pred, x_all: np.ndarray, n_iters: int) -> dict[str, float]:
    bench = make_compiled_bench(pred)
    bench(x_all, 100)  # warm-up / JIT compile
    t0 = time.perf_counter()
    bench(x_all, n_iters)
    elapsed = time.perf_counter() - t0
    return {"ns_per_query": elapsed / n_iters * 1e9, "n_iters": int(n_iters)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_10K)
    parser.add_argument("--metrics-json", type=Path, default=DEFAULT_METRICS_JSON)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Trained MLP output dir (split.json, scalers, checkpoint).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Defaults to the directory of --metrics-json.")
    parser.add_argument("--clamp-to-minnaert", action="store_true", help=CLAMP_FLAG_HELP)
    args = parser.parse_args()

    clamp_bounds = minnaert_log_clamp_bounds() if args.clamp_to_minnaert else None

    def _clamped(fn):
        """Add the clamp to a batch predictor so its cost is inside the timing."""
        if fn is None or clamp_bounds is None:
            return fn
        return lambda x, _f=fn: clamp_log_predictions(_f(x), clamp_bounds)

    def _clamped_compiled(pred):
        if clamp_bounds is None:
            return pred
        return make_compiled_clamp(pred, clamp_bounds[0], clamp_bounds[1])

    metrics_json = args.metrics_json.resolve()
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else metrics_json.parent

    with metrics_json.open("r", encoding="utf-8") as f:
        chosen = json.load(f)["models"]

    x_raw, y_raw, log_fs_raw, df_used = load_xy(args.dataset.resolve())
    split = split_dataset(
        x_raw, y_raw, log_fs_raw,
        val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    n_test = check_split_against_paper(df_used, split, model_dir / "split.json")
    print(f"split matches split.json (n_test={n_test})")

    x_scaler = StandardScaler().fit(split.x_train)
    mu = x_scaler.mean_.astype(np.float64)
    sig = x_scaler.scale_.astype(np.float64)
    x_train = x_scaler.transform(split.x_train)
    x_test_raw = np.asarray(split.x_test, dtype=np.float64)
    y_train = split.y_train.astype(np.float64)
    y_test = split.y_test.astype(np.float64)

    def _poly(degree: int):
        alpha = float(chosen[f"poly{degree}"]["hparams"]["ridge_alpha"])
        reg = LinearRegression() if alpha == 0.0 else Ridge(alpha=alpha)
        return make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), reg)

    models = {
        "linear": LinearRegression(),
        "poly2": _poly(2),
        "poly3": _poly(3),
        "rbf": KernelRidge(
            kernel="rbf",
            alpha=float(chosen["rbf"]["hparams"]["alpha"]),
            gamma=float(chosen["rbf"]["hparams"]["gamma"]),
        ),
    }
    raw_makers = {
        "linear": lambda m: make_raw_linear(m, mu, sig),
        "poly2": lambda m: make_raw_poly(m, mu, sig),
        "poly3": lambda m: make_raw_poly(m, mu, sig),
        "rbf": lambda m: make_raw_rbf(m, mu, sig),
    }
    compiled_makers = {
        "linear": lambda m: make_compiled_linear(m, mu, sig),
        "poly2": lambda m: make_compiled_poly(m, mu, sig),
        "poly3": lambda m: make_compiled_poly(m, mu, sig),
        "rbf": lambda m: make_compiled_rbf(m, mu, sig),
    }
    compiled_iters = {"linear": 5_000_000, "poly2": 2_000_000, "poly3": 1_000_000,
                      "rbf": 5_000, "mlp_cpu": 200_000}

    batches = {"single": x_test_raw[:1], "test_set": x_test_raw}
    timing: dict[str, dict] = {}

    def _time_both(name: str, framework_fn, raw_fn) -> None:
        for path, fn in (("framework", framework_fn), ("raw", raw_fn)):
            if fn is None:
                continue
            timing[name][path] = {}
            for bname, xb in batches.items():
                stats = time_call(lambda f=fn, x=xb: f(x), warmup=args.warmup, repeats=args.repeats)
                stats["us_per_bubble"] = stats["mean_ms"] * 1e3 / len(xb)
                stats["batch_size"] = len(xb)
                timing[name][path][bname] = stats
        raw_note = ""
        if raw_fn is not None:
            r = timing[name]["raw"]
            raw_note = (
                f" | raw: single {r['single']['mean_ms']*1e3:.1f} us, "
                f"batch {r['test_set']['mean_ms']:.3f} ms ({r['test_set']['us_per_bubble']:.3f} us/bubble)"
            )
        fw = timing[name]["framework"]
        print(
            f"[{name}] framework: single {fw['single']['mean_ms']*1e3:.1f} us, "
            f"batch({n_test}) {fw['test_set']['mean_ms']:.3f} ms "
            f"({fw['test_set']['us_per_bubble']:.3f} us/bubble){raw_note}"
        )

    for name, model in models.items():
        model.fit(x_train, y_train)
        framework_fn = lambda x, m=model: m.predict(x_scaler.transform(x))  # noqa: E731
        test_mape = metrics_from_log(framework_fn(x_test_raw), y_test)["mape"]
        recorded = float(chosen[name]["metrics"]["test"]["mape"])
        # 1e-3 pp: loose enough for BLAS non-determinism in the KRR solve,
        # tight enough to catch hyperparameter or dataset drift.
        if abs(test_mape - recorded) > 1e-3:
            raise SystemExit(
                f"{name}: refit test MAPE {test_mape:.6f}% != recorded {recorded:.6f}%"
            )
        raw_fn = raw_makers[name](model)
        if not np.allclose(raw_fn(x_test_raw), framework_fn(x_test_raw), rtol=1e-9, atol=1e-12):
            raise SystemExit(f"{name}: raw predictor does not reproduce sklearn predictions")
        timing[name] = {"hparams": chosen[name]["hparams"], "test_mape": test_mape}
        if clamp_bounds is not None:
            timing[name]["test_mape_clamped"] = metrics_from_log(
                clamp_log_predictions(framework_fn(x_test_raw), clamp_bounds), y_test
            )["mape"]
        # Cross-path agreement is checked UNCLAMPED above; the clamp is added only
        # to the timed callables so the recorded MAPE stays the reproducible one.
        _time_both(name, _clamped(framework_fn), _clamped(raw_fn))
        if HAS_NUMBA:
            cpred = compiled_makers[name](model)
            cpreds = np.array([cpred(x_test_raw[i]) for i in range(len(x_test_raw))])
            if not np.allclose(cpreds, framework_fn(x_test_raw), rtol=1e-9, atol=1e-12):
                raise SystemExit(f"{name}: compiled predictor does not reproduce sklearn predictions")
            timing[name]["compiled"] = time_compiled(
                _clamped_compiled(cpred), x_test_raw, compiled_iters[name]
            )
            print(f"    compiled: {timing[name]['compiled']['ns_per_query']:.1f} ns/query")

    # --- MLP (framework: torch; raw: numpy forward, scalers folded in) -----
    mlp_scaler = joblib.load(model_dir / "feature_scaler.joblib")
    y_scaler = joblib.load(model_dir / "target_log_scaler.joblib")
    ckpts = sorted(model_dir.glob("*_best.pt"))
    if not ckpts:
        raise SystemExit(f"no *_best.pt checkpoint in {model_dir}")
    net = BubbleFreqNet(input_dim=len(FEATURE_COLS), hidden_dim=64, hidden_dim2=32, dropout=0.1)
    net.load_state_dict(torch.load(ckpts[0], map_location="cpu"))
    net.eval()

    raw_mlp = make_raw_mlp(
        net,
        mlp_scaler.mean_.astype(np.float64),
        mlp_scaler.scale_.astype(np.float64),
        float(y_scaler.mean_[0]),
        float(y_scaler.scale_[0]),
    )

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for device in devices:
        net_d = net.to(device)
        name = f"mlp_{device}"
        timing[name] = {"checkpoint": ckpts[0].name}

        def _mlp_predict(x_np: np.ndarray) -> np.ndarray:
            xs = mlp_scaler.transform(x_np).astype(np.float32)
            with torch.no_grad():
                t = torch.from_numpy(xs).to(device)
                out = net_d(t)
                if device == "cuda":
                    torch.cuda.synchronize()
                out_np = out.cpu().numpy()
            return y_scaler.inverse_transform(out_np.reshape(-1, 1)).ravel()

        mape = metrics_from_log(_mlp_predict(x_test_raw), y_test)["mape"]
        timing[name]["test_mape"] = mape
        if clamp_bounds is not None:
            timing[name]["test_mape_clamped"] = metrics_from_log(
                clamp_log_predictions(_mlp_predict(x_test_raw), clamp_bounds), y_test
            )["mape"]
        raw_fn = None
        if device == "cpu":
            if not np.allclose(raw_mlp(x_test_raw), _mlp_predict(x_test_raw), rtol=1e-4, atol=1e-6):
                raise SystemExit("mlp: raw numpy forward does not reproduce the torch predictions")
            raw_fn = raw_mlp
        _time_both(name, _clamped(_mlp_predict), _clamped(raw_fn))
        if device == "cpu" and HAS_NUMBA:
            cpred = make_compiled_mlp(
                net,
                mlp_scaler.mean_.astype(np.float64),
                mlp_scaler.scale_.astype(np.float64),
                float(y_scaler.mean_[0]),
                float(y_scaler.scale_[0]),
            )
            cpreds = np.array([cpred(x_test_raw[i]) for i in range(len(x_test_raw))])
            # float64 compiled forward vs float32 torch: agreement to float32 precision.
            if not np.allclose(cpreds, _mlp_predict(x_test_raw), rtol=1e-4, atol=1e-5):
                raise SystemExit("mlp: compiled forward does not reproduce the torch predictions")
            timing[name]["compiled"] = time_compiled(
                _clamped_compiled(cpred), x_test_raw, compiled_iters["mlp_cpu"]
            )
            print(f"    compiled: {timing[name]['compiled']['ns_per_query']:.1f} ns/query")

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "metrics_json": str(metrics_json),
        "model_dir": str(model_dir),
        "n_train": int(len(x_train)),
        "n_test": int(n_test),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "protocol": {
            "framework": "StandardScaler.transform -> predict (sklearn / torch incl. "
                         "tensor conversion, device transfer, cuda synchronize, "
                         "inverse target scaling)",
            "raw": "pure-numpy math, feature+target scalers folded into coefficients; "
                   "verified to reproduce framework predictions",
            "compiled": "numba njit single-query function (same folded math, verified), "
                        "timed inside a compiled loop cycling through the test set; "
                        "includes a few ns of loop/row-indexing overhead, so tiny "
                        "models (linear) read as an upper bound"
                        + ("" if HAS_NUMBA else " -- SKIPPED, numba not installed"),
        },
        "minnaert_clamp": (
            {
                "factor": MINNAERT_CLAMP_FACTOR,
                "log_lo": clamp_bounds[0],
                "log_hi": clamp_bounds[1],
                "note": "included in every timed path",
            }
            if clamp_bounds is not None
            else None
        ),
        "timing": timing,
    }
    json_path = out_dir / "inference_timing.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    display = {
        "linear": "Linear regression",
        "poly2": "Polynomial (deg 2) + ridge",
        "poly3": "Polynomial (deg 3) + ridge",
        "rbf": "RBF kernel ridge",
        "mlp_cpu": "MLP (CPU)",
        "mlp_cuda": "MLP (GPU, incl. transfer)",
    }
    lines = [
        "# Inference time of the 8-feature regressors",
        "",
        f"Warm-up {args.warmup}, mean over {args.repeats} repeats; single-query "
        f"latency and full test-set batch (n={n_test}). **Compiled** = numba-jitted "
        "folded math timed inside a compiled loop over varying inputs -- the pure "
        "inference cost a C++ caller would see (linear is an upper bound: it is "
        "comparable to the loop's ~2 ns row-indexing overhead). **Raw** = the same "
        "math in numpy, which still pays ~1-2 us dispatch per operation at batch 1. "
        "**Framework** = sklearn/torch call path, dominated at batch 1 by fixed "
        "per-call overhead. All three paths are verified to produce identical "
        "predictions before being timed.",
        "",
        f"| Model | Test MAPE (%) | Compiled single (ns) | Raw single (us) | Raw us/bubble (n={n_test}) | Framework single (us) | Framework us/bubble |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, label in display.items():
        if key not in timing:
            continue
        t = timing[key]
        fw = t["framework"]
        compiled_cell = (
            f"{t['compiled']['ns_per_query']:.1f}" if "compiled" in t else "--"
        )
        if "raw" in t:
            raw = t["raw"]
            raw_cells = (
                f"{raw['single']['mean_ms']*1e3:.2f} | "
                f"{raw['test_set']['us_per_bubble']:.3f}"
            )
        else:
            raw_cells = "-- | --"
        lines.append(
            f"| {label} | {t['test_mape']:.3f} | {compiled_cell} | {raw_cells} | "
            f"{fw['single']['mean_ms']*1e3:.1f} | "
            f"{fw['test_set']['us_per_bubble']:.3f} |"
        )
    md_path = out_dir / "inference_timing.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {json_path}\n       {md_path}")


if __name__ == "__main__":
    main()
