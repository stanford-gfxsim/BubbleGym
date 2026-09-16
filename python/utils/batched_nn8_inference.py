"""Batched GPU inference per-bubble time for Table 1 (Fruits n=6511, Exhale n=32464).

Matches the paper protocol: one forward over the whole scene's bubble batch on
GPU, including host->device transfer and device->host readback, with
cuda.synchronize; per-sample time = batch wall time / n.
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import time
import sys
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "python" / "freq_model"))
import joblib
from freq_model.NN.bub_freq_net import BubbleFreqNet

ART = REPO / "python" / "freq_model" / "output" / "output_8feature_direct_bubblegym_10k"
scaler = joblib.load(ART / "feature_scaler.joblib")
y_scaler = joblib.load(ART / "target_log_scaler.joblib")
net = BubbleFreqNet(input_dim=8, hidden_dim=64, hidden_dim2=32, dropout=0.1)
net.load_state_dict(torch.load(next(ART.glob("*_best.pt")), map_location="cpu"))
net.eval()

# Realistic feature rows: sample from the 10k dataset's LBM rows.
import pandas as pd
df = pd.read_csv(REPO / "dataset" / "bubble_gym" / "dataset_bubblegym_10k.csv")
from shape_feature.nonspherical_features import compute_nonsphericity_columns
df = compute_nonsphericity_columns(df)
df["i11_over_i00"] = df["i11"] / df["i00"]
df["i22_over_i00"] = df["i22"] / df["i00"]
cols = ["i11_over_i00", "i22_over_i00", "non_sph_va", "non_sph_vm", "non_sph_w", "eta_V", "eta_A", "eta_M"]
base = df[cols].to_numpy(dtype=np.float64)

results = {}
for name, n, device in [("fruits_gpu", 6511, "cuda"), ("exhale_gpu", 32464, "cuda"),
                        ("fruits_cpu", 6511, "cpu"), ("exhale_cpu", 32464, "cpu")]:
    if device == "cuda" and not torch.cuda.is_available():
        continue
    rows = base[np.random.default_rng(0).integers(0, len(base), n)]
    xs = scaler.transform(rows).astype(np.float32)
    net_d = net.to(device)
    def run():
        with torch.no_grad():
            t = torch.from_numpy(xs).to(device)
            out = net_d(t)
            if device == "cuda":
                torch.cuda.synchronize()
            o = out.cpu().numpy()
        return y_scaler.inverse_transform(o.reshape(-1, 1))
    for _ in range(5):
        run()
    reps = 50
    t0 = time.perf_counter()
    for _ in range(reps):
        run()
    dt = (time.perf_counter() - t0) / reps
    results[name] = {"n": n, "batch_ms": dt * 1e3, "ms_per_bubble": dt * 1e3 / n}
    print(f"{name}: n={n} batch={dt*1e3:.3f} ms -> {dt*1e3/n*1e3:.3f} us/bubble ({dt*1e3/n:.6f} ms/bubble)")

out = (
    REPO / "results" / "experiments" / "table01_timing_across_scenes"
    / "batched_gpu_inference.json"
)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                           "protocol": "transfer + forward + readback, synchronize, mean of 50 after 5 warm-up",
                           "results": results}, indent=2))
print("wrote", out)
