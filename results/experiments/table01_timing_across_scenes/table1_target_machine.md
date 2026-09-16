# Table 1 re-collection — target machine (fresh measurements)

Measured 2026-07-03/04 on **Intel i9-14900K (24 cores / 32 threads), NVIDIA GeForce RTX 5090**,
Windows 11 — the hardware named in Sec. 6 of the paper. Model: 8-feat MLP retrained on
`dataset/bubble_gym/dataset_bubblegym_10k.csv` (6,000 Tim2016 + 4,000 LBM; test MAPE 0.07%),
artifacts `python/freq_model/output/output_8feature_direct_bubblegym_10k`.
BEM: P1-DP0 Galerkin, mass precond, GMRES tol=1e-12, quadrature 6/6 (dataset profile), CPU,
sequential single-core. bempp JIT warm-up excluded everywhere.

## Table 1 (assemble + solve split)

| Scene | #Bubbles | t_BEM assemble/s | t_BEM solve/s | t_Ours feat/ms | t_Ours infer/ms | Speedup |
|---|---|---|---|---|---|---|
| Ellipsoid | 1 | 0.782 | 0.012 | 7.4 | 0.30 | 103× |
| Curl Noise | 1 | 0.866 | 0.016 | 7.3 | 0.30 | 117× |
| Enright Test | 1 | 26.150 | 1.349 | 50.8 | 0.35 | 538× |
| Fruits (fruit08) | 7,053 / 75,686 solves | 0.789 (med 0.450) | 0.023 (med 0.020) | 5.73 (med 3.74) | 0.0005 † | 142× |
| Exhalation (exhale15) | 25,668 / 226,562 solves | 0.713 (med 0.319) | 0.029 (med 0.020) | 9.03 (med 5.79) | 0.0005 † | 82× |

- Procedural rows: per-frame averages over every frame of
  `dataset/bubble_theater/{ellipsoid (240), curl_noise (241), enright_test (119)}`;
  AVERAGE rows of `nn8_vs_bem_<scene>.csv` in this directory. Inference = GPU forward incl.
  transfer, batch 1. Speedup = mean BEM total / mean (feat + GPU infer).
  Enright Test frame 1 is excluded (non-finite convex-hull feature row; the same frame was
  excluded in the previous collection, 119 frames both times).
- The Fruits and Exhalation rows above are now the **final** sweeps the paper prints,
  read straight from `all_bubbles/fruit08/fruit08_all_summary.json` and
  `all_bubbles/exhale15/exhale15_all_summary.json` (both i9-14900K). `#Bubbles` is
  unique bubbles with an exported mesh / BEM solves; the two populations differ, so
  one column does not divide into the other.
- † Their inference cell is the one constant every Table 1 row carries — the
  batch-512 whole-path figure (`h64_h32`, `fullpath_us_per_bubble_mean` = 0.5155 µs)
  from `results/experiments/supp_table02_width_ablation/width_timing.json`. The speedup is
  mean BEM total / mean (feat + that constant):
  0.8124/0.005733 = 142× and 0.7417/0.009033 = 82×.
  `batched_gpu_inference.json` in this directory times a *single* batch over a whole
  scene (n = 6,511 / 32,464) and reads an order of magnitude lower; it is **not** the
  Table 1 constant.

## Per-scene mesh statistics + BEM vs MLP timing (combined)

Mesh columns are avg / median / min / max over the timed meshes of each scene;
timing is per mesh (procedural: per-frame average over all frames; Fruits:
interim mean over the all-bubbles rows so far, medians in parentheses —
BEM cost is heavy-tailed).

| Scene | n timed | #Vertices | #Faces | BEM asm/s | BEM solve/s | BEM total/s | Ours feat/ms | Ours infer(GPU)/ms | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| Ellipsoid | 240 | 492 / 492 / 492 / 492 | 980 / 980 / 980 / 980 | 0.782 | 0.012 | 0.794 | 7.4 | 0.30 | 103× |
| Curl Noise | 241 | 496 / 492 / 492 / 1,529 | 989 / 980 / 980 / 3,054 | 0.866 | 0.016 | 0.883 | 7.3 | 0.30 | 117× |
| Enright Test | 119 | 2,997 / 2,913 / 638 / 5,291 | 5,990 / 5,822 / 1,272 / 10,578 | 26.150 | 1.349 | 27.498 | 50.8 | 0.35 | 538× |
| ~~Fruits~~ (superseded) | 33,571 BEM (+210 NN-only, 871 skips) | 375 / 190 / 101 / 37,854 | 746 / 376 / 178 / 74,110 | 1.313 (0.454) | 0.040 (0.016) | 1.352 (0.470) | 7.7 (4.1) | 0.31 | 169× |
| ~~Exhale~~ (superseded) | 94,508 BEM (+1,565 NN-only, 2,454 skips) | 365 / 162 / 101 / 28,572 | 726 / 320 / 174 / 57,396 | 0.723 (0.393) | 0.025 (0.021) | 0.748 (0.413) | 9.3 (5.1) | 0.45 | 77× |

**The last two rows are superseded and are not the paper's.** They are the
earlier `fruits` / `exhale` sweeps (6,511 and 32,464 bubbles, 169× and 77×),
kept only as mesh-size context; their summary JSONs are no longer shipped, so
the timing cells cannot be re-derived here. The paper's Fruits and Exhalation
numbers are the `fruit08` / `exhale15` rows of Table 1 above.

## Regressors on the 10k dataset: accuracy + inference timing

Same 8 features, split (seed 42, 70/15/15) and MAPE protocol for every row;
timing re-measured with
`python/freq_model/regression/measure_baseline_inference_time.py`
(warm-up 20, mean of 200 reps; **compiled** = numba-jitted folded math ≈ what a
C++ caller pays; **framework** = sklearn/torch call path incl. per-call
overhead). The ledger these cells were transcribed from is not shipped.

| Model | Test MAPE (%) | Max APE (%) | Compiled single (ns) | Raw single (µs) | Framework single (µs) | Framework µs/bubble (batch 1499) |
|---|---|---|---|---|---|---|
| Linear regression | 0.220 | 3.51 | 1.4 | 1.29 | 51.9 | 0.055 |
| Polynomial (deg 2) + ridge | 0.062 | 1.60 | 57.9 | 11.54 | 148.9 | 0.283 |
| Polynomial (deg 3) + ridge | 0.052 | 2.18 | 144.5 | 9.14 | 179.3 | 1.184 |
| RBF kernel ridge | 0.055 | 2.01 | 35,182 | 50.55 | 372.3 | 66.875 |
| MLP (paper, CPU) | 0.074 | 3.67 | 3,338 | 67.27 | 360.8 | 0.443 |
| MLP (paper, GPU incl. transfer) | 0.074 | 3.67 | — | — | 500.7 | 0.383 |

- MLP per-source test MAPE: Tim2016 0.053%, LBM-3k 0.124%, LBM-1k 0.048%
  (`metrics.json`). Whole-scene batched GPU inference: Fruits (n=6,511)
  0.069 µs/bubble, Exhale (n=32,464) 0.015 µs/bubble (`batched_gpu_inference.json`).
- RBF pays per-query kernel evaluation against all 7,001 training samples —
  accurate but 2–4 orders of magnitude slower than the parametric models.
- An all-bubbles regressor post-pass (`apply_regressors_all_bubbles.py`) was run
  over the superseded `fruits` and `exhale` sweeps. Its summary JSONs are no
  longer shipped, so those numbers are not reproducible here. The qualitative
  result it established still holds and is what the paper argues: the degree-2
  ridge matches the MLP in-distribution but **extrapolates catastrophically**
  off it, producing exp-overflow frequencies in log-f space (up to literal
  Infinity), while the bounded MLP degrades gracefully.

## Notes

- Meshes with degenerate geometry fail feature extraction (`bad principal inertia` /
  `non-finite chull row`) and are skipped + logged (`chunks/*.fail`). This matches the
  production NN pipeline (`python/scene/_common/write_trackedbubinfo_nn.py` marks such
  frames invalid and excludes them from NN inference).

## Raw artifacts (this directory)

- `nn8_vs_bem_ellipsoid.csv`, `nn8_vs_bem_curl_noise.csv`, `nn8_vs_bem_enright_test.csv`
  (per-frame rows + AVERAGE + hardware metadata); `enright_summary.json`.
- `batched_gpu_inference.json` — batched Fruits/Exhale inference (GPU + CPU).
- `all_bubbles/fruit08/` and `all_bubbles/exhale15/` — the two final sweeps, run with
  `python/utils/nn8_vs_bem_all_bubbles_chunked.py`. Only `<scene>_all_summary.json`
  (plus fruit08's per-bubble BEM timing CSV) ships; the merged per-row CSVs and vertex
  scans are hundreds of MB and were left out.
