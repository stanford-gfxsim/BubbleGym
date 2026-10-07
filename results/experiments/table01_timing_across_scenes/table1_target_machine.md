# Table 1 — target machine

Measured 2026-07-03/04 on **Intel i9-14900K (24 cores / 32 threads), NVIDIA GeForce RTX 5090**,
Windows 11 — the hardware named in Sec. 6 of the paper. Model: the released 8-feature MLP,
`python/freq_model/output/output_8feature_direct_bubblegym_10k`, trained on
`dataset/bubble_gym/dataset_bubblegym_10k.csv` (6,000 Langlois2016 + 4,000 LBM; test MAPE 0.089%).
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
  Enright Test frame 1 is excluded (non-finite feature row).
- The Fruits and Exhalation rows are read from
  `all_bubbles/fruit08/fruit08_all_summary.json` and
  `all_bubbles/exhale15/exhale15_all_summary.json` (both i9-14900K). `#Bubbles` is
  unique bubbles with an exported mesh / BEM solves; the two populations differ, so
  one column does not divide into the other.
- † Their inference cell is the one constant every Table 1 row carries — the
  batch-512 whole-path figure (`h64_h32`, `fullpath_us_per_bubble_mean` = 0.5155 µs)
  from `results/experiments/supp_table02_width_ablation/width_timing.json`. The speedup is
  mean BEM total / mean (feat + that constant):
  0.8124/0.005733 = 142× and 0.7417/0.009033 = 82×.
  `batched_gpu_inference.json` in this directory times a *single* batch over a whole
  scene and reads an order of magnitude lower; it is **not** the Table 1 constant.

## Per-scene mesh statistics + BEM vs MLP timing (procedural scenes)

Mesh columns are avg / median / min / max over the timed meshes of each scene;
timing is the per-frame average over all frames.

| Scene | n timed | #Vertices | #Faces | BEM asm/s | BEM solve/s | BEM total/s | Ours feat/ms | Ours infer(GPU)/ms | Speedup |
|---|---|---|---|---|---|---|---|---|---|
| Ellipsoid | 240 | 492 / 492 / 492 / 492 | 980 / 980 / 980 / 980 | 0.782 | 0.012 | 0.794 | 7.4 | 0.30 | 103× |
| Curl Noise | 241 | 496 / 492 / 492 / 1,529 | 989 / 980 / 980 / 3,054 | 0.866 | 0.016 | 0.883 | 7.3 | 0.30 | 117× |
| Enright Test | 119 | 2,997 / 2,913 / 638 / 5,291 | 5,990 / 5,822 / 1,272 / 10,578 | 26.150 | 1.349 | 27.498 | 50.8 | 0.35 | 538× |

Regressor accuracy and timing (Table 2) are in
[`../table02_regressor_comparison/`](../table02_regressor_comparison).

## Notes

- Meshes with degenerate geometry fail feature extraction (`bad principal inertia` /
  `non-finite chull row`) and are skipped + logged (`chunks/*.fail`). This matches the
  production NN pipeline (`python/scene/_common/write_trackedbubinfo_nn.py` marks such
  frames invalid and excludes them from NN inference).

## Raw artifacts (this directory)

- `nn8_vs_bem_ellipsoid.csv`, `nn8_vs_bem_curl_noise.csv`, `nn8_vs_bem_enright_test.csv`
  (per-frame rows + AVERAGE + hardware metadata); `enright_summary.json`.
- `batched_gpu_inference.json` — batched Fruits/Exhale inference (GPU + CPU).
- `all_bubbles/fruit08/` and `all_bubbles/exhale15/` — the two sweeps, run with
  `python/utils/nn8_vs_bem_all_bubbles_chunked.py`. Only `<scene>_all_summary.json`
  (plus fruit08's per-bubble BEM timing CSV) ships; the merged per-row CSVs and vertex
  scans are hundreds of MB and are not included.
