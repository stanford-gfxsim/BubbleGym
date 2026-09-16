# fruit08 end-to-end pipeline timing (run `selected-dataMR3D_fruit_hires_fullrate_v9`)

Measured 2026-08-20/21 on one run instance. Figure:
`fruit08_exhale08_end2end_timing_pies.png` (drawn by
`python/utils/plot_end2end_timing_pies.py`, which reads
`fruit08_end2end_timings.json`).

Scene: five-fruit drop, 240x240x320 grid, dx = 1/480 m, dt = 1/11520 s, 4.0 s,
46,080 LBM iterations, 240 exported frames (60 fps), 125,262 per-bubble meshes
written (36,692 unique bubbles raw; 76,665 meshes above the 100-vertex NN/BEM floor).

Hardware: AMD Ryzen Threadripper 9970X (32 cores / 64 threads), NVIDIA GeForce RTX 5090
for the frame-driven stages (simulation, tracking, dump, mesh extraction, audio);
the frequency stages were re-swept on the i9-14900K named in the paper, so the pie
is not a single-machine wall clock.

| stage | seconds | source |
|---|---|---|
| LBM simulation | 2,994.41 | run `log.txt` `[stage timing final] sim=` (64.98 ms/iter) |
| bubble tracking | 2,010.01 | same line `track=` (43.62 ms/iter) |
| phi/tag dump I/O | 494.88 | same line `dump_io=` (10.74 ms/iter) |
| per-bubble mesh extraction | 159.73 | `[stage timing final] per_bubble_mc_export=` |
| frequency prediction — BEM | 61,488.0 (17.08 h) | sweep `table01_timing_across_scenes/all_bubbles/fruit08/fruit08_all_summary.json` `total_bem_hours`; 75,686 solves on the i9-14900K, mean 0.812 s (assemble 0.789 / solve 0.023), median 0.471 s, max 81.2 s |
| frequency prediction — NN8 (ours) | 796.5 | sweep ledger, same 75,686 meshes: mesh load 122.1 + shape features 648.8 + GPU inference 25.6 s (8.55 ms features, 0.34 ms inference per mesh) |
| sound synthesis | 48.5 | wall time of the FluidSound renderer on the delivered NN trackedBubInfo (scheme 0, 48 kHz, damping 0.8; 44,364 event times; includes reading the 415 MB file). FluidSound is a separate project and is not driven from this repo, so this stage cannot be re-measured here |

BEM settings: P1-DP0, mass preconditioner, GMRES tol 1e-12, quadrature 6/6
(NUMBA_NUM_THREADS=24). Sweep: 77 chunks of <= 1000 meshes, 75,885 rows =
75,686 BEM + 199 NN-only, 780 failures (1.0 %); 25.5 h wall including a ~1 h
restart loss (resumed from chunk sentinels).

## Printed by the plot script

```
BEM reference: total 67,195.5 s = 18.7 h
  LBM simulation            2,994.4 s   49.9 min    4.46%
  Bubble tracking           2,010.0 s   33.5 min    2.99%
  phi/tag dump I/O            494.9 s    8.2 min    0.74%
  Mesh extraction             159.7 s    2.7 min    0.24%
  Frequency estimation     61,488.0 s     17.1 h   91.51%
  Audio synthesis              48.5 s     48.5 s    0.07%

Our surrogate: total 5,803.9 s = 1.6 h
  LBM simulation            2,994.4 s   49.9 min   51.59%
  Bubble tracking           2,010.0 s   33.5 min   34.63%
  phi/tag dump I/O            494.9 s    8.2 min    8.53%
  Mesh extraction             159.7 s    2.7 min    2.75%
  Frequency estimation         96.4 s    1.6 min    1.66%
  Audio synthesis              48.5 s     48.5 s    0.84%

end-to-end speedup: 11.6x
bottleneck with the surrogate: LBM simulation (51.6%)
```

## Caption / prose numbers (restate from the block above, never hand-edit)

With BEM the frequency stage is 91.5 % of an 18.7 h pipeline; the learned
surrogate brings the whole pipeline to 1.6 h (11.6x end-to-end), after which
the LBM simulation itself (51.6 %) and bubble tracking (34.6 %) dominate and
frequency prediction is 1.7 %. These are the numbers in the paper's Fig. 5
caption.

The surrogate seconds start from an 8.55 ms/mesh feature timing rather than
fruit08's 5.73 ms, so 96.4 s is the conservative figure.

## Parallel feature extraction

The ledger's 796.5 s is single-threaded. Feature extraction is embarrassingly
parallel, so it was re-timed through a process pool on 2,000 meshes drawn
uniformly from this run's 76,665 candidates (>100 vertices), on the
Threadripper 9970X:

| workers | mesh/s | speedup | efficiency |
|---|---|---|---|
| 1 | 57.8 | 1.00x | -- |
| 4 | 221.7 | 3.84x | 96 % |
| 8 | 385.4 | 6.67x | 83 % |
| 16 | 629.4 | 10.89x | 68 % |

The OS file cache is warmed before the serial pass; without that the serial run
pays every cold read and the pool passes reuse its cache, which inflates the
speedup.

Applying 10.89x to mesh load + features and leaving GPU inference alone (it was
not in the pool): (122.1 + 648.8)/10.89 + 25.6 = **96.4 s**, which is the
frequency-estimation figure the surrogate column of the printed block uses
throughout.
