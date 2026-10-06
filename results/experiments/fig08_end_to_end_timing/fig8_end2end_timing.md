# Fig. 8 end-to-end pipeline timing

Figure: `fruit08_exhale08_end2end_timing_pies.png`, drawn by
`python/utils/plot_end2end_timing_pies.py`, which reads
`fruit08_end2end_timings.json` and `exhale15_end2end_timings.json`.

Two scenes, a BEM pie and a surrogate pie each: Fruit Splash (`fruit08`) and
Underwater Exhalation (`exhale15`).

## Hardware

Every number below was taken on **one workstation**, the one the paper names:
Intel Core i9-14900K (24 cores / 32 threads), NVIDIA GeForce RTX 5090, 96 GB
RAM. Two slices of the exhalation pie are carried from an earlier ledger rather
than re-measured there; they are called out in that scene's section.

## Fruit Splash (fruit08)

Scene: five-fruit drop, 240x240x320 grid, dx = 1/480 m, dt = 1/11520 s, 4.0 s,
46,080 LBM iterations, 240 exported frames (60 fps). The shipped tracked files
span t = 0.0792 .. 4.0166 s over 54,116 Bub blocks and 6,196,722 sample lines.

### Stage costs

| stage | seconds | source |
|---|---|---|
| LBM simulation | 2,890.23 | rerun `log.txt`, `[stage timing final] sim=` (62.72 ms/iter) |
| bubble tracking | 2,082.07 | same line, `track=` (45.18 ms/iter) |
| phi/tag dump I/O | 532.34 | same line, `dump_io=` (11.55 ms/iter) |
| per-bubble mesh extraction | 99.70 | `[stage timing final] per_bubble_mc_export=` |
| frequency prediction BEM | 61,488.0 (17.08 h) | sweep `table01_timing_across_scenes/all_bubbles/fruit08/fruit08_all_summary.json`, `total_bem_hours`; 75,686 solves, mean 0.812 s (assemble 0.789 / solve 0.023), median 0.471 s, max 81.2 s |
| frequency prediction NN8 (ours) | 52.49 | `fruit08_nn_step_timing.json`; see "Surrogate frequency step" below |
| sound synthesis | 14.30 | wall time of `runFluidSound.exe` rendering the shipped `dataset/fruit_splash/trackedBubInfo_NN.txt` (389.7 MB of raw text, file cache warm; the repository now ships it as `.txt.xz`, so reproducing this needs `xz -dk` first) to the full 4.0 s at 48 kHz, scheme 0 (uncoupled), damping 0.8; median of three runs (14.47 / 14.30 / 14.17 s), byte-identical output. FluidSound is a separate project not driven from this repo |

The four simulation stages sum to 5,604.3 s against the run's own 5,611.1 s
wall clock, so 99.9 % of the run is accounted for.

BEM settings: P1-DP0, mass preconditioner, GMRES tol 1e-12, quadrature 6/6
(NUMBA_NUM_THREADS=24). Sweep: 77 chunks of <= 1000 meshes, 75,885 rows =
75,686 BEM + 199 NN-only, 780 failures (1.0 %); 25.5 h wall including a ~1 h
restart loss (resumed from chunk sentinels).

### Surrogate frequency step

Measured directly on the workstation, replacing an earlier estimate that scaled
a single-threaded ledger by a process-pool speedup taken on another machine:

```
python python/utils/time_scene_nn_step.py --label fruit08 --workers 16 \
    --rows <fruit08_all_rows.csv> --json-out fruit08_nn_step_timing.json
```

It runs the production surrogate path over exactly the meshes the BEM sweep
solved, so the two frequency slices cover identical work: mesh load and the
eight shape features in a pool of 16 worker processes, each pinned to one BLAS
thread, then one batched model call.

| | |
|---|---|
| meshes | 75,686, the pre-smoothed per-bubble OBJs the sweep read |
| load + features | 51.22 s |
| batched inference | 1.27 s |
| **total** | **52.49 s** (0.69 ms per mesh) |

- The OS file cache is warmed with one untimed read of every mesh first.
- 80 meshes (0.1 %) fail feature extraction as non-watertight; the time includes
  the attempt.
- Every one of the other 75,606 predictions matches the sweep's recorded NN
  frequency to 1.2e-7 relative, float32 rounding, so the timed path is the
  production one.

## Underwater exhalation (exhale15)

Scene: the 12-second run, the larger of the two by every measure. The shipped
tracked files span t = 0.0167 .. 12.0166 s over 216,578 Bub blocks and
20,573,045 sample lines: 3.0x the duration of fruit08, 4.0x the bubbles and
3.3x the sample lines. Of those bubbles, 25,668 have an exported per-bubble
mesh, which is what the frequency stages cost.

### Stage costs

| stage | seconds | source |
|---|---|---|
| LBM simulation | 12,767.1 | exhalation run `log.txt`, `[stage timing final] sim=` |
| bubble tracking | 5,773.84 | same line, `track=` |
| phi/tag dump I/O | 1,862.8 | same line, `dump_io=` |
| per-bubble mesh extraction | 439.63 | `[stage timing final] per_bubble_mc_export=` |
| frequency prediction BEM | 168,120.0 (46.7 h) | sweep `table01_timing_across_scenes/all_bubbles/exhale15/exhale15_all_summary.json`, `total_bem_hours` = 46.675 h, entered as 46.7 h; 226,562 solves, mean 0.742 s (assemble 0.713 / solve 0.029), median 0.340 s, max 87.7 s |
| frequency prediction NN8 (ours) | 246.0 | carried from the earlier ledger, not re-measured; cross-checked below |
| sound synthesis | 72.0 | carried from the earlier ledger, not re-measured; FluidSound is a separate project not driven from this repo |

The four simulation stages sum to 20,843.4 s.

BEM settings as for fruit08: P1-DP0, mass preconditioner, GMRES tol 1e-12,
quadrature 6/6. Sweep: 228,413 rows = 226,562 BEM + 1,851 NN-only, 2,513
failures (1.1 %).

### The two slices that are not a fresh measurement

The four simulation stages are the exact `[stage timing final]` values of this
scene's own run log on the workstation above, and its BEM sweep ran there too.
The surrogate frequency and audio slices were not re-measured the way fruit08's
were, because both need inputs the release does not carry: the per-bubble mesh
tree (simulator output) and the FluidSound renderer. They are the earlier
ledger's values, and both sit where an estimate from fruit08's measured rates
would put them:

- **Frequency, 246.0 s.** fruit08's measured rate is 0.6935 ms per mesh. This
  scene's meshes cost more per mesh in feature extraction, by the ratio of the
  two sweeps' `mean_nn_feat_s` (9.033 ms against 5.733 ms, both single
  threaded), which puts it at 1.09 ms per mesh, or 247.6 s over its 226,562
  solved meshes. That is within 1 % of the ledger's 246.0 s.
- **Audio, 72.0 s.** fruit08 renders 4.0 s of sound from 6.2 M sample lines in
  14.30 s. This scene is 3.0x as long with 3.3x the sample lines and 4.0x the
  bubbles, and 72.0 s is 5.0x fruit08's figure, so it is the right order but
  not pinned to any of those ratios.

Neither number feeds anything else: every other slice of either pie rebuilds
from data in this repository.

## Printed by the plot script

```
===== Fruit Splash =====

BEM reference: total 67,106.6 s = 18.6 h
  LBM simulation            2,890.2 s   48.2 min    4.31%
  Bubble tracking           2,082.1 s   34.7 min    3.10%
  VOF field/bubble tag dump I/O      532.3 s    8.9 min    0.79%
  Mesh extraction              99.7 s    1.7 min    0.15%
  Frequency estimation     61,488.0 s     17.1 h   91.63%
  Audio synthesis              14.3 s     14.3 s    0.02%

Our surrogate: total 5,671.1 s = 1.6 h
  LBM simulation            2,890.2 s   48.2 min   50.96%
  Bubble tracking           2,082.1 s   34.7 min   36.71%
  VOF field/bubble tag dump I/O      532.3 s    8.9 min    9.39%
  Mesh extraction              99.7 s    1.7 min    1.76%
  Frequency estimation         52.5 s     52.5 s    0.93%
  Audio synthesis              14.3 s     14.3 s    0.25%

end-to-end speedup: 11.8x
bottleneck with the surrogate: LBM simulation (51.0%)

===== Exhalation =====

BEM reference: total 189,035.4 s = 52.5 h
  LBM simulation           12,767.1 s      3.5 h    6.75%
  Bubble tracking           5,773.8 s      1.6 h    3.05%
  VOF field/bubble tag dump I/O    1,862.8 s   31.0 min    0.99%
  Mesh extraction             439.6 s    7.3 min    0.23%
  Frequency estimation    168,120.0 s     46.7 h   88.94%
  Audio synthesis              72.0 s    1.2 min    0.04%

Our surrogate: total 21,161.4 s = 5.9 h
  LBM simulation           12,767.1 s      3.5 h   60.33%
  Bubble tracking           5,773.8 s      1.6 h   27.28%
  VOF field/bubble tag dump I/O    1,862.8 s   31.0 min    8.80%
  Mesh extraction             439.6 s    7.3 min    2.08%
  Frequency estimation        246.0 s    4.1 min    1.16%
  Audio synthesis              72.0 s    1.2 min    0.34%

end-to-end speedup: 8.9x
bottleneck with the surrogate: LBM simulation (60.3%)
```


