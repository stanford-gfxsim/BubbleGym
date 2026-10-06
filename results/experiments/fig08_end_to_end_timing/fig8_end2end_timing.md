# Fig. 8 end-to-end pipeline timing

Figure: `end2end_timing_pies.png`, drawn by
`python/utils/plot_end2end_timing_pies.py`, which reads
`fruit08_end2end_timings.json` and `exhale15_end2end_timings.json`.

Two scenes, a BEM pie and a surrogate pie each: Fruit Splash (`fruit08`) and
Underwater Exhalation (`exhale15`).

## Hardware

Every number below was taken on **one workstation**, the one the paper names:
Intel Core i9-14900K (24 cores / 32 threads), NVIDIA GeForce RTX 5090, 96 GB
RAM.

## Fruit Splash (fruit08)

Scene: five-fruit drop, 240x240x320 grid, dx = 1/480 m, dt = 1/11520 s, 4.0 s,
Stage costs
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
| frequency prediction NN8 (ours) | 58.435 | `fruit08_nn_step_timing.json`; mean of five runs, see "Surrogate frequency step" below |
| sound synthesis | 14.30 | FluidSound is a separate project not driven from this repo |

The four simulation stages sum to 5,604.3 s against the run's own 5,611.1 s
wall clock, so 99.9 % of the run is accounted for.

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
thread, then the model over the whole scene. Inference follows Table 1's
protocol -- batches of 512 on the GPU, host to device, forward, back, with
`cuda.synchronize()` -- so the figure and the table price the model the
same way.

| | |
|---|---|
| meshes | 75,686, the pre-smoothed per-bubble OBJs the sweep read |
| load + features | 57.99 s |
| batched inference (512, GPU) | 0.445 s |
| **total** | **58.435 s** (0.77 ms per mesh) |

- Mean of five runs: 58.133, 58.427, 58.535, 58.835, 58.243 s (sd 0.273 s).
- The OS file cache is warmed with one untimed read of every mesh first.
- 80 meshes (0.1 %) fail feature extraction as non-watertight; the time includes
  the attempt.
- Every one of the other 75,606 predictions matches the sweep's recorded NN
  frequency to 1.2e-7 relative, float32 rounding, so the timed path is the
  production one.
- Inference is 5.9 us per mesh here against the 0.59 us per bubble Table 1
  quotes: the table times one pre-made batch through the kernel, a whole-scene
  pass also pays the per-batch dispatch, the copies of real data, feature
  scaling and the target inverse transform.

## Underwater exhalation (exhale15)

Scene: the 12-second run, the larger of the two by every measure. The shipped
tracked files span t = 0.0167 .. 12.0166 s over 216,578 Bub blocks and
20,573,045 sample lines: 3.0x the duration of fruit08, 4.0x the bubbles and
3.3x the sample lines. 25,668 of those bubbles have an exported mesh tree, and
the frequency stages are priced over the 226,562 mesh frames the sweep solved
across them, 8.8 frames per bubble against fruit08's 10.7.

### Stage costs

| stage | seconds | source |
|---|---|---|
| LBM simulation | 12,767.1 | exhalation run `log.txt`, `[stage timing final] sim=` |
| bubble tracking | 5,773.84 | same line, `track=` |
| phi/tag dump I/O | 1,862.8 | same line, `dump_io=` |
| per-bubble mesh extraction | 439.63 | `[stage timing final] per_bubble_mc_export=` |
| frequency prediction BEM | 168,120.0 (46.7 h) | sweep `table01_timing_across_scenes/all_bubbles/exhale15/exhale15_all_summary.json`, `total_bem_hours` = 46.675 h, entered as 46.7 h; 226,562 solves, mean 0.742 s (assemble 0.713 / solve 0.029), median 0.340 s, max 87.7 s |
| frequency prediction NN8 (ours) | 153.704 | `exhale15_nn_step_timing.json`: the same 16-worker pool and batch-512 GPU inference fruit08 gets, over exactly the 226,562 meshes the BEM sweep solved. Mean of five runs (149.667 .. 162.574 s, sd 5.418 s), 0.678 ms per mesh. Features dominate at 152.2 s; inference is 1.49 s. The spread is wider than fruit08's 0.27 s because one run lost ~13 s to background load |
| sound synthesis | 72.0 | FluidSound is a separate project not driven from this repo |

The four simulation stages sum to 20,843.4 s.

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

Our surrogate: total 5,677.1 s = 1.6 h
  LBM simulation            2,890.2 s   48.2 min   50.91%
  Bubble tracking           2,082.1 s   34.7 min   36.68%
  VOF field/bubble tag dump I/O      532.3 s    8.9 min    9.38%
  Mesh extraction              99.7 s    1.7 min    1.76%
  Frequency estimation         58.4 s     58.4 s    1.03%
  Audio synthesis              14.3 s     14.3 s    0.25%

end-to-end speedup: 11.8x
bottleneck with the surrogate: LBM simulation (50.9%)

===== Exhalation =====

BEM reference: total 189,035.4 s = 52.5 h
  LBM simulation           12,767.1 s      3.5 h    6.75%
  Bubble tracking           5,773.8 s      1.6 h    3.05%
  VOF field/bubble tag dump I/O    1,862.8 s   31.0 min    0.99%
  Mesh extraction             439.6 s    7.3 min    0.23%
  Frequency estimation    168,120.0 s     46.7 h   88.94%
  Audio synthesis              72.0 s    1.2 min    0.04%

Our surrogate: total 21,069.1 s = 5.9 h
  LBM simulation           12,767.1 s      3.5 h   60.60%
  Bubble tracking           5,773.8 s      1.6 h   27.40%
  VOF field/bubble tag dump I/O    1,862.8 s   31.0 min    8.84%
  Mesh extraction             439.6 s    7.3 min    2.09%
  Frequency estimation        153.7 s    2.6 min    0.73%
  Audio synthesis              72.0 s    1.2 min    0.34%

end-to-end speedup: 9.0x
bottleneck with the surrogate: LBM simulation (60.6%)
```


