# fruit08 end-to-end pipeline timing

Figure: `fruit08_exhale08_end2end_timing_pies.png`, drawn by
`python/utils/plot_end2end_timing_pies.py`, which reads
`fruit08_end2end_timings.json` and `exhale15_end2end_timings.json`.

Scene: five-fruit drop, 240x240x320 grid, dx = 1/480 m, dt = 1/11520 s, 4.0 s,
46,080 LBM iterations, 240 exported frames (60 fps).

## Hardware and run setting

Every stage of the Fruit Splash pie was measured on **one workstation**, the one
the paper names: Intel Core i9-14900K (24 cores / 32 threads), NVIDIA GeForce
RTX 5090, 96 GB RAM.

The frame-driven stages (simulation, tracking, dump, mesh extraction) come from
a **timing rerun** of the fruit08 scene on that machine, made on 2026-09-16 to
replace an earlier measurement taken on a different workstation. It used the
same simulator setting as the exhalation run behind the second pie, so the two
pies are directly comparable:

```
lbm_flow_proj.exe --config sim_setting_fruit_dx2p08mm_fullrate_v9.yaml \
    --export-mc --no-gui --bub-track --bub-track-sample-stride 1 \
    --bub-mesh-iter-stride 50 --export-per-bubble
```

- Simulator `module/lbm_sim` at `2f88f0e`, the commit the original fruit08 run
  recorded, built Release for CUDA compute capability 12.0.
- Tracker bridging at its default and the context-aware shell export off, both
  as in the exhalation run.
- 123,729 per-bubble meshes written. The GPU simulation is not bit-deterministic,
  so this differs slightly from the original run's 125,262; the rerun exists
  only to time the stages.
- The shipped `trackedBubInfo_*.txt` files and the BEM sweep below still come
  from the original run instance, which the delivery was built from.

## Stage costs

| stage | seconds | source |
|---|---|---|
| LBM simulation | 2,890.23 | rerun `log.txt`, `[stage timing final] sim=` (62.72 ms/iter) |
| bubble tracking | 2,082.07 | same line, `track=` (45.18 ms/iter) |
| phi/tag dump I/O | 532.34 | same line, `dump_io=` (11.55 ms/iter) |
| per-bubble mesh extraction | 99.70 | `[stage timing final] per_bubble_mc_export=` |
| frequency prediction â€” BEM | 61,488.0 (17.08 h) | sweep `table01_timing_across_scenes/all_bubbles/fruit08/fruit08_all_summary.json`, `total_bem_hours`; 75,686 solves, mean 0.812 s (assemble 0.789 / solve 0.023), median 0.471 s, max 81.2 s |
| frequency prediction â€” NN8 (ours) | 52.49 | `fruit08_nn_step_timing.json`; see "Surrogate frequency step" below |
| sound synthesis | 14.30 | wall time of `runFluidSound.exe` rendering the shipped `dataset/fruit_splash/trackedBubInfo_NN.txt` (389.7 MB, file cache warm) to the full 4.0 s at 48 kHz, scheme 0 (uncoupled), damping 0.8; median of three runs (14.47 / 14.30 / 14.17 s), byte-identical output. FluidSound is a separate project not driven from this repo |

The four simulation stages sum to 5,604.3 s against the run's own 5,611.1 s
wall clock, so 99.9 % of the run is accounted for.

BEM settings: P1-DP0, mass preconditioner, GMRES tol 1e-12, quadrature 6/6
(NUMBA_NUM_THREADS=24). Sweep: 77 chunks of <= 1000 meshes, 75,885 rows =
75,686 BEM + 199 NN-only, 780 failures (1.0 %); 25.5 h wall including a ~1 h
restart loss (resumed from chunk sentinels).

## Printed by the plot script

```
BEM reference: total 67,106.6 s = 18.6 h
  LBM simulation            2,890.2 s   48.2 min    4.31%
  Bubble tracking           2,082.1 s   34.7 min    3.10%
  Phase field/bubble tag dump I/O      532.3 s    8.9 min    0.79%
  Mesh extraction              99.7 s    1.7 min    0.15%
  Frequency estimation     61,488.0 s     17.1 h   91.63%
  Audio synthesis              14.3 s     14.3 s    0.02%

Our surrogate: total 5,671.1 s = 1.6 h
  LBM simulation            2,890.2 s   48.2 min   50.96%
  Bubble tracking           2,082.1 s   34.7 min   36.71%
  Phase field/bubble tag dump I/O      532.3 s    8.9 min    9.39%
  Mesh extraction              99.7 s    1.7 min    1.76%
  Frequency estimation         52.5 s     52.5 s    0.93%
  Audio synthesis              14.3 s     14.3 s    0.25%

end-to-end speedup: 11.8x
bottleneck with the surrogate: LBM simulation (51.0%)
```

## Caption / prose numbers (restate from the block above, never hand-edit)

With BEM the frequency stage is 91.6 % of an 18.6 h pipeline; the learned
surrogate brings the whole pipeline to 1.6 h (11.8x end-to-end), after which
the LBM simulation itself (51.0 %) and bubble tracking (36.7 %) dominate and
frequency prediction is 0.9 %. These are the numbers the paper's Fig. 8
caption should carry.

The exhalation pie is left as it was: its four simulation stages are the exact
`[stage timing final]` values of its run log on this workstation, and its BEM
sweep ran here too, but its surrogate frequency and audio slices were not
re-measured, so that pie is not yet a verified single-machine clock.

## Surrogate frequency step

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
