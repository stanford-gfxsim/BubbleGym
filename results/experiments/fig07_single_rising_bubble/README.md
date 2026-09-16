# Fig. 7 — BEM ground truth for the single rising bubble

BEM frequencies for every frame of the bubble Fig. 7 follows, so the figure's
learned-model curve can be shown against ground truth rather than against
Minnaert alone. This directory is also the source of Table 1's **Single Rising
Bubble** row and of the two totals in the Fig. 7 legend.

## What is here

| File | What |
| --- | --- |
| `bub1_only_nocap_all_rows.csv` | one row per mesh frame: BEM frequency, assembly/solve times, NN frequency and its timings, vertex counts |
| `bub1_only_nocap_all_summary.json` | run summary — solver settings, hardware, totals |
| `bub1_bem_solve_timing.csv` / `.md` | the per-frame BEM ledger and its digest; **this is where Table 1's `103603+2361` ms comes from** |
| `single_bubble_nn_timing.csv` / `.json` | the standalone NN pass over the scene (3,078 meshes); its `bub1` block is Table 1's `74.2` ms and the legend's `20.9 s` |
| `bub1_only_nocap_regressor_freqs.csv` | the baseline regressors (incl. degree-2 polynomial) on the same meshes |
| `bub1_only_nocap_regressor_summary.json` | their summary |
| `vertex_scan.csv` | vertex count per mesh, from the pre-pass that sized the sweep |

## The run

| | |
| --- | --- |
| Meshes solved | 282 — every frame of `bub_1`, 0 failures |
| BEM cost | 8.30 h total; 106.0 s mean, 193.9 s worst |
| Solver | bempp-cl Galerkin, P1–DP0, mass-preconditioned GMRES to 1e-12, quadrature 6/6 |
| Hardware | Intel i9-14900K + RTX 5090 — the Sec. 6 machine |
| Completed | 2026-09-11 |
| Frequencies | 5.29–6.75 Hz (unit-volume; physical frequencies follow the V^-1/3 scaling) |

## How Table 1's row is assembled

| Table 1 cell | value | file | key |
| --- | --- | --- | --- |
| $t^{\text{BEM}}$ | `103603+2361` ms | `bub1_bem_solve_timing.md` | assembly / GMRES means |
| $t^{\text{Ours}}$ | `74.2+0.0005` ms | `single_bubble_nn_timing.json` | `bub1.feat_mean_s`; the inference constant is the batch-512 figure shared by every row |
| Speedup | `1428x` | — | $(103603+2361)/74.2005$ |

**The NN number does not come from the rows CSV**, whose `t_feat_s` reads
74.71 ms. That column was sampled while the BEM sweep had all 24 cores busy;
the standalone pass measures the surrogate as it would actually run. See
`bub1_bem_solve_timing.md` for the full note. The BEM side is the same either
way.

### Superseded measurement

An earlier sweep of these same 282 meshes on a Threadripper 9970X (7.29 h,
865x) was re-collected on the i9-14900K of Sec. 6, which is what ships here.
The frequencies agree to 1e-15 between the two, so Fig. 7's curves are
identical; only the legend totals and Table 1's row moved.

## Provenance of the meshes

Source meshes, named in the `path` column of the rows CSV:

```
selected-dataMR3D_single_bubble_dx2p00mm_v6_rollL/
    ppm_ve_home_test_phi_iter/mc_surface_per_bubble_smoothed_lap3/bub_1/
```

That run is the same simulation instance as the `single_bubble_04-dx2mm`
delivery the NN and Minnaert curves come from — verified by comparing the
delivery's intermediate `trackedBubInfo_bridged.txt` against the run's, which
are byte-identical. The two sets of curves are therefore directly comparable;
they are not separate runs of the same configuration.

## Using it

`plot_single_bubble_freq.py` takes both files directly:

```bash
python python/utils/plot_single_bubble_freq.py \
    --out fig7_freq.png \
    --bem-rows   <this dir>/bub1_only_nocap_all_rows.csv \
    --poly2-rows <this dir>/bub1_only_nocap_regressor_freqs.csv \
    --legend-times
```

`--bem-rows` overlays the ground truth, `--poly2-rows` the degree-2 baseline,
and `--legend-times` appends each method's total compute time for this bubble
to its legend entry. `--legend-times` reads the rows CSV, so it prints the
sweep's own NN total (21.1 s) rather than the standalone 20.9 s the published
legend carries.

## Scope

Only `bub_1` was solved for the figure. The scene holds **797 bubbles / 4,676
meshes**, so 4,394 meshes remain unsolved. That was deliberate: the whole-scene
sweep was estimated at ~248 h, and 96% of it was bubbles this figure does not
follow. The wider `single_bubble_rollL` sweep that produced the NN timing file
covers 3,078 of those meshes; only bubble 1 was solved with the 5,000-vertex cap
disabled.
