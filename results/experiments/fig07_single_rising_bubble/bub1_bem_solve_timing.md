# BEM solve timing — bubble 1 (282 mesh frames)

- frames: 0..14400
- vertices: min 2622, median 5805, max 7302
- solved with the cap disabled for 223 frames >5,000 verts; 59 came from the main sweep

| stage | total | mean | median | min | max |
|---|---|---|---|---|---|
| matrix assembly | **486.9 min** | 103.603 s | 110.916 s | 17.370 s | 189.914 s |
| GMRES solve | **11.1 min** | 2.361 s | 2.455 s | 0.322 s | 6.057 s |
| BEM total | **498.0 min** | 105.964 s | 113.138 s | 17.695 s | 193.948 s |
| NN8 features | 21.1 s | 74.71 ms | 76.48 ms | 48.21 ms | 99.46 ms |
| NN8 inference (GPU) | 0.09 s | 0.315 ms | 0.298 ms | 0.287 ms | 0.718 ms |

**Total BEM wall time for bubble 1: 8.30 h (498.0 min).** Assembly is 97.8% of it.

Against the NN surrogate on the same 282 meshes: 21.2 s total, i.e. **1412x**.

The paper prints **1428x**: the `t_feat_s` column above was sampled inside the
BEM sweep with all 24 cores busy (74.71 ms/mesh), while Table 1 and the Fig. 7
legend use the standalone NN pass over the same 282 meshes
(`single_bubble_nn_timing.json`, key `bub1` — 74.18 ms/mesh, 20.92 s total).
The BEM side (103.603 + 2.361 s) is identical in both.

| vertices | frames | mean BEM s | total min |
|---|---|---|---|
| 0-4,000 | 21 | 25.93 | 9.1 |
| 4,000-5,000 | 37 | 64.36 | 39.7 |
| 5,000-6,000 | 114 | 98.90 | 187.9 |
| 6,000-7,000 | 107 | 141.29 | 252.0 |
| >7,000 | 3 | 187.70 | 9.4 |

Per-frame ledger: `bub1_bem_solve_timing.csv`
