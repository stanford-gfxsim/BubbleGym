# `python/` — what lives where

74 modules in eight packages. Section numbers refer to the BubbleGym paper.

| Package | Files | Role |
|---|---:|---|
| [`bem/`](bem) | 9 | Galerkin BEM ground truth (§4.1) |
| [`shape_feature/`](shape_feature) | 4 | The eight shape descriptors (§4.2) |
| [`freq_model/`](freq_model) | 26 | Network, baselines, training, evaluation, ablations (§5–6) |
| [`tracked_bubinfo/`](tracked_bubinfo) | 3 | The `trackedBubInfo` format layer |
| [`scene/`](scene) | 12 | Per-scene drivers that write each frequency column |
| [`visualization/`](visualization) | 8 | Paper figures and the interactive HTML |
| [`utils/`](utils) | 9 | Timing benchmarks and figure scripts |
| [`tests/`](tests) | 3 | Unit tests |

Run everything from the repo root with `PYTHONPATH=python`.

---

## `bem/` — ground truth

`compute_freq_bempp_galerkin.py` is the solver: it computes bubble capacitance
with a mixed Galerkin BEM and converts it to frequency. Note it
rescales the mesh to unit volume **only when handed a path** — a `(v, f)` tuple
is used as given. It also checks the right-hand side against the solid-angle
identity (`b ≡ −1` for a closed, outward-oriented surface) and refuses a mesh
whose residual exceeds `rhs_orientation_tol`.

- **Solver + support**: `bempp_io.py` (mesh loading, plus the grid and API shims
  that work with either the `bempp` or `bempp_cl` package name),
  `geometry_utils.py`, `_threading.py`.
- **Validation**: `validate_galerkin_ellipsoid.py`,
  `validation/compare_bem_with_strasberg_minneart.py`.
- **Dataset build driver**: `solve_dataset_bem.py` — the resumable
  batch driver that produced the ground-truth frequencies in the shipped CSV. 
- **Dataset-scale re-solve sweep**: `eval_galerkin_full_dataset.py`, a parallel
  resumable sweep over a whole dataset CSV. The shipped CSV already contains
  these frequencies; this rebuilds them from the meshes at the same settings.

## `shape_feature/` — descriptors

The measurement half of §4.2, all operating on unit-volume meshes, in three
modules:

- `mesh_utils.py` — OBJ I/O, watertightness checks, connected components,
  orientation, signed volume / area / rescale, Mirtich–Eberly moments (Eq. 6),
  and the cached three-iteration Laplacian filter every marching-cubes consumer
  applies first.
- `nonspherical_features.py` — mean curvature `M` and Willmore energy
  (Eq. 7–8), the convex-hull η ratios (Eq. 10), and the eight features the
  frequency model consumes. `nonspherical_features_from_unit_mesh()` is the
  in-memory entry point used at inference time.
- `build_dataset.py` — computes every feature for a list of meshes in one pass
  per mesh, with a process pool and an on-disk cache, and writes the columns
  back to a dataset CSV.

Every feature assumes a **closed** surface: `load_obj_mesh` validates topology
and raises `NonWatertightMeshError` on a mesh with holes or non-manifold edges,
rather than returning an integral computed over an open surface.

## `freq_model/` — model, baselines, evaluation

- **Network** (`NN/`): `bub_freq_net.py` (architecture, §5.2.2),
  `fit_shape_freq_model.py` (the one trainer — 8 features → `log f`, Eq. 16),
  `nn_inference.py` (load a checkpoint and run it over per-frame features),
  `bubble_dataset.py`, `training.py`, `stratified_eval_model.py` (per-bin error,
  `--model-kind nn8`, feeding `visualization/plot_per_bin_mape_bars.py`),
  `add_baseline_frequencies_to_dataset.py`.
- **Analytic baselines** (`analytical/`): `minnaert_freq.py`,
  `strasberg_freq.py` (§5.1), `physics.py` (capacitance↔frequency relations).
- **Non-neural regressors (Table 2)** (`regression/`): `baseline_regressors_8feat.py`,
  `export_baseline_models_8feat.py`, `measure_baseline_inference_time.py`.
- **Ablations** (`ablation_study/`), the §6 sweeps and their controls:
  - `run_exp1_features.py` — the feature-subset sweep: retrains the network on
    each subset of the eight descriptors (`all8`, `chull`, `inertia`,
    `inertia_chull`, `inertia_nonsph`, …), writing one
    `results/experiments/supp_table01_feature_ablation/<variant>/` per run with checkpoint, `metrics.json`,
    `curated_eval.json` and `split.json`. Backs supplement Table 1 and Figs. 1–2.
  - `run_exp2_arch.py` — the width sweep: the same trainer over the five hidden
    widths `h16_h8 … h256_h128`, into `results/experiments/supp_table02_width_ablation/`. Its checkpoints are
    what the timing benchmark loads, and its accuracy is supplement Table 2.
  - `aggregate_results.py` — collapses either sweep into `summary_exp{1,2}.csv`
    and the supplement's bar figures (`--dark` for the web-page copies).
  - `cross_source_generalization.py` — trains on one data source and tests on
    the other (Tim2016 VOF vs LBM), the cross-source numbers §6 quotes.
  - `near_duplicate_controlled_mape.py` — train/test leakage control: re-scores
    the test split with near-duplicate shapes held out.
  - `benchmark_width_timing.py` + `plot_width_timing.py` — supplement Fig. 5 and
    Table 1's inference constant; see `README_timing.md` for the two measurement
    traps that will silently corrupt the result.
  - `ablation_common.py` — the shared split, feature and metric plumbing all of
    the above use, so every variant is scored under one protocol.
- **Shipped artifacts**: `output_8feature_direct_bubblegym_10k/`
  (checkpoint, scalers, `split.json`, `metrics.json`) and
  `output_baseline_regressors_8feature/`.

## `tracked_bubinfo/` — the file format

`trackedBubInfo.txt` is the bubble-tracker export shared by the LBM simulator
and the FluidSound renderer, both separate projects.

- `io.py` — parse blocks and sample lines, read the per-bubble header radii,
  rewrite the frequency column, optionally truncate blocks at a cutoff time.
- `mesh_index.py` — discover the per-bubble marching-cubes OBJ tree that
  accompanies a tracked file, resolve the simulation timestep, and map each
  sample line to its nearest mesh frame.

## `scene/` — writing a frequency column

`_common/` holds the three writers, one per frequency model. All take a base
tracked file and emit a variant that differs only in the `f` column, so any two
are directly comparable:

- `write_trackedbubinfo_nn.py` — the learned surrogate over per-bubble meshes.
- `write_trackedbubinfo_bem.py` — BEM ground truth, grafted from an all-bubbles
  sweep by nearest mesh frame. Pass `--dt-lbm` explicitly; the module docstring
  explains why inference is not safe here.
- `replace_trackedbubinfo_freq_with_minnaert.py` — analytic, constant per bubble.
- `drop_nonwatertight_bubbles.py` — the shared validity gate, applied *before*
  any estimator so every model sees the same bubble population.

`bubble_theater/` drives the three procedural sequences of Fig. 1 end to end
(frequencies, per-method trackedBubInfo, `freq_curves.csv` and the overlay plot)
and holds the Fig. 9 regressor-comparison plot. `exhalation/` holds
`plot_nn_vs_minnaert.py`, the NN-vs-Minnaert comparison plot for the Fig. 6
scene.

## `visualization/` — figures

`plot_dataset_distribution_3d_png.py` generates the paper's shape-space figures
(linear axes, matplotlib `rainbow`). `plot_nonsphericity_axes_3d_html.py` is the
interactive counterpart: any three of the eight descriptors on the axes, VOF/LBM
toggles, per-bubble thumbnails with feature bars, and a light/dark theme.
`plot_selected_bubbles_features.py` is the bubble grid in Fig. 2's left panel,
`visualize_bubble_vs_hull.py` the §4.2.3 hull illustration,
`render_dataset_thumbnails.py` renders one PNG per benchmark bubble, and
`plot_dataset_distribution.py` holds the shared thumbnail renderer.

## `utils/` — timing and figure data

Table 1's columns come from four of these, which are not variants of one
another: `nn8_vs_bem_timing.py` (per-bubble NN vs BEM),
`nn8_vs_bem_all_bubbles_chunked.py` (full-scene sweep),
`batched_nn8_inference.py` (batched GPU inference), and
`apply_regressors_all_bubbles.py` (regressor accuracy/timing on timed meshes).
`plot_end2end_timing_pies.py` draws Fig. 8 and `plot_single_bubble_freq.py`
Fig. 7's frequency panel. `mesh_complexity_stats.py` reports vertex/face/genus
statistics for a dataset's meshes, and `worst_case_utils.py` is shared
tail-error helper code. Recorded outputs live in `utils/results/`.
