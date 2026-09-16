# Fig. 4 — per-bin model error

Frequency-model error across ten bins of increasing nonsphericity, ten bubbles
per bin, for two analytic baselines and two learned surrogates.

**The published figure used six-feature models trained on an old incomplete
predecessor of the benchmark.** Its evaluation was recovered from the authoring
machine and ships here as data. Both surrogates have since been **retrained on
the released 10k dataset with the production eight-feature set**, and the new
plots are drawn from those.

MAPE (%) over Fig. 4's 100 bubbles:

| | Minnaert | Ellipsoid | Residual | Direct |
| --- | --- | --- | --- | --- |
| published, 6 feat | 4.449 | 1.845 | 0.311 | 0.144 |
| retrained, 8 feat | 4.449 | 1.845 | **0.114** | **0.112** |

The baselines are formulas, not fits, so they do not move. Retrained at matched
features the surrogates reproduce the published curves to within 0.04 pp; the
gain to ≈0.11 % comes from the two inertia-ratio features, not the extra data.
At eight features the residual and direct objectives converge.

## Figures

| File | What |
| --- | --- |
| `per_bin_mape_4series_curves.png` | the figure as published |
| `per_bin_mape_retrained_10k_only.png` | **the new plot** — same layout and same 100 bubbles, retrained models |
| `per_bin_mape_retrained_10k.png` | the retrains laid against the published curves (`_6feat` = the matched-feature control) |
| `feature_axis_panels.png` | one panel per model feature, binned on each in turn |
| `alt_bin_sets.png` | four stratification axes side by side |
| `alt_bin_sets/<axis>/per_bin_mape.png` | a standalone panel for each of the eleven axes |

## Data

| File | What |
| --- | --- |
| `selected_rows.csv` | the 100 evaluated bubbles: identity in the 10k CSV, features, BEM ground truth, every prediction and its APE |
| `per_bin_mape.csv` | the published figure's curves |
| `per_bin_mape_retrained_10k.csv` | the retrained curves, all variants |
| `feature_axis_panels.csv` | per-feature totals |
| `summary_{residual,direct,minnaert}.json`, `summary_three_way.json` | the original 2026 evaluation summaries |
| `alt_bin_sets/<axis>/` | `selected_rows.csv`, `per_bin_mape.csv`, `summary.json` per axis |

The archived rows were mapped onto `dataset_bubblegym_10k.csv` by joining on
`(source, mesh_filename)`: 100 of 100 matched, geometry and `frequency`
bit-identical. Recomputing the published panel from `selected_rows.csv`
reproduces it to 3.2e-15 pp.

## Every point is out of sample

Every bubble scored in these figures is absent from the training and validation
sets of the model scoring it. The scripts assert it rather than assume it.

The models are retrained on this split.

## Reproducing

```bash
# retrain (objective x feature set), holding out Fig. 4's 100 bubbles
for f in 8 6; do for b in strasberg none; do
  python python/freq_model/NN/fit_shape_freq_model_variants.py \
      --features $f --baseline $b \
      --holdout-mesh-ids results/experiments/fig04_per_bin_model_error/selected_rows.csv
done; done

# the new plot, in the published layout
python python/visualization/plot_fig04_retrained_comparison.py --only-retrained

# per-axis bin sets and their panels
for a in i11_over_i00 i22_over_i00 non_sph_va non_sph_vm non_sph_w eta_V eta_A eta_M \
         inertia chull willmore; do
  python python/freq_model/NN/eval_alt_bin_sets.py --bin-by $a
done
python python/visualization/plot_feature_axis_panels.py
python python/visualization/plot_alt_bin_sets.py
```

Checkpoints land in `python/freq_model/output/fig04_retrain/<N>feature_<objective>/`.
The thumbnail strips come from
`dataset/bubble_gym/bubble_mesh_thumbnails_100x100/`, which ships a render for
every benchmark bubble; `python/visualization/bin_thumbnail_strip.py` places
them. The 2026 checkpoints were never archived, so the published curves cannot
be regenerated — they ship as data instead.
