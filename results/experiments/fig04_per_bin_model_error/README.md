# Fig. 4 — per-bin model error

Frequency-model error across ten bins of increasing nonsphericity, ten bubbles
per bin, for two analytic baselines and two learned surrogates. Both surrogates
use the production eight features and are trained on the released 10k dataset.

## What the paper prints

`per_bin_mape_wadell_10k.png` is Fig. 4: 100 bubbles of the models' test split,
binned on Wadell nonsphericity (`alt_bin_sets/non_sph_va`). The direct head
scores 0.160 % MAPE against 0.187 % for the residual head, lower in eight of the
ten bins. The ten renders beneath the axis illustrate each bin's shape range;
`plot_fig04_wadell_panel.py` derives them and checks each falls inside its bin.

`heads_feature_panels.png` is the supplement's Section C figure: the same test
split stratified on each of the eight input features in turn. The direct head
has the lower overall error on all eight panels, and 0.083 % against 0.098 % on
the full 1,485-row test split.

Every bubble scored here is absent from the training and validation sets of the
model scoring it; the scripts assert that rather than assume it.

## Files

| File | What |
| --- | --- |
| `per_bin_mape_wadell_10k.png` | the paper's Fig. 4 |
| `heads_feature_panels.png` | the supplement's Section C figure |
| `alt_bin_sets/<axis>/` | per axis: `per_bin_mape.csv`, `selected_rows.csv`, `summary.json`, and a standalone panel |
| `selected_rows.csv` | the 100-bubble hold-out set both heads were trained without (the renders under Fig. 4 come from it): identity in the 10k CSV, features, BEM ground truth, and the two analytic baselines with their APE (the heads' predictions on these bubbles are in each head's `holdout_predictions.csv`) |
| `paper_thumbnails.json` | the ten bubbles rendered under Fig. 4, and how they were identified |
| `feature_axis_panels.csv` | the per-feature panels, as data |

Only the two published figures ship. The working plots (`feature_axis_panels.png`,
`alt_bin_sets.png`) are rebuilt by the last two commands below.

## Reproducing

```bash
# train both heads, holding out the 100-bubble hold-out set
for b in strasberg none; do
  python python/freq_model/NN/fit_shape_freq_model_variants.py \
      --baseline $b \
      --holdout-mesh-ids results/experiments/fig04_per_bin_model_error/selected_rows.csv
done

# per-axis bin sets
for a in i11_over_i00 i22_over_i00 non_sph_va non_sph_vm non_sph_w eta_V eta_A eta_M \
         inertia chull willmore; do
  python python/freq_model/NN/eval_alt_bin_sets.py --bin-by $a
done

python python/visualization/plot_fig04_wadell_panel.py      # the paper's Fig. 4
python python/visualization/plot_heads_feature_panels.py    # the supplement's figure
python python/visualization/plot_feature_axis_panels.py     # working plots, not shipped
python python/visualization/plot_alt_bin_sets.py
```

Checkpoints land in `python/freq_model/output/fig04_retrain/8feature_<objective>/`.
The bubble renders come from `dataset/bubble_gym/bubble_mesh_thumbnails_100x100/`,
placed by `python/visualization/bin_thumbnail_strip.py`.
