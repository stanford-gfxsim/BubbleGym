# Experiments — every figure and table, and what produced it

One folder per figure or table in the paper, named by its printed index. Each
holds the artifact as it appears in the paper, whatever raw data produced it,
and **a README saying how to rebuild it**. Start there; this file is only the
index.

Numbers below are those printed in the PDF, not `\label` names.

**The filenames in the paper do not match the filenames the scripts write.**
Figures were copied into the paper's `fig/` directory and renamed by hand, so
the mapping here is the only record of which output became which figure. `fig/`
lives in the paper (LaTeX) repository, not in this one, so every `fig/...` name
below is a destination, not a path you can open.

## Main paper

| Folder | Fig. | Name in the paper |
|---|---|---|
| [`fig01_bubble_theater/`](fig01_bubble_theater) | 1 | `fig/teaser/*/freq_curves.jpg`, `fig/teaser/*/*.jpg` |
| [`fig02_dataset_shape_space/`](fig02_dataset_shape_space) | 2 | `fig/dataset_10k_distribution.png` |
| [`fig03_nonsphericity_distribution/`](fig03_nonsphericity_distribution) | 3 | `fig/fig3.jpg` |
| [`fig04_per_bin_model_error/`](fig04_per_bin_model_error) | 4 | `fig/per_bin_mape_4series_curves.png` |
| [`fig05_fruit_splash/`](fig05_fruit_splash) | 5 | `fig/fruits_final.jpg` |
| [`fig06_underwater_exhalation/`](fig06_underwater_exhalation) | 6 | `fig/exhale_final.jpg` |
| [`fig07_single_rising_bubble/`](fig07_single_rising_bubble) | 7 | `fig/fig7-single-bubble.jpg` |
| [`fig08_end_to_end_timing/`](fig08_end_to_end_timing) | 8 | `fig/fruit08_exhale08_end2end_timing_pies.png` |
| [`fig09_regressor_comparison/`](fig09_regressor_comparison) | 9 | `fig/baseline-regressors-vertical.png` |
| — | — | `fig/BVP.png`, a hand-drawn diagram |

Fig. 1's fifteen bubble renders are not copied here; they are large and live in
the paper's `fig/teaser/`. The three frequency-curve panels, which are what this
repository's scripts draw, are.

## Tables

| Folder | Table |
|---|---|
| [`table01_timing_across_scenes/`](table01_timing_across_scenes) | 1 |
| [`table02_regressor_comparison/`](table02_regressor_comparison) | 2 |
| [`supp_table01_feature_ablation/`](supp_table01_feature_ablation) | Supp. 1 |
| [`supp_table02_width_ablation/`](supp_table02_width_ablation) | Supp. 2 |

Neither main-paper table is printed by a script. Each folder's README accounts
for its columns cell by cell.

## Supplement figures

Both ablation folders hold one subfolder per variant — checkpoint, scalers,
`metrics.json`, `curated_eval.json`, `split.json` — plus the aggregated summary
and its bar charts. The charts are named by the script that writes them; the
paper renames them on import.

| Supp. Fig. | Name in the paper | Written as | In |
|---|---|---|---|
| 1 | `fig/ablation_exp1_testset_mape.png` | `summary_exp1_testset.png` | `supp_table01_feature_ablation/` |
| 2 | `fig/ablation_exp1_curated_mape.png` | `summary_exp1_curated_overall.png` | `supp_table01_feature_ablation/` |
| 3 | `fig/ablation_exp2_testset_mape.png` | `summary_exp2_testset.png` | `supp_table02_width_ablation/` |
| 4 | `fig/ablation_exp2_curated_mape.png` | `summary_exp2_curated_overall.png` | `supp_table02_width_ablation/` |
| 5 | `fig/ablation_exp2_inference_batch512.png` | `plot_width_timing.py`, from `width_timing.json` | `supp_table02_width_ablation/` |

## Dark-theme copies

The project page serves a dark rendering of each figure it shows. These come
from the same scripts and the same data, with only the lettering and the paper
changed, via a `--dark` flag. The commands are in the folder READMEs. The paper
itself uses only the light renderings.

## Claims with no shipped artifact, reproducible on demand

Two numbers in Sec. 6 belong to no figure folder. Both re-derive from the
shipped dataset and checkpoint. Verified 2026-09-15; the values are what the
commands printed.

**The near-duplicate control**, 0.089 % → 0.16 %. Runs in seconds:

```bash
python python/freq_model/ablation_study/near_duplicate_controlled_mape.py --mode merged
```

It re-derives the headline test MAPE as 0.08858 %, matching the shipped
`metrics.json`, then reports 0.1564 % after dropping the half of the test set
nearest the training set. It also prints a verdict attributing the rise to
shape rarity rather than leakage.

**Cross-source generalization**, 0.47 % / 0.17 %. Each direction trains one
network, a few minutes apiece:

```bash
python python/freq_model/ablation_study/cross_source_generalization.py
python python/freq_model/ablation_study/cross_source_generalization.py \
    --train-source LBM --test-source VOF
```

## Not reproducible in this repository

Printed in the paper, but with no shipped ledger, script input or artifact.
Figure-specific gaps are noted in the folder that owns them; this one belongs
to no folder.

- The **mean mesh size** of the benchmark: 3,959 vertices and 7,914 faces over
  all 10,000 meshes. `mesh_complexity_stats.py` recomputes it, but it needs the
  mesh archive, which is distributed separately: Google drive link to the meshes: https://drive.google.com/file/d/1RaK-cJ7NlHzVyHUncHQMwzrxlvt204PB/view?usp=sharing
