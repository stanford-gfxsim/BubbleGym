# Fig. 9 — baseline regressors on the procedural scenes

The learned model against the fitted baseline regressors, frame by frame:
ellipsoid and curl noise (240 frames over 6 s) and the Enright test (120 frames
over 4 s).

```bash
# fit the baselines and evaluate them on every mesh frame; writes
# <scene>_result/freq_curves_baselines.csv next to each scene's freq_curves.csv
python python/scene/bubble_theater/plot_baseline_regressor_freq_curves.py

# stack the three scenes into the figure
python python/scene/bubble_theater/plot_baseline_regressor_freq_curves.py --stacked \
    --stacked-out results/experiments/fig09_regressor_comparison/baseline-regressors-vertical.png
```

Inputs, all shipped: the per-frame Minnaert, ellipsoid-proxy, learned-model and
BEM curves from the three `freq_curves.csv` files in
[`fig01_bubble_theater/`](../fig01_bubble_theater); the meshes and
`trackedBubInfo_Minnaert.txt` (for `R_eq`) in `dataset/bubble_theater/<scene>/`;
and the baseline hyperparameters recorded in
[`table02_regressor_comparison/baseline_metrics.json`](../table02_regressor_comparison/baseline_metrics.json).
The baselines are re-fitted on the paper's training split (asserted against the
released checkpoint's `split.json`).

The same regressors supply the accuracy columns of
[Table 2](../table02_regressor_comparison).
