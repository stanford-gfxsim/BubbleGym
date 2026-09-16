# Fig. 9 — baseline regressors on the procedural scenes

The learned model against the fitted baseline regressors, frame by frame.

```bash
python python/scene/bubble_theater/plot_baseline_regressor_freq_curves.py
```

Two inputs, both shipped. The per-frame curves come from the three
`freq_curves.csv` files in
[`fig01_bubble_theater/`](../fig01_bubble_theater), and the fitted baselines
from `python/freq_model/output/output_baseline_regressors_8feature/`.

The same regressors supply the accuracy columns of
[Table 2](../table02_regressor_comparison).
