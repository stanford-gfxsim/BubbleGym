# Non-neural baselines on the 8-feature frequency task

Same features, split (seed 42, 70/15/15, verified against `python/freq_model/output/output_8feature_direct_bubblegym_10k/split.json`), target log(f_BEM), and MAPE formula as the paper's MLP. Hyperparameters fixed from `python/freq_model/output/output_baseline_regressors_8feature/export_config.json`.

| Model | Test MAPE (%) | RMSE(log f) | Max APE (%) |
|---|---|---|---|
| Linear regression | 0.226 | 0.00384 | 3.19 |
| Polynomial (deg 2) + ridge | 0.073 | 0.00168 | 1.93 |
| Polynomial (deg 3) + ridge | 0.058 | 0.00148 | 2.50 |
| RBF kernel ridge | 0.065 | 0.00164 | 2.48 |
| **MLP (paper)** | **0.089** | 0.00230 | 3.71 |
