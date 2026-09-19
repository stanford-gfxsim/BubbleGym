# Table 2 — the learned model against baseline regressors

Accuracy, parameter count and per-bubble cost for each regressor.

**No single script prints this table.** Its columns come from two places, both
shipped.

The `t` column is `nn_framework_timing_tables.json` here. Its `table2` block
holds the printed microseconds per bubble, together with the matching test
MAPEs and the protocol: one batched call over the 1,500-row test set, warm-up
20, mean of 200.

The accuracy and parameter columns come from the fitted models in
`python/freq_model/output/output_baseline_regressors_8feature/`, whose
`export_config.json` carries the numbers the paper prints. Table 2 was refit on
the 10k benchmark, so any older ledger you find reports different accuracy.

`baseline_metrics.json` (and its `baseline_metrics.md` rendering) is the
regressors' ledger at those exported settings: test MAPE, RMSE(log f) and max
APE for each of the four baselines and the MLP, plus the hyperparameters. Four
scripts read it for those settings, and `export_config.json` names it as its
source. Regenerate it with

```bash
python python/freq_model/regression/baseline_regressors_8feat.py \
    --hparams-json python/freq_model/output/output_baseline_regressors_8feature/export_config.json
```

The Fruit-Splash scene MAPE and max columns have no shipped source.
