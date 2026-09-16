# Supplement Table 2 — network width

Five widths from 16/8 up to 256/128, with the eight-feature set fixed.

Train all five variants, then aggregate:

```bash
python python/freq_model/ablation_study/run_exp2_arch.py
python python/freq_model/ablation_study/aggregate_results.py
```

The first writes one subfolder per width here, each with a checkpoint, its
scalers, `metrics.json`, `curated_eval.json` and `split.json`. The second reads
those and writes `summary_exp2.csv`, the table's accuracy source, plus the bar
charts the supplement prints as its Figures 3 and 4.

**Those runs ship, so the first command skips every width** and finishes
immediately. Pass `--force` to retrain, which overwrites what is here. With no
`--variants`, all five run.

The inference column comes from `width_timing.json` instead:

```bash
python python/freq_model/ablation_study/benchmark_width_timing.py
python -m python.freq_model.ablation_study.plot_width_timing \
    --out fig/ablation_exp2_inference_batch512.png
```

That plot is the supplement's Figure 5. Read
`python/freq_model/ablation_study/README_timing.md` first: it documents the
protocol and two measurement traps that silently corrupt the result.

`width_timing.json` also supplies Table 1's inference constant, so the two
numbers are the same measurement, not independent ones.

The `h64_h32` row here is this sweep's retrain, not the shipped production
model. It reads test MAPE 0.089 and RMSE(log) 0.00234, where
`python/freq_model/output/output_8feature_direct_bubblegym_10k/metrics.json`
reads 0.0886 and 0.002301. Same architecture, different training run, so the
two are not expected to match bit for bit.
