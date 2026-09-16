# Supplement Table 1 — feature ablation

Six feature subsets, each trained with the production width, 64/32.

Train all six variants, then aggregate:

```bash
python python/freq_model/ablation_study/run_exp1_features.py
python python/freq_model/ablation_study/aggregate_results.py
```

The first writes one subfolder per variant here (`all8`, `chull`, `inertia`,
`inertia_chull`, `inertia_nonsph`, `nonsph`), each with a checkpoint, its
scalers, `metrics.json`, `curated_eval.json` and `split.json`. The second reads
those and writes `summary_exp1.csv`, the table's source, plus the two bar
charts the supplement prints as its Figures 1 and 2. Add `--dark` for the
project-page copies.

**Those runs ship, so the first command skips every variant** and finishes
immediately. Pass `--force` to retrain, which overwrites what is here. With no
`--variants`, all six run.

Note the stratified panel is binned on **inertia anisotropy**, ten bubbles per
bin, not on Wadell nonsphericity. The `bin_edges_selected` array in every
`*/curated_eval.json` matches that scalar's quantile edges over the test split
exactly, and no nonsphericity measure reaches its range.
