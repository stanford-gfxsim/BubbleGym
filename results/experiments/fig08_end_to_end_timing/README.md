# Fig. 8 — end-to-end cost

Two pies, Fruit Splash and Exhalation, showing where the wall clock goes.

```bash
python python/utils/plot_end2end_timing_pies.py \
    --out fig/fruit08_exhale08_end2end_timing_pies.png
```

It defaults to the two ledgers here, `fruit08_end2end_timings.json` and
`exhale15_end2end_timings.json`, and prints the per-stage shares, the totals and
the end-to-end speedup, so the caption can be restated from the same source.

Add `--dark` for the project-page rendering.

Where every slice of both pies comes from, stage by stage, is in
[`fig8_end2end_timing.md`](fig8_end2end_timing.md).

Two caveats, both recorded there: the `sound synthesis` slice was measured in
FluidSound, a separate project not driven from this repository, so the pie
script reads that number from the ledger rather than re-measuring it; and the
exhalation pie's `frequency estimation` slice for our surrogate is carried from
an earlier ledger too, since re-measuring it needs the per-bubble mesh tree,
which is simulator output and not part of the release. Every other slice
rebuilds from data here.
