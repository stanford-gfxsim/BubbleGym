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

One caveat: the `sound synthesis` slice was measured in FluidSound, a separate
project not driven from this repository, so the pie script reads that number
from the ledger rather than re-measuring it. Every other slice rebuilds from
data here.
