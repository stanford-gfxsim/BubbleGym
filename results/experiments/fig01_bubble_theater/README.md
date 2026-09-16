# Fig. 1 — Bubble Theater

The three frequency-curve panels of the teaser, one per procedural scene.

```bash
python python/scene/bubble_theater/batch_plot_overlay_procedural_results.py
```

It reads the three `freq_curves.csv` files here, one in each `*_result/`
folder, holding `f_minnaert`, `f_strasberg`, `f_nn8` and `f_bem` per frame. The
scenes themselves are in `dataset/bubble_theater/`.

Those CSVs were verified on import: each `f_bem` column is bit-identical, row
for row, to the frequency column of its scene's `trackedBubInfo_BEM.txt`. 

The fifteen bubble renders in the teaser come from
`python/scene/bubble_theater/render_bubble_theater.py` and are not reproduced by
the command above. The thumbnails in `frames/` are downscaled copies.

The same three CSVs also drive [Fig. 9](../fig09_regressor_comparison).
