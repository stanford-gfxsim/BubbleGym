# Fig. 1 — Bubble Theater

The three frequency-curve panels of the teaser, one per procedural scene:
ellipsoid and curl noise (240 frames over 6 s) and the Enright test (120 frames
over 4 s).

```bash
python python/scene/bubble_theater/batch_plot_overlay_procedural_results.py
```

It reads each scene's `trackedBubInfo_{Minnaert,Ellipsoid,NN,BEM}.txt` in
`dataset/bubble_theater/<scene>/` and writes `teaser_{ellipsoid,curl_noise,enright}_freq_curves.jpg`
here. The learned model is the released checkpoint
(`python/freq_model/output/output_8feature_direct_bubblegym_10k`).

The scene files themselves come from the meshes in `dataset/bubble_theater/<scene>/mesh/`:

```bash
python python/scene/bubble_theater/render_bubble_theater.py dataset/bubble_theater/ellipsoid/mesh/bub \
    --methods minnaert,ellipsoid,nn8,bem --shrink-radius-scale 48.5 --frame-count 240 --duration 6 \
    --output-dir <out>
# curl_noise: the same; enright_test: --frame-count 120 --duration 4
```

`<scene>_result/freq_curves.csv` holds the same four series per frame
(`f_minnaert`, `f_strasberg` for the ellipsoid proxy, `f_nn8`, `f_bem`) at full
precision; every column equals its `trackedBubInfo_*.txt` at the files' six
decimals.

The fifteen bubble renders in the teaser are not reproduced by the commands
above. The thumbnails in `frames/` are downscaled copies.

The same three CSVs also drive [Fig. 9](../fig09_regressor_comparison).
