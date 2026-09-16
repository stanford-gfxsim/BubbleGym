# Fig. 2 — the benchmark's shape space

Three panels: a grid of six bubbles with their shape descriptors as bars, and
two 3D scatters of the dataset coloured by frequency ratio `f/f_M`.

The two 3D panels, at the published camera:

```bash
python python/visualization/plot_dataset_distribution_3d_png.py \
    --output-dir results/experiments/fig02_dataset_shape_space \
    --elev 20.7 --azim 115 --no-preview
```

Without `--elev/--azim` the script opens an interactive window and saves
whatever angle you leave it at. The published angle is also recorded in
`dataset_distribution_camera.json`.

The left bubble grid:

```bash
python python/visualization/plot_selected_bubbles_features.py \
    --selected results/experiments/fig02_dataset_shape_space/selected_bubbles.txt
```

Both read `dataset/bubble_gym/dataset_bubblegym_10k.csv`. The three panels are
combined into the composite by hand; no script does that.
