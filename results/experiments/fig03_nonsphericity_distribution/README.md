# Fig. 3 — frequency against nonsphericity

Every benchmark bubble as one point, with mesh thumbnails called out along the
spread.

```bash
python python/visualization/plot_dataset_distribution.py
```

It reads `dataset/bubble_gym/dataset_bubblegym_10k.csv` and the thumbnails in
`dataset/bubble_gym/bubble_mesh_thumbnails_100x100/`.

`fig3_dark.png` is the same plot for dark backgrounds, used on the project page
and in slides. It is transparent rather than dark-papered, so it sits on any
background:

```bash
python python/visualization/plot_dataset_distribution.py --dark --output fig3_dark.png
```

`--dark` forces PNG, since JPEG has no alpha.
