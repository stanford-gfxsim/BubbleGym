# Scenes and datasets

| Directory | What |
| --- | --- |
| `bubble_gym/` | the 10k benchmark: `dataset_bubblegym_10k.csv`, mesh thumbnails |
| `bubble_theater/` | the three procedural sequences of the teaser, `trackedBubInfo_{BEM,NN,Minnaert,Ellipsoid}.txt` and the per-frame meshes. The frequency curves they produce live in `results/experiments/fig01_bubble_theater/` |
| `single_rising_bubble/` | Fig. 7 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt`|
| `fruit_splash/` | Fig. 5 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt` |
| `exhalation/` | Fig. 6 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt` |

## The benchmark meshes

The 10,000 meshes behind `dataset_bubblegym_10k.csv` are **not in the
repository**. Download them from
[Google Drive](https://drive.google.com/file/d/1RaK-cJ7NlHzVyHUncHQMwzrxlvt204PB/view?usp=sharing),
unpack so the `VOF/` and `LBM/` folders sit side by side, and either place them
at `dataset/bubble_gym/meshes10k` or set `BUBBLEGYM_MESH_ROOT` to that folder.
The CSV's `mesh_id` column is the path of each mesh inside the archive.

Nothing needs the archive to reproduce the models or the figures: the CSV
already carries every descriptor and reference frequency, and a thumbnail of
every bubble ships under `bubble_gym/`. It is needed only to recompute features
or BEM frequencies from geometry, or to render new thumbnails.

## Provenance and licence

The benchmark draws on two sources, recorded per row in the CSV's `source`
column.

| `source` | Bubbles | Origin |
| --- | ---: | --- |
| `VOF` | 6,000 | Bubble meshes derived from the two-phase VOF simulations of Langlois, Zheng and James, *Toward Animating Water with Complex Acoustic Bubbles*, ACM Transactions on Graphics 35(4), Article 95, 2016. |
| `LBM` | 4,000 | Bubble meshes from the lattice Boltzmann simulations of this work. |

For both sources, the unit-volume rescaling, the shape descriptors and the BEM
reference frequencies were computed for this work.

Code, file paths and a few result columns name the `VOF` source
`langlois2016` (for example `LANGLOIS2016_MESH_ROOT`,
`dataset/langlois2016/`, `testset_mape_langlois2016`); it always means these
6,000 bubbles from Langlois et al. 2016.

**Licence.** The data in this directory is released under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
That covers the benchmark CSV, the thumbnails, the scene `trackedBubInfo`
files, and the separately distributed mesh archive. The code in the rest of the
repository remains under MIT; see the top-level `LICENSE`. If you use the data,
cite BubbleGym, and for the `VOF` bubbles also cite Langlois et al. 2016.

## trackedBubInfo format

trackedBubInfo.txt stores the bubble tracking information for the simulation. For each bubble, the format is:

    Bub <global, unique bubble ID> <radius>
      Start: <event type> <start time> <(optional) parent bubble ID(s)>
      <time> <frequency (Hz)> <x> <y> <z> <(optional) pressure> <(optional)depth>
      .
      .
      End: <event type> <end time> <optional child bubble IDs>
    
The start and end event types can be:
    N: new (entrain); start-only
    M: merge; the bubble IDs that this bubble merges to/from are listed.
    S: split; the bubble IDs that this bubble splits to/from are listed.
    C: collapse; end-only 