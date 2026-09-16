# Scenes and datasets

| Directory | What |
| --- | --- |
| `bubble_gym/` | the 10k benchmark: `dataset_bubblegym_10k.csv`, mesh thumbnails |
| `bubble_theater/` | the three procedural sequences of the teaser, `trackedBubInfo_{BEM,NN,Minnaert,Ellipsoid}.txt` and the per-frame meshes. The frequency curves they produce live in `results/experiments/fig01_bubble_theater/` |
| `single_rising_bubble/` | Fig. 7 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt`|
| `fruit_splash/` | Fig. 5 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt` |
| `exhalation/` | Fig. 6 scene, `trackedBubInfo_{BEM,NN,Minnaert}.txt` |

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