# BubbleGym: A Practical Shape-to-Frequency Model for Acoustic Bubbles

**SIGGRAPH Asia 2026 Conference Papers**

![Fruit falling into a water tank, entraining a cloud of nonspherical bubbles](results/experiments/fig05_fruit_splash/fruits_final.jpg)

[Zhehao Li](https://zhehaoli1999.github.io/)<sup>1</sup> &nbsp;·&nbsp;
[Kui Wu](https://kuiwuchn.github.io/)<sup>2</sup> &nbsp;·&nbsp;
[Wei Li](https://lwkobe.github.io/)<sup>3</sup> &nbsp;·&nbsp;
[Doug L. James](https://graphics.stanford.edu/~djames/)<sup>1</sup>

<sup>1</sup>Stanford University &nbsp;&nbsp;
<sup>2</sup>Independent Researcher &nbsp;&nbsp;
<sup>3</sup>Shanghai Jiao Tong University

[Paper (author's version)](https://drive.google.com/file/d/1zbNWpLHeS4T3vRF14Bb3kKyYksr4LAya/view?usp=sharing) ·
[Project page](https://stanford-gfxsim.github.io/BubbleGym/)

## Abstract

Water sounds are dominated by the acoustic radiation of entrained air bubbles.
The resonant frequency of a bubble depends critically on its size, shape, and
proximity to nearby boundaries, and is governed by the capacitance of the bubble
in the exterior mixed-boundary Laplace problem. Existing bubble-frequency models
face a difficult trade-off: spherical approximations are efficient but inaccurate
for deformed bubbles common in fluid simulations (multiple semitones of pitch
error), whereas boundary element methods (BEM) accurately account for
nonspherical geometry but are prohibitively expensive. A practical method for
bubble frequency estimation is, therefore, highly desirable for the efficient
authoring of physics-based water sounds. In this paper, we present *BubbleGym*,
a framework for efficient, shape-aware bubble source modeling. BubbleGym
introduces a benchmark dataset of 10k nonspherical bubble meshes with
ground-truth frequencies. We first propose a moment-matching ellipsoidal proxy
model that improves efficiency but remains limited when handling strongly
deformed shapes, such as bent bubbles. We then adopt a learned, compact,
scale-invariant mapping from a curated set of bubble shape features to resonant
frequency. We observe that the resulting model runs up to 1428× faster than BEM
while maintaining a mean error of less than 1% (under 17 cents, below the
perceptual JND for transient sounds) on the test set. Integrated into a complete
water-sound synthesis pipeline, our method enables high-fidelity audio for
complex, bubble-rich scenes at a fraction of the cost of direct BEM evaluation.

## What is here

This repository covers the benchmark dataset and the shape-to-frequency models
of the paper. Each directory has its own README with the details.

| Directory | What it holds |
|---|---|
| [`dataset/`](dataset) | The 10k benchmark CSV and the four paper scenes, each with one `trackedBubInfo` per frequency model. The 10,000 meshes themselves are a [separate download](https://drive.google.com/file/d/1RaK-cJ7NlHzVyHUncHQMwzrxlvt204PB/view?usp=sharing). See [`dataset/README.md`](dataset/README.md) for the file format and how to install the meshes |
| [`python/`](python) | All code, in eight packages: the BEM solver, the eight shape descriptors, the models and ablations with trained weights, the format layer, and the per-scene drivers. See [`python/README.md`](python/README.md) |
| [`results/experiments/`](results/experiments) | Every figure and table, one folder each, holding the plot and the raw data behind it. See [`results/experiments/README.md`](results/experiments/README.md) |

## Getting started

**Install git-LFS first.** The scene files are tracked with it, and a clone
without it does not degrade into pointer stubs, it fails outright:

```
fatal: the remote end hung up unexpectedly
warning: Clone succeeded, but checkout failed.
```

```bash
git lfs install
git clone https://github.com/stanford-gfxsim/BubbleGym.git
cd BubbleGym
```

If you already have a broken tree, or want the code without the 620 MB of scene
data, skip the content and fetch it later:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/stanford-gfxsim/BubbleGym.git
cd BubbleGym && git lfs pull        # when you want the scenes
```

Then the environment:

```bash
conda create -n soundlab python=3.11 -y
conda activate soundlab

pip install numpy scipy pandas matplotlib   # enough to read the data and plot
pip install -r requirements.txt             # full stack: torch, bempp-cl, ...
```

Run everything from the repository root with `PYTHONPATH=python`.

**Predict frequencies for a scene.** Each scene ships one
`trackedBubInfo_<model>` per frequency model, all sharing one bubble graph and
differing only in the frequency column, so any two are directly comparable. The
three large scenes ship theirs xz-compressed as `.txt.xz`, `bubble_theater/` as
plain `.txt`. Every reader in this repository takes either form; `xz -dk
<file>.txt.xz` unpacks one for a tool that needs the raw text, such as the
FluidSound renderer. The three writers in `python/scene/_common/` take the same
shape, a base tracked file in and a frequency column out:

```bash
python python/scene/_common/write_trackedbubinfo_nn.py \
    --tracked <scene>/trackedBubInfo.txt --out <scene>/trackedBubInfo_NN.txt

python python/scene/_common/write_trackedbubinfo_bem.py \
    --tracked <scene>/trackedBubInfo_NN.txt --rows-csv <sweep>_all_rows.csv \
    --out <scene>/trackedBubInfo_BEM.txt --dt-lbm <seconds per LBM step>

python python/scene/_common/replace_trackedbubinfo_freq_with_minnaert.py \
    --src <scene>/trackedBubInfo_NN.txt --out <scene>/trackedBubInfo_Minnaert.txt
```

**For a single mesh** rather than a scene. Both are packages, so run them with
`-m`; invoking the files by path fails on their relative imports.

```bash
PYTHONPATH=python python -m freq_model.NN.nn_inference bubble.obj --radius-mm 5
PYTHONPATH=python python -m bem.compute_freq_bempp_galerkin bubble.obj
```

The first prints the surrogate's unit-volume frequency, and the physical
frequency too when given a radius. The second solves the BEM reference.

**Reproduce a figure or table.** Every trained model and ablation variant ships
with its weights, so nothing needs retraining.
[`results/experiments/README.md`](results/experiments/README.md) gives the
command and the data for each one, and records which paper numbers are
regenerable, which are not, and why.

## License

The code is released under MIT; see [`LICENSE`](LICENSE). The data, meaning the
benchmark CSV, thumbnails, scene files and the separately distributed mesh
archive, is released under CC BY 4.0; see [`dataset/LICENSE`](dataset/LICENSE).
The `VOF` bubbles derive from Langlois et al. 2016, so please cite that work as
well when you use them.

## Citation

```bibtex
@inproceedings{li2026bubblegym,
  author    = {Li, Zhehao and Wu, Kui and Li, Wei and James, Doug L.},
  title     = {BubbleGym: A Practical Shape-to-Frequency Model for Acoustic Bubbles},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers (SA Conference Papers '26)},
  year      = {2026},
  location  = {Kuala Lumpur, Malaysia},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3829340.3842193},
  isbn      = {979-8-4007-2842-6}
}
```
