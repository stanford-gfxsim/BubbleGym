# Width-ablation inference timing — how to run it on another machine

This directory holds the five trained MLP width variants (`results/experiments/supp_table02_width_ablation/`)
and a benchmark that times them under the **same protocol as Table 1 of the main
paper**: batch 512 on the GPU, TorchScript forward pass, 30 warmup iterations and
150 timed repeats.

The output feeds two places in the paper:

- **Supplement, Table 2** — the `Inference (µs/bubble)` column.
- **Supplement, Fig. 5** — the per-bubble bar chart.

It is also the number that has to agree with **Table 1 of the main paper**, whose
caption quotes the production width `h64_h32` twice: 0.3721 µs/bubble for the
forward pass alone and 0.5155 µs/bubble for the whole path including transfer,
both at batch 512.

## What you need

- A CUDA GPU and a PyTorch build that sees it (`torch.cuda.is_available()`).
- Nothing else. No dataset, no training. The checkpoints ship with the repo, and
  the input batch is random — **timing does not depend on the weights or the
  data**, only on the network shape. The trained checkpoints are loaded anyway so
  the measured graph is exactly the released one.

## Run it

From the repository root:

```bash
python -m python.freq_model.ablation_study.benchmark_width_timing
```

Takes about a minute. Defaults are batch 512, 7 interleaved passes, 30 warmup and
150 repeats. Results are written to `results/experiments/supp_table02_width_ablation/width_timing.json`.

To write somewhere else, or to change the protocol:

```bash
python -m python.freq_model.ablation_study.benchmark_width_timing \
    --batch 512 --passes 7 --out results/experiments/width_timing_<machine>.json
```

## What to send back

**Just the JSON file.** It records the GPU, CPU, OS, PyTorch and CUDA versions,
and driver version alongside the numbers, so it identifies its own machine — no
need to write down what you ran it on. It also stores the per-pass values, not
only the means, so the scatter can be re-checked later.

If you want to paste something into a message instead, the console table is
enough; it has the same means and standard deviations.

## Reading the output

The shipped `results/experiments/supp_table02_width_ablation/width_timing.json` (Intel i9-14900K + RTX 5090 — the Sec. 6
machine, recorded in the file's `environment` block) reads:

```
    variant   #params |     forward us/bubble |   full path us/bubble
     h16_h8       641 |      0.3549 +/- 0.0043 |     0.5383 +/- 0.0417
    h32_h16     2,049 |      0.3941 +/- 0.0878 |     0.5208 +/- 0.0311
    h64_h32     7,169 |      0.3721 +/- 0.0451 |     0.5155 +/- 0.0390
   h128_h64    26,625 |      0.3655 +/- 0.0082 |     0.5134 +/- 0.0367
  h256_h128   102,401 |      0.4001 +/- 0.0791 |     0.5289 +/- 0.0359
```

- `forward us/bubble` is the column that goes into supplement Table 2, and the
  figure comparable to the main paper's 0.3721 µs.
- `full path us/bubble` is comparable to the main paper's 0.5155 µs.
- Widest vs narrowest (h256_h128 vs h16_h8, 160x the parameters) is
  0.4001 vs 0.3549 µs — +0.045 µs, against a pooled run-to-run scatter of
  0.056 µs. That is the claim the supplement makes in prose: no width penalty at
  this batch size. If the script prints **`WIDTH PENALTY`** on your machine, the
  supplement text is wrong there and needs changing — please report it either way.

## Two things that will bite you if changed

**Do not time the variants in blocks.** Run-to-run scatter on a warm GPU is
easily 10–20%, which is *larger* than the spread across the five widths. If you
time all passes of `h16`, then all passes of `h32`, and so on, a slow period
lands entirely on one variant and manufactures a trend. Measuring one pass of
every variant and then repeating spreads any drift across all of them. This is
not hypothetical: timing in blocks on the development machine produced an
apparent 44% penalty for `h256` that vanished to 5% once interleaved.

**Do not switch to eager mode.** The production pipeline exports to TorchScript,
and these networks are small enough that Python dispatch overhead dominates the
actual kernel time — eager measurements read roughly 20% high.

## Why the batch size is pinned

The call is *dispatch-bound*: its wall-clock cost is essentially independent of
how many bubbles are in the batch, so any per-bubble figure is a fixed per-call
cost divided by the batch and is only meaningful next to the batch it was
measured at. Pin it to 512 to stay comparable with Table 1. For the same reason
the **CPU** moves the result as much as the GPU does; the JSON records both, so
a run on other silicon identifies itself.
