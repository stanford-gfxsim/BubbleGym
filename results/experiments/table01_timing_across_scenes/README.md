# Table 1 — cost across scenes

BEM against the learned model on every scene in the paper.

**No single script prints this table.** It is assembled from the ledgers here,
one group of cells per row.

| Row | Comes from |
| --- | --- |
| the three procedural scenes | `nn8_vs_bem_*.csv` here |
| Fruit Splash | `all_bubbles/fruit08/fruit08_all_summary.json` |
| Exhalation | `all_bubbles/exhale15/`, the 12-second run |
| Single Rising Bubble | [`fig07_single_rising_bubble/`](../fig07_single_rising_bubble), whose README accounts for that row cell by cell |
| the machine | `table1_target_machine.md` |

Three things that are easy to get wrong.

The inference column, `+0.0005` ms, is one constant across all rows, taken from
`supp_table02_width_ablation/width_timing.json`. It is not a per-scene
measurement, and it is **not** `batched_gpu_inference.json` here, which times a
whole scene in one batch and reads an order of magnitude lower.

`#Unique Bubbles` counts bubbles with at least an exported mesh while `#Samples` counts
BEM solves. 

The caption's percentages are `n_nn_only_rows / n_rows` from the two sweep
summaries, the share of meshes the 5,000-vertex cap leaves to the learned
model: 199/75,885 for Fruit Splash and 1,851/228,413 for Exhalation.
