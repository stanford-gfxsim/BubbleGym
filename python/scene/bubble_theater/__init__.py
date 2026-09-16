"""Scene scripts for procedural single-bubble shape sequences.

Computes frequency curves for a 90-frame OBJ sequence in which the mesh shape
varies but the enclosed volume is held fixed. For each frame the driver
computes the bubble resonance frequency under three models -- the spherical
Minnaert baseline, the ellipsoidal Strasberg baseline, and the chull-direct
neural-network surrogate -- writes a one-bubble trackedBubInfo.txt for each
model with position pinned at (0, 0, 0), and emits a ``freq_curves.csv`` plus
an overlay plot comparing the models against the BEM ground truth.

The trackedBubInfo files are the sound-synthesis input, but rendering them to
audio is FluidSound's job and is not driven from this repository.
"""
