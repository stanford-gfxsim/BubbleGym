"""Bubble shape-to-frequency surrogate network.

:class:`BubbleFreqNet` is the shallow-and-wide MLP behind every BubbleFreq
trainer, evaluator and inference path. It is feature-agnostic: it maps an
``input_dim``-vector to one scalar and the caller decides what that scalar
means. The shipped model regresses ``log(f_BEM)`` directly; ``training.py``
also supports a log-residual convention on top of a physics baseline.

``input_dim`` is a caller argument, not a property of this module:
``fit_shape_freq_model.py`` uses the 8 shape features while the ablation study
instantiates the same class at several input widths.
"""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["BubbleFreqNet"]


class BubbleFreqNet(nn.Module):
    """Shallow-and-wide MLP with a skip connection.

    Architecture (``input_dim`` -> 1)::

        Input (input_dim)
          |
          +- [input_proj]  Linear(input_dim -> hidden_dim)  + LayerNorm + GELU + Dropout
          |                       | (skip)
          +- [block1]      Linear(hidden_dim -> hidden_dim) + LayerNorm + GELU + Dropout
          |                                                 <- + skip from input_proj
          +- [block2]      Linear(hidden_dim -> hidden_dim2) + LayerNorm + GELU + Dropout
          |
          +- [head]        Linear(hidden_dim2 -> 1)

    LayerNorm rather than BatchNorm because BatchNorm's variance estimate is
    unstable at these batch sizes. The head has no output activation.

    Naming trap: only ``block1_norm`` carries "norm" in its parameter names --
    the LayerNorms inside ``input_proj`` and ``block2`` are auto-named
    ``input_proj.1.*`` / ``block2.1.*``. See ``training.make_optimizer``.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 hidden_dim2: int = 32, dropout: float = 0.1):
        super().__init__()

        # Input projection (also the source of the skip connection).
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Block 1: consumes input_proj's output, then adds it back as a skip.
        self.block1_linear = nn.Linear(hidden_dim, hidden_dim)
        self.block1_norm = nn.LayerNorm(hidden_dim)
        self.block1_act = nn.GELU()
        self.block1_drop = nn.Dropout(dropout)

        # Block 2.
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim2),
            nn.LayerNorm(hidden_dim2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output head.
        self.head = nn.Linear(hidden_dim2, 1)

        self._init_weights()

    def _init_weights(self):
        """He initialization, appropriate for the GELU activations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection.
        h = self.input_proj(x)              # (B, hidden_dim)

        # Block 1 + skip connection.
        h2 = self.block1_linear(h)
        h2 = self.block1_norm(h2)
        h2 = self.block1_act(h2)
        h2 = self.block1_drop(h2)
        h2 = h2 + h                        # residual add

        # Block 2.
        h3 = self.block2(h2)               # (B, hidden_dim2)

        # Output.
        return self.head(h3)               # (B, 1)
