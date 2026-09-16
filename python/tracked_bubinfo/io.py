"""Parse and rewrite WaveBlender-format ``trackedBubInfo`` files.

File layout::

    Bub <global bubble id> <equivalent-sphere radius>
      <t> <f> <x> <y> <z> <pressure>      <- "sample line"
      <t> <f> <x> <y> <z> <pressure>
      Start: <kind> <t> ...               <- optional event lines
      End:   <kind> <t> ...
    Bub <next id> <radius>
      ...

A *sample line* is any line whose first non-space character is a digit. Every
function here works on the raw ``list[str]`` of lines so a caller can rewrite
one column without disturbing the rest of the file byte-for-byte.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

__all__ = [
    "MISSING_POLICIES",
    "MISSING_POLICY_HELP",
    "format_freq",
    "iter_sample_line_indices",
    "parse_bub_header_radii",
    "parse_sample_y",
    "parse_trackedbubinfo_blocks",
    "resolve_missing_samples",
    "rewrite_freq_column",
    "rewrite_with_string_freqs",
    "summarize_freqs",
    "truncate_and_rewrite_blocks",
    "write_tracked_lines",
]


# ---------------------------------------------------------------------------
# What to do with a sample that has no valid frequency
# ---------------------------------------------------------------------------
#
# Every frequency writer -- NN, Minnaert, Strasberg, BEM, the regression
# baselines -- shares this one policy, so a tracked file means the same thing
# whichever estimator produced it. The usual cause of a missing frequency is a
# mesh the feature pipeline rejected, typically a non-closed surface: it
# encloses no volume, so it has no resonant frequency and is not a bubble.

MISSING_POLICIES = ("drop", "nan", "minus_one", "keep_original", "minnaert")

MISSING_POLICY_HELP = (
    "What to do with a sample line the estimator could not evaluate -- usually "
    "a mesh rejected by the feature pipeline, typically a non-closed surface, "
    "which is not a bubble and has no resonant frequency. "
    "'drop' (default) removes the sample line, and any Bub block left empty is "
    "removed with it, so such frames are never recorded as bubbles. "
    "'nan' / 'minus_one' write a sentinel the consumer must special-case. "
    "'keep_original' leaves the input frequency in place. "
    "'minnaert' substitutes MINNAERT_CONSTANT / r from the block radius -- "
    "convenient, but it invents a frequency for geometry the estimator could "
    "not evaluate, which is why it is not the default."
)

_BUB_ID_RE = re.compile(r"^\s*Bub\s+(\d+)")
_BUB_HEADER_RE = re.compile(r"^\s*Bub\s+(\d+)\s+([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)")
_BUB_ANY_RE = re.compile(r"^\s*Bub\s+\d+")
_END_LINE_RE = re.compile(r"^\s*End:\s+\S+\s+([+\-]?\d*\.?\d+(?:[eE][+\-]?\d+)?)")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def iter_sample_line_indices(lines: Sequence[str]) -> Iterator[int]:
    """Yield the index of every sample line (first non-space char is a digit)."""
    for i, ln in enumerate(lines):
        s = ln.lstrip()
        if not s:
            continue
        if s[0].isdigit():
            yield i


def parse_sample_y(ln: str) -> tuple[float, float, float]:
    """Return ``(t, y, 0.0)`` from a ``<t> <f> <x> <y> <z> <p>`` sample line.

    The third element is an unused placeholder kept for call-site compatibility.
    """
    parts = ln.split()
    return float(parts[0]), float(parts[3]), 0.0


def resolve_missing_samples(
    *,
    have_value: np.ndarray,
    values: Sequence[float],
    drop_mask: np.ndarray,
    policy: str,
    fmt: str,
    orig_tokens: Sequence[str] | None = None,
    bub_ids: Sequence[int] | None = None,
    radius_of: dict[int, float] | None = None,
    minnaert_constant: float | None = None,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Turn per-sample estimates + a missing-value policy into output tokens.

    The one place every frequency writer decides what a sample line says when
    the estimator had no answer for it. See :data:`MISSING_POLICIES`.

    Parameters
    ----------
    have_value:
        Boolean mask over sample lines; ``True`` where ``values`` holds a real
        estimate.
    values:
        Per-sample frequencies. Only read where ``have_value`` is True.
    drop_mask:
        Samples already slated for removal for an unrelated reason (past the
        time cutoff, above a frequency cap, ...). Returned extended, never
        shrunk, so a caller's existing drops always survive.
    policy:
        One of :data:`MISSING_POLICIES`.
    fmt:
        ``str.format`` spec for a frequency, e.g. ``"{:.6f}"``.
    orig_tokens:
        The input file's frequency tokens. Required for ``keep_original``.
    bub_ids, radius_of, minnaert_constant:
        Required for ``minnaert``: the per-sample bubble id, the
        ``{bub_id: radius}`` map from the ``Bub`` headers, and the constant, so
        the fill is ``minnaert_constant / radius``.

    Returns
    -------
    ``(freq_strings, drop_mask, stats)`` where ``stats`` counts
    ``filled`` (real estimates), ``dropped`` (added to the mask here),
    ``substituted`` (given a policy value) and ``no_radius`` (wanted Minnaert
    but the block had no usable radius, so fell back to ``nan``).

    Notes
    -----
    Under ``drop`` the returned strings for dropped samples are ``"nan"``
    placeholders; :func:`truncate_and_rewrite_blocks` never reads them, but they
    keep the list index-aligned with ``sample_idxs``.
    """
    if policy not in MISSING_POLICIES:
        raise ValueError(
            f"unsupported missing-sample policy {policy!r}; "
            f"expected one of {MISSING_POLICIES}"
        )
    have_value = np.asarray(have_value, dtype=bool)
    drop_mask = np.asarray(drop_mask, dtype=bool).copy()
    n = len(have_value)
    if len(drop_mask) != n:
        raise ValueError("have_value and drop_mask must be the same length")
    if policy == "keep_original" and orig_tokens is None:
        raise ValueError("policy 'keep_original' needs orig_tokens")
    if policy == "minnaert" and (
        bub_ids is None or radius_of is None or minnaert_constant is None
    ):
        raise ValueError(
            "policy 'minnaert' needs bub_ids, radius_of and minnaert_constant"
        )

    stats = {"filled": 0, "dropped": 0, "substituted": 0, "no_radius": 0}
    out: list[str] = ["nan"] * n

    for i in range(n):
        if bool(drop_mask[i]):
            continue
        if bool(have_value[i]):
            out[i] = format_freq(float(values[i]), fmt)
            stats["filled"] += 1
            continue

        # No estimate for this sample -- apply the policy.
        if policy == "drop":
            drop_mask[i] = True
            stats["dropped"] += 1
        elif policy == "nan":
            out[i] = "nan"
            stats["substituted"] += 1
        elif policy == "minus_one":
            out[i] = fmt.format(-1.0)
            stats["substituted"] += 1
        elif policy == "keep_original":
            out[i] = orig_tokens[i]
            stats["substituted"] += 1
        elif policy == "minnaert":
            r = float(radius_of.get(int(bub_ids[i]), float("nan")))
            if math.isfinite(r) and r > 0.0:
                out[i] = fmt.format(float(minnaert_constant) / r)
                stats["substituted"] += 1
            else:
                out[i] = "nan"
                stats["no_radius"] += 1

    return out, drop_mask, stats


def parse_trackedbubinfo_blocks(
    src: Path,
) -> tuple[list[str], list[int], list[tuple[int, float, float]]]:
    """Read a tracked file and index its sample lines.

    Returns ``(lines, sample_line_indices, per_sample_meta)`` where
    ``per_sample_meta[i] = (bub_id, t_s, y_m)`` for the ``i``-th sample line.
    """
    lines = Path(src).read_text(encoding="utf-8").splitlines()
    sample_idxs: list[int] = []
    meta: list[tuple[int, float, float]] = []
    cur_bub: int | None = None
    for i, ln in enumerate(lines):
        m = _BUB_ID_RE.match(ln)
        if m:
            cur_bub = int(m.group(1))
            continue
        s = ln.lstrip()
        if not s:
            continue
        if s[0].isdigit():
            if cur_bub is None:
                raise ValueError(
                    f"Sample line at {i} appears before any Bub block: {ln!r}"
                )
            t_s, y_m, _ = parse_sample_y(ln)
            sample_idxs.append(i)
            meta.append((cur_bub, t_s, y_m))
    return lines, sample_idxs, meta


def parse_bub_header_radii(lines: Sequence[str]) -> dict[int, float]:
    """Return ``{bub_id: radius}`` from every ``Bub <id> <r>`` header line."""
    radii: dict[int, float] = {}
    for ln in lines:
        m = _BUB_HEADER_RE.match(ln)
        if not m:
            continue
        bub_id = int(m.group(1))
        try:
            r = float(m.group(2))
        except ValueError:
            continue
        radii[bub_id] = r
    return radii


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_tracked_lines(dst: Path, lines: Sequence[str]) -> None:
    """Write tracked-file lines to ``dst``, one per line, always LF-terminated.

    ``Path.write_text`` applies the platform's newline translation, so the same
    filter run on Windows would emit CRLF and rewrite every byte of a versioned
    dataset artifact. These files are diffed and committed, so the ending is
    pinned rather than inherited from whoever ran the pipeline.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def format_freq(value: float, fmt: str) -> str:
    """Format one frequency, emitting literal ``nan``/``inf`` tokens.

    ``"{:.6f}".format(float("nan"))`` yields ``"nan"`` on CPython but the
    behaviour is not guaranteed across formats; downstream parsers expect the
    bare token, so non-finite values bypass ``fmt`` entirely.
    """
    if not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    return fmt.format(value)


def rewrite_with_string_freqs(
    src_lines: Sequence[str],
    sample_idxs: Sequence[int],
    freq_strings: Sequence[str],
    dst: Path,
) -> None:
    """Replace the frequency column with pre-formatted tokens and write ``dst``."""
    if len(sample_idxs) != len(freq_strings):
        raise ValueError(
            f"sample count {len(sample_idxs)} != freq count {len(freq_strings)}"
        )
    out = list(src_lines)
    for j, li in enumerate(sample_idxs):
        ln = out[li]
        prefix = ln[: len(ln) - len(ln.lstrip())]
        parts = ln.split()
        if len(parts) < 6:
            raise ValueError(f"unexpected sample line at {li}: {ln!r}")
        parts[1] = freq_strings[j]
        out[li] = prefix + " ".join(parts)
    write_tracked_lines(Path(dst), out)


def rewrite_freq_column(
    src_lines: Sequence[str],
    sample_idxs: Sequence[int],
    freqs: Sequence[float],
    dst: Path,
    fmt: str,
) -> None:
    """Replace the frequency column with ``fmt``-formatted floats and write ``dst``."""
    if len(sample_idxs) != len(freqs):
        raise ValueError(f"sample count {len(sample_idxs)} != freq count {len(freqs)}")
    rewrite_with_string_freqs(
        src_lines,
        sample_idxs,
        [fmt.format(float(x)) for x in freqs],
        dst,
    )


def _identify_bub_block_spans(src_lines: Sequence[str]) -> list[tuple[int, int]]:
    """Return ``[(start, end_exclusive)]`` for every ``Bub <id>`` block.

    A block runs from its ``Bub`` line up to (but not including) the next
    ``Bub`` line, or EOF for the last block.
    """
    starts = [i for i, ln in enumerate(src_lines) if _BUB_ANY_RE.match(ln)]
    if not starts:
        return []
    spans: list[tuple[int, int]] = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(src_lines)
        spans.append((s, e))
    return spans


def truncate_and_rewrite_blocks(
    src_lines: Sequence[str],
    sample_idxs: Sequence[int],
    drop_sample_mask: np.ndarray,
    new_freq_strings: Sequence[str],
    t_max: float,
) -> tuple[list[str], int, int]:
    """Rewrite frequencies and drop everything past a global cutoff time.

    Parameters
    ----------
    src_lines:
        Original lines from the input file (no trailing newline).
    sample_idxs:
        Line indices of every sample line, in the order returned by
        :func:`parse_trackedbubinfo_blocks`.
    drop_sample_mask:
        Boolean mask, same length as ``sample_idxs``; ``True`` means the sample
        line is past the cutoff and must be dropped. A ``Bub`` block whose
        samples are all dropped is removed entirely.
    new_freq_strings:
        Replacement frequency tokens, same length as ``sample_idxs``. Only read
        for surviving samples.
    t_max:
        Global cutoff time. ``End:`` lines whose event time exceeds ``t_max``
        are dropped, since the bubble is still alive at the cutoff.

    Returns
    -------
    ``(out_lines, n_blocks_dropped, n_samples_dropped)``
    """
    line_to_sample_pos: dict[int, int] = {li: pos for pos, li in enumerate(sample_idxs)}

    spans = _identify_bub_block_spans(src_lines)
    out_lines: list[str] = []
    n_blocks_dropped = 0
    n_samples_dropped = 0

    # Lines outside any Bub block (rare; e.g. a leading blank line) pass through.
    first_block_start = spans[0][0] if spans else len(src_lines)
    out_lines.extend(src_lines[:first_block_start])

    for s, e in spans:
        block = src_lines[s:e]
        survive_sample_pos: list[int] = []
        for offset in range(len(block)):
            pos = line_to_sample_pos.get(s + offset)
            if pos is None:
                continue
            if not bool(drop_sample_mask[pos]):
                survive_sample_pos.append(pos)

        if not survive_sample_pos:
            # Drop the entire block.
            n_blocks_dropped += 1
            n_samples_dropped += sum(
                1 for offset in range(len(block)) if (s + offset) in line_to_sample_pos
            )
            continue

        n_samples_dropped += sum(
            1
            for offset in range(len(block))
            if (s + offset) in line_to_sample_pos
            and bool(drop_sample_mask[line_to_sample_pos[s + offset]])
        )

        for offset, ln in enumerate(block):
            pos = line_to_sample_pos.get(s + offset)
            if pos is not None:
                if bool(drop_sample_mask[pos]):
                    continue
                prefix = ln[: len(ln) - len(ln.lstrip())]
                parts = ln.split()
                if len(parts) < 6:
                    out_lines.append(ln)
                    continue
                parts[1] = new_freq_strings[pos]
                out_lines.append(prefix + " ".join(parts))
                continue

            # Non-sample line: honour the End: cutoff, otherwise pass through.
            m_end = _END_LINE_RE.match(ln)
            if m_end:
                try:
                    end_t = float(m_end.group(1))
                except ValueError:
                    end_t = float("nan")
                if math.isfinite(end_t) and end_t > t_max:
                    continue
            out_lines.append(ln)

    return out_lines, n_blocks_dropped, n_samples_dropped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize_freqs(name: str, values: Iterable[float], width: int = 22) -> str:
    """One-line min/median/max summary of a frequency column, ignoring NaNs."""
    a = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return f"  {name:<{width}} (no finite values)"
    return (
        f"  {name:<{width}} n={a.size:>6}  min={a.min():9.3f}"
        f"  median={float(np.median(a)):9.3f}  max={a.max():9.3f}  Hz"
    )
