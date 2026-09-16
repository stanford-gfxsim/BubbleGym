"""Replace every sample-line frequency with the Minnaert default.

For each Bub block in a trackedBubInfo-style file, parse the radius from
the ``Bub <id> <radius>`` header and overwrite the frequency column
(``parts[1]``) of every sample line in that block with
``MINNAERT_CONSTANT / radius``. Sample lines outside a block, the
``Start:``/``End:`` event lines, and every other column (time, position,
optional pressure) are passed through unchanged.

Default: rewrites
``dataset/exhalation/trackedBubInfo_NN.txt`` in place.

Examples:
    # Overwrite the dropped file in place (default).
    python python/scene/_common/replace_trackedbubinfo_freq_with_minnaert.py

    # Read one file, write to another.
    python python/scene/_common/replace_trackedbubinfo_freq_with_minnaert.py \\
        --src dataset/exhalation/trackedBubInfo_NN.txt \\
        --out dataset/exhalation/trackedBubInfo-2-dropped-minnaert.txt
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from freq_model.analytical.minnaert_freq import MINNAERT_CONSTANT  # noqa: E402

REPO_ROOT = PYTHON_ROOT.parent
DEFAULT_SRC = REPO_ROOT / "dataset" / "exhalation" / "trackedBubInfo-2-dropped.txt"


_BUB_HEADER_RE = re.compile(r"^\s*Bub\s+(\d+)\s+(\S+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path; defaults to overwriting --src in place.",
    )
    parser.add_argument("--frequency-format", type=str, default="{:.6f}")
    args = parser.parse_args()

    src = args.src.resolve()
    out = (args.out or args.src).resolve()
    fmt = args.frequency_format

    if not src.is_file():
        raise SystemExit(f"missing src: {src}")

    lines = src.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = list(lines)

    cur_bub: int | None = None
    cur_radius: float = float("nan")
    cur_freq_token: str = "nan"

    n_blocks = 0
    n_blocks_no_radius = 0
    n_samples_rewritten = 0
    n_samples_skipped_no_radius = 0
    n_samples_malformed = 0

    for i, ln in enumerate(lines):
        m = _BUB_HEADER_RE.match(ln)
        if m:
            n_blocks += 1
            cur_bub = int(m.group(1))
            try:
                cur_radius = float(m.group(2))
            except ValueError:
                cur_radius = float("nan")
            if math.isfinite(cur_radius) and cur_radius > 0.0:
                cur_freq_token = fmt.format(MINNAERT_CONSTANT / cur_radius)
            else:
                n_blocks_no_radius += 1
                cur_freq_token = "nan"
            continue

        s = ln.lstrip()
        if not s:
            continue
        if not s[0].isdigit():
            continue

        if cur_bub is None:
            continue

        if not (math.isfinite(cur_radius) and cur_radius > 0.0):
            n_samples_skipped_no_radius += 1
            continue

        prefix = ln[: len(ln) - len(ln.lstrip())]
        parts = ln.split()
        if len(parts) < 6:
            n_samples_malformed += 1
            continue
        parts[1] = cur_freq_token
        out_lines[i] = prefix + " ".join(parts)
        n_samples_rewritten += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print("Summary:")
    print(f"  src:                          {src}")
    print(f"  out:                          {out}")
    print(f"  Minnaert constant:            {MINNAERT_CONSTANT}")
    print(f"  Bub blocks scanned:           {n_blocks}")
    print(f"  Bub blocks lacking radius:    {n_blocks_no_radius}")
    print(f"  sample lines rewritten:       {n_samples_rewritten}")
    print(f"  sample lines (no radius):     {n_samples_skipped_no_radius}")
    print(f"  sample lines (malformed):     {n_samples_malformed}")


if __name__ == "__main__":
    main()
