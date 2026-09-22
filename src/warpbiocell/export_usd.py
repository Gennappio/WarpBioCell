"""Export a run directory's checkpoints to an OpenUSD stage (needs the ``usd`` extra).

    python -m warpbiocell.export_usd runs/<timestamp>_tumor_spheroid
    python -m warpbiocell.export_usd runs/<dir> --out spheroid.usda --color oxygen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="output .usdc (binary, default: <run_dir>/cells.usdc) or .usda (text)")
    parser.add_argument("--color", choices=["state", "oxygen"], default="state")
    args = parser.parse_args(argv)

    try:
        from warpbiocell.io.export import export_usd

        path = export_usd(args.run_dir, args.out, color_by=args.color)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
