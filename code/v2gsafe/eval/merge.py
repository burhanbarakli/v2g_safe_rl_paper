"""Merge per-configuration part files of every run directory into episodes.csv (and step logs).

    python -m v2gsafe.eval.merge --raw results/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import merge_parts


def merge_all(raw: str | Path) -> list[Path]:
    out = []
    for d in sorted(Path(raw).glob("*")):
        if (d / "parts").is_dir():
            out.append(merge_parts(d))
    return out


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    a = ap.parse_args(argv)
    for p in merge_all(a.raw):
        print(f"merged {p}")


if __name__ == "__main__":
    _main()
