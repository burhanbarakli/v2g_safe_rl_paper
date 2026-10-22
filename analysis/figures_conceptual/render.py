"""Render the hand-written conceptual SVGs to PDF (vector) and PNG (3x pixel density) with headless Chromium.

    python3 render.py [name.svg ...]        (default: all fig_*.svg in this folder)

Needs the `playwright` Python package and a Chromium build; set PW_CHROMIUM to the executable if it is
not found automatically (e.g. /opt/pw-browsers/chromium-1194/chrome-linux/chrome).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def _exe():
    if os.environ.get("PW_CHROMIUM"):
        return os.environ["PW_CHROMIUM"]
    for c in sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome")):
        return str(c)
    return None


def size(svg: str):
    m = re.search(r'<svg[^>]*\bwidth="([\d.]+)"[^>]*\bheight="([\d.]+)"', svg)
    return float(m.group(1)), float(m.group(2))


def render(files):
    with sync_playwright() as p:
        kw = {"executable_path": _exe()} if _exe() else {}
        b = p.chromium.launch(**kw)
        for f in files:
            svg = f.read_text()
            w, h = size(svg)
            html = (f"<!doctype html><html><head><style>@page {{ size: {w}px {h}px; margin: 0 }} "
                    f"html,body {{ margin:0; padding:0; background:#fff }} svg {{ display:block }}</style></head>"
                    f"<body>{svg}</body></html>")
            pg = b.new_page(viewport={"width": int(round(w)), "height": int(round(h))}, device_scale_factor=3)
            pg.set_content(html)
            pg.screenshot(path=str(f.with_suffix(".png")), full_page=False, omit_background=False)
            pg.pdf(path=str(f.with_suffix(".pdf")), width=f"{w}px", height=f"{h}px", print_background=True,
                   margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            pg.close()
            print("rendered", f.name, f"{w:.0f}x{h:.0f}px")
        b.close()


if __name__ == "__main__":
    names = [HERE / a for a in sys.argv[1:]] or sorted(HERE.glob("fig_*.svg"))
    render(names)
