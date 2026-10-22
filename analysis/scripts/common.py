"""Shared definitions for the analysis pipeline.

Watermarks, constants and small helpers. The method-name mapping is in names.py
(re-exported here for convenience).
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

SYNTH_MARK = "SYNTHETIC TEST DATA – NOT RESULTS"

from names import *  # noqa: F401,F403  (method registry lives in names.py)
from names import OI, MethodInfo, METHODS, info, label, method_order, conf_alpha  # noqa: F401

SMOKE_MARK = "SMOKE TEST OUTPUT \u2013 NOT RESULTS"
TRAIN_SIZES = [10, 20]
TEST_SIZES = [5, 10, 20, 50, 100, 200]
TIGHTNESS = {4.0: "tight", 7.0: "moderate"}


def tightness(rho: float) -> str:
    return TIGHTNESS.get(float(rho), f"rho{rho:g}")


def lighten(hex_color: str, amount: float = 0.75) -> str:
    """Blend with white (opaque colour, so EPS/PS output needs no transparency)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(round(c + (255 - c) * amount)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


_ACTIVE_MARK = [SYNTH_MARK]


def set_mark(mark: str):
    """Watermark used for synthetic or smoke outputs (set by make_all.run)."""
    _ACTIVE_MARK[0] = mark


def active_mark() -> str:
    return _ACTIVE_MARK[0]
