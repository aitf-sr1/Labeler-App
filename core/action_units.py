"""
core/action_units.py — DEPRECATED

File ini dilestarikan hanya untuk backward compatibility.
Seluruh logika telah dipindahkan ke core/blendshape_features.py

Chain paper: Craig 2008 (AU→emosi) + Turrisi 2026 (BF→AU, κ=0.92) +
             Aldenhoven 2026 (native blendshapes validated).
"""
from .blendshape_features import (
    AU_NAMES,
    DEFAULT_AU_CALIB,
    compute_blendshape_features,
    compute_action_units,
    _raw_blendshape_signals as _raw_action_units,
)

__all__ = [
    "AU_NAMES",
    "DEFAULT_AU_CALIB",
    "compute_blendshape_features",
    "compute_action_units",
    "_raw_action_units",
]
