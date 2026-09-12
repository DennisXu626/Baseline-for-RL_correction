"""Install the frozen C3-P1R hand range and R2 palm contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from c3p1r.action_calibration import install_side as _install_r_side


LEFT_PALM_Z_MINIMUM_M = -0.03


def install_side(controller, sidecar, side: str) -> None:
    """Install R ranges, changing only Clean3 left palm z minimum."""

    if side not in ("right", "left"):
        raise ValueError(f"unknown side: {side}")
    if side == "left":
        palm_minimum = controller.palm_minimum.clone()
        palm_minimum[2] = LEFT_PALM_Z_MINIMUM_M
        controller.palm_minimum = palm_minimum
    _install_r_side(controller, sidecar, side)


def install_bimanual(controller, path: str | Path) -> None:
    """Install the immutable R2 calibration on both real controllers."""

    with np.load(path, allow_pickle=False) as sidecar:
        install_side(controller.right, sidecar, "right")
        install_side(controller.left, sidecar, "left")

