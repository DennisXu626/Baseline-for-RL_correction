"""Fixed Clean3 hand-action ranges centered on the shared static posture."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


MARGIN_FRACTION = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrated_range(matrix, center, minimum, maximum, q0_hand) -> dict[str, np.ndarray]:
    """Apply the exact reviewer-frozen C3-P1R algebra in absolute Bq space."""

    matrix = np.asarray(matrix, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    q0_hand = np.asarray(q0_hand, dtype=np.float64)
    x0 = matrix @ q0_hand
    original_low = minimum + matrix @ center
    original_high = maximum + matrix @ center
    width = original_high - original_low
    runtime_low = np.minimum(original_low, x0 - MARGIN_FRACTION * width)
    runtime_high = np.maximum(original_high, x0 + MARGIN_FRACTION * width)
    if not np.all(runtime_low < x0) or not np.all(x0 < runtime_high):
        raise ValueError("calibrated range must place x0 strictly inside every dimension")
    return {
        "x0": x0,
        "original_low": original_low,
        "original_high": original_high,
        "runtime_low": runtime_low,
        "runtime_high": runtime_high,
        "width_ratio": (runtime_high - runtime_low) / width,
    }


def build_sidecar(*, right_pca: Path, left_pca: Path, q0_right, q0_left, output: Path) -> None:
    payload = {"margin_fraction": np.float64(MARGIN_FRACTION)}
    for side, path, q0 in (
        ("right", right_pca, q0_right),
        ("left", left_pca, q0_left),
    ):
        with np.load(path, allow_pickle=False) as data:
            spec = calibrated_range(
                data["matrix"], data["center"], data["minimum"], data["maximum"], q0
            )
        for name, value in spec.items():
            payload[f"{side}_{name}"] = value.astype(np.float64)
        payload[f"{side}_pca_sha256"] = np.asarray(_sha256(path))
    np.savez_compressed(output, **payload)


def install_side(controller, sidecar, side: str) -> None:
    """Wire the sidecar into the real controller fields used by set_targets."""

    import torch

    palm_center = controller.palm_target.detach().clone()
    tolerance = 1e-6
    if bool(((palm_center < controller.palm_minimum - tolerance) |
             (palm_center > controller.palm_maximum + tolerance)).any()):
        raise ValueError(f"{side} static palm FK is outside existing V12 palm bounds")
    device = controller.hand_target.device
    controller.palm_policy_center = palm_center
    controller.hand_policy_center = torch.as_tensor(
        sidecar[f"{side}_x0"], dtype=torch.float32, device=device
    ).expand(controller.num_envs, -1).clone()
    controller.hand_minimum = torch.as_tensor(
        sidecar[f"{side}_runtime_low"], dtype=torch.float32, device=device
    )
    controller.hand_maximum = torch.as_tensor(
        sidecar[f"{side}_runtime_high"], dtype=torch.float32, device=device
    )


def install_bimanual(controller, path: str | Path) -> None:
    """Install both sides once; the fixed centers survive controller resets."""

    with np.load(path, allow_pickle=False) as sidecar:
        install_side(controller.right, sidecar, "right")
        install_side(controller.left, sidecar, "left")
