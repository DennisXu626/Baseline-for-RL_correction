"""CPU checks for the independent Run08 action-coordinate calibrations."""

import json
from pathlib import Path

import torch

from rl_rebuild.baselines.h2s2r.controller import rescale_from_policy_centered


HERE = Path(__file__).resolve().parent


def _check_spec(name: str) -> None:
    spec = json.loads((HERE / f"{name}_workspace_bounds.json").read_text())
    for target_index, action_dim in ((0, 6), (1, 5)):
        minimum = torch.tensor(spec["expanded"][2 * target_index])
        maximum = torch.tensor(spec["expanded"][2 * target_index + 1])
        center = torch.tensor(spec["policy_center"][target_index])
        policy = torch.stack(
            (-torch.ones(action_dim), torch.zeros(action_dim), torch.ones(action_dim))
        )
        decoded = rescale_from_policy_centered(policy, minimum, center, maximum)
        assert torch.allclose(decoded[0], minimum)
        assert torch.allclose(decoded[1], center)
        assert torch.allclose(decoded[2], maximum)


def test_left_centered_workspace() -> None:
    _check_spec("left")


def test_right_centered_workspace() -> None:
    _check_spec("right")


def test_side_centers_are_independent() -> None:
    left = json.loads((HERE / "left_workspace_bounds.json").read_text())
    right = json.loads((HERE / "right_workspace_bounds.json").read_text())
    assert left["controlled_side"] == "left"
    assert right["controlled_side"] == "right"
    assert left["policy_center"] != right["policy_center"]
