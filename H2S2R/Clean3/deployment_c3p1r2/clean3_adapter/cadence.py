"""Bilateral target/update cadence used by the isolated Clean3 environment."""

from __future__ import annotations


def set_bimanual_targets(controller, actions, side_action_dim: int = 11) -> None:
    """Set both side targets exactly once at a control-step boundary."""

    controller.right.set_targets(actions[:, :side_action_dim])
    controller.left.set_targets(actions[:, side_action_dim:])


def advance_bimanual(controller):
    """Advance both independent FABRICS graphs once in one physics substep."""

    return controller.right.advance(), controller.left.advance()

