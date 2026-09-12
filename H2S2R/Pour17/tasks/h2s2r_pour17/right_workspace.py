"""Right-side absolute workspace with normalized zero at the near-grasp reset.

The audited Run04 target limits remain unchanged.  Only the normalized action
coordinate is calibrated, independently from the left side.
"""
import hashlib
import json
from pathlib import Path
import torch

SPEC_PATH = Path(__file__).with_name('right_workspace_bounds.json')

def workspace_record():
    payload = SPEC_PATH.read_bytes()
    spec = json.loads(payload)
    return {**spec, 'sha256': hashlib.sha256(payload).hexdigest(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def apply_centered_bounds(controller):
    spec = workspace_record()
    buffers = (controller.palm_minimum, controller.palm_maximum,
               controller.hand_minimum, controller.hand_maximum)
    for current, expected, expanded in zip(buffers, spec['original'], spec['expanded']):
        before = torch.tensor(expected, device=current.device, dtype=current.dtype)
        after = torch.tensor(expanded, device=current.device, dtype=current.dtype)
        if not torch.allclose(current, before, atol=1e-6, rtol=0):
            raise ValueError('Right calibration requires the audited original controller limits')
        current.copy_(after)
    palm_center = torch.tensor(
        spec['policy_center'][0], device=controller.palm_target.device,
        dtype=controller.palm_target.dtype,
    )
    hand_center = torch.tensor(
        spec['policy_center'][1], device=controller.hand_target.device,
        dtype=controller.hand_target.dtype,
    )
    if not torch.allclose(controller.palm_target[0], palm_center, atol=1e-5, rtol=0):
        raise ValueError('Right palm center does not match the measured reset target')
    if not torch.allclose(controller.hand_target[0], hand_center, atol=1e-5, rtol=0):
        raise ValueError('Right hand center does not match the measured reset target')
    for center, lower, upper in (
        (palm_center, buffers[0], buffers[1]),
        (hand_center, buffers[2], buffers[3]),
    ):
        if not torch.all((center > lower) & (center < upper)):
            raise ValueError('Right policy center must be strictly inside its target bounds')
    controller.palm_policy_center = palm_center
    controller.hand_policy_center = hand_center
    controller.workspace_bounds_record = spec
