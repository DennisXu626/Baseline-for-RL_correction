"""Frozen Run05 bounds: extend only the five limits verified by short probes.

Keep absolute 11-D actions, the existing PCA basis and all FABRICS terms.
No random sampling here: model initialization uses the same RNG sequence as 03.
"""
import hashlib
import json
from pathlib import Path
import torch

SPEC_PATH = Path(__file__).with_name('left_workspace_bounds.json')

def workspace_record():
    payload = SPEC_PATH.read_bytes()
    spec = json.loads(payload)
    return {**spec, 'sha256': hashlib.sha256(payload).hexdigest(),
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

def apply_frozen_bounds(controller):
    spec = workspace_record()
    buffers = (controller.palm_minimum, controller.palm_maximum,
               controller.hand_minimum, controller.hand_maximum)
    for current, expected, expanded in zip(buffers, spec['original'], spec['expanded']):
        before = torch.tensor(expected, device=current.device, dtype=current.dtype)
        after = torch.tensor(expanded, device=current.device, dtype=current.dtype)
        if not torch.allclose(current, before, atol=1e-6, rtol=0):
            raise ValueError('Run05 bounds require the audited original controller limits')
        current.copy_(after)
    for value, lower, upper in ((controller.palm_target, buffers[0], buffers[1]),
                                 (controller.hand_target, buffers[2], buffers[3])):
        if not torch.all((value >= lower-1e-6) & (value <= upper+1e-6)):
            raise ValueError('Run05 initial target is not covered by its frozen bounds')
    controller.workspace_bounds_record = spec
