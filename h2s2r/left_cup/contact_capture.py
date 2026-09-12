"""Pure support for six-array PhysX contact snapshots; imports no Isaac modules."""
from __future__ import annotations
import numpy as np


def _copy(value):
    """Break GPU/CPU tensor aliases before a later simulation refresh."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def unpack_six(result, *, dt: float, expected_pairs: int = 1):
    if not isinstance(result, (tuple, list)) or len(result) != 6:
        raise ValueError(f"get_contact_data(dt) must return six arrays, got {type(result)} len={getattr(result, '__len__', lambda: '?')()}")
    force, point, normal, separation, count, start = map(_copy, result)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"physics dt must be positive finite, got {dt}")
    count = count.astype(np.int64, copy=False).reshape(-1)
    start = start.astype(np.int64, copy=False).reshape(-1)
    if len(count) != expected_pairs or len(start) != expected_pairs:
        raise ValueError(f"contact pair count mismatch: count={count.shape} start={start.shape} expected={expected_pairs}")
    capacity = int(force.shape[0])
    if point.shape[0] != capacity or normal.shape[0] != capacity or separation.shape[0] != capacity:
        raise ValueError("six-array contact buffers do not share capacity")
    if force.ndim != 2 or force.shape[1] != 1:
        raise ValueError(f"normal force must have interface shape (N,1), got {force.shape}")
    if point.ndim != 2 or point.shape[1] != 3 or normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError(f"point/normal must have interface shape (N,3), got {point.shape}/{normal.shape}")
    if separation.ndim != 2 or separation.shape[1] != 1:
        raise ValueError(f"separation must have interface shape (N,1), got {separation.shape}")
    if (count < 0).any() or (start < 0).any() or (start + count > capacity).any():
        raise OverflowError(f"invalid contact ranges count={count.tolist()} start={start.tolist()} capacity={capacity}")
    ranges = []
    for pair, (n, begin) in enumerate(zip(count, start)):
        end = int(begin + n)
        slices = [x[int(begin):end].copy() for x in (force, point, normal, separation)]
        if any(not np.isfinite(x).all() for x in slices):
            raise ValueError(f"non-finite value in valid contact range for pair {pair}")
        ranges.append(dict(pair=pair, start=int(begin), count=int(n), end=end,
                           force=slices[0], point=slices[1], normal=slices[2], separation=slices[3]))
    # Equality cannot prove overflow, but it is a fail-closed capacity warning.
    capacity_hit = bool(sum(int(x) for x in count) >= capacity and capacity > 0)
    return dict(dt=float(dt), capacity=capacity, count=count.copy(), start=start.copy(),
                ranges=ranges, capacity_hit=capacity_hit)


def float_close(actual, expected, *, dtype=np.float32, ulps=4):
    a=np.asarray(actual,dtype=dtype);e=np.asarray(expected,dtype=dtype)
    scale=np.maximum(np.maximum(np.abs(a),np.abs(e)),np.array(1,dtype=dtype))
    tol=np.finfo(dtype).eps*ulps*scale
    return bool(np.all(np.abs(a-e)<=tol)),float(np.max(tol))
