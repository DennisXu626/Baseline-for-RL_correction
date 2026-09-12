"""Install and verify the frozen right-hand zero-margin FABRICS patch."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path


EXPECTED_PATCH_SHA256 = "360d0955ab99c18a5694d1161c6d6ab6ec56c3dcf9ae3461692501af3e24f653"
_MODULE_NAME = "h2s2r_training_pilot_zero_margin_patch"
_PATCH_MODULE = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_right_training_patch() -> tuple[type, dict[str, object]]:
    """Install the exact P1 patch before importing the environment/controller."""

    global _PATCH_MODULE
    root = Path(__file__).resolve().parents[2]
    patch_path = (
        root / "runtime/cadence_20260912_deploy/sharpa_right_zero_margin_limits.py"
    ).resolve()
    actual_sha = _sha256(patch_path)
    if actual_sha != EXPECTED_PATCH_SHA256:
        raise RuntimeError(
            f"right zero-margin patch SHA contradiction: {actual_sha}"
        )
    if _PATCH_MODULE is None:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, patch_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("right zero-margin patch could not be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        module.install_right_only_zero_margin_patch()
        _PATCH_MODULE = module
    expected_class = _PATCH_MODULE.SharpaRightZeroMarginPoseFabric
    return expected_class, {
        "path": str(patch_path),
        "sha256": actual_sha,
        "module": _MODULE_NAME,
        "class_name": expected_class.__name__,
        "qualified_class_name": f"{expected_class.__module__}.{expected_class.__name__}",
    }


def verify_installed_right_fabric(fabric: object, expected_class: type) -> None:
    """Reject subclasses or a same-named class from a different module."""

    if type(fabric) is not expected_class:
        raise RuntimeError(
            "right controller did not construct the exact zero-margin patch class"
        )


def _verify_source_file(path: Path, expected: str) -> dict:
    path = Path(path).resolve()
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"limit dispatch source identity mismatch: {path}: {actual}")
    return {"path": str(path), "sha256": actual}


def _dispatch_sources(expected_class: type) -> dict:
    """Read actual method owners, not class names or an extra GPU reference fabric."""
    from fabrics_sim.fabrics.kuka_allegro_pose_fabric import KukaAllegroPoseFabric
    from rl_rebuild.baselines.h2s2r import contract

    root = Path(__file__).resolve().parents[2]
    patch = root / "runtime/cadence_20260912_deploy/sharpa_right_zero_margin_limits.py"
    actual_patch = Path(inspect.getsourcefile(expected_class.add_joint_limit_repulsion)).resolve()
    if actual_patch != patch.resolve():
        raise RuntimeError("limit patch method resolves outside frozen deployment")
    adapter = expected_class.__bases__[0]
    if adapter.__bases__ != (KukaAllegroPoseFabric,):
        raise RuntimeError("limit patch inheritance differs from frozen adapter")
    sources = {
        "patch": (actual_patch, EXPECTED_PATCH_SHA256),
        "base": (inspect.getsourcefile(KukaAllegroPoseFabric.add_joint_limit_repulsion),
                 "86a42b4d94e883e847094b924077a6c05d9de24fa63268c512f0ee2435749c85"),
        "adapter": (inspect.getsourcefile(adapter),
                    "d81874853e8b123235137d125565d66b5c6b569a14fd0012fd5c762ad47c9247"),
        "contract": (contract.__file__, "03ba63ae0e618e93c53a826c44695f0d1cd537417cb5900786ece190005f4afb"),
        "controller": (root / "rl_rebuild/baselines/h2s2r/controller.py",
                       "65ed5d0c3ff5d13aec6914d59cf1092864a33f307d05f52e7480126a13626d4a"),
        "params": (root / "rl_rebuild/baselines/h2s2r/fabric_params.py",
                   "9296ebdff5cc6755bf7e3f716cf44dcba4b59738c7a494a2246c70dc36fe2731"),
    }
    return {key: _verify_source_file(Path(path), sha) for key, (path, sha) in sources.items()}


def verify_limit_dispatch(right: object, left: object, expected_class: type) -> dict:
    """Read all environments' actual maps; preserve historical left contraction."""
    import numpy as np
    import xml.etree.ElementTree as ET
    from rl_rebuild.baselines.h2s2r.contract import ControlledSide, joint_names_for_side

    verify_installed_right_fabric(right, expected_class)
    if type(left).add_joint_limit_repulsion is not expected_class.add_joint_limit_repulsion:
        raise RuntimeError('left actual limit method differs from frozen delegating implementation')
    sources = _dispatch_sources(expected_class)
    records = {}
    for side, fabric, urdf_sha in (
        (ControlledSide.RIGHT, right, "181272687b5d38fc0177304f74aa0a195036f3831dd30b434e27b094c7f69e6e"),
        (ControlledSide.LEFT, left, "e3bf25fe1316491cc919ce68586893c2ebfc82cf5cebd62b3d57d2761dc2287d"),
    ):
        names = tuple(joint_names_for_side(side))
        if tuple(fabric.joint_names) != names or len(names) != 29:
            raise RuntimeError(f"{side.value}: joint order contradiction")
        source = _verify_source_file(Path(fabric.urdf_path), urdf_sha)
        joints = [j for j in ET.parse(fabric.urdf_path).getroot().findall('joint')
                  if j.attrib['type'] == 'revolute']
        if tuple(j.attrib['name'] for j in joints) != names:
            raise RuntimeError(f"{side.value}: URDF joint order contradiction")
        record = {"urdf": source, "joint_names": list(names), "limits": {}}
        for which, sign in (("lower", 1), ("upper", -1)):
            # Same Python/NumPy source arithmetic as the frozen assembly, then dtype cast.
            expected = np.asarray([float(j.find('limit').attrib[which]) + (
                sign * np.deg2rad(10.) if side == ControlledSide.LEFT or i < 7 else 0.)
                for i, j in enumerate(joints)])
            tensor = getattr(fabric.get_taskmap(f'{which}_joint_limit'),
                             f'{which}_joint_limits_batch')
            actual = tensor.detach().cpu().numpy()
            if actual.shape != (fabric.batch_size, 29) or actual.dtype.kind != 'f':
                raise RuntimeError(f"{side.value}/{which}: limit shape or dtype contradiction")
            expected = np.broadcast_to(expected.astype(actual.dtype), actual.shape)
            if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
                mismatches = np.argwhere(actual != expected)[:8].tolist()
                raise RuntimeError(f"{side.value}/{which}: actual limit assembly mismatch at {mismatches}; "
                                   f"actual={[float(actual[tuple(i)]) for i in mismatches]}; "
                                   f"expected={[float(expected[tuple(i)]) for i in mismatches]}")
            record['limits'][which] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                                      "all_envs_exact": True, "expected_row": expected[0].tolist()}
        records[side.value] = record
    return {"right_patch_exact_class": True, "left_limit_assembly_unchanged": True,
            "right_limit_assembly_correct": True, "sources": sources, "readback": records,
            "left_narrow_intervals": "preserved original; not repaired or required nonempty"}
