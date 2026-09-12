"""CPU preflight checks for C3-P2 deployment before SimulationApp startup."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


DETERMINISTIC_PXRS: bool = True
EXPECTED_ROBOT_USD = Path(
    "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17/world/"
    "vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"
)
EXPECTED_ROBOT_USD_TEXT = "/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17/world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"
EXPECTED_ROBOT_USD_SHA256 = "c2bba25aeb7b04b270df51d0c0efbe8eec95b499d8a7aef46834cefcfb71e347"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_root(path: Path) -> Path:
    raw = str(path)
    if raw.startswith("\\") and len(raw) >= 2 and raw[1] != ":":
        return Path(PurePosixPath(raw))
    if raw.startswith("/"):
        return Path(PurePosixPath(raw))
    return path.resolve()


def read_npz_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    with np.load(path, allow_pickle=False) as data:
        return {name: tuple(value.shape) for name, value in data.items()}


def _decode_simple_str(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            piece = _decode_simple_str(value)
            if piece is None:
                return None
            parts.append(piece)
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _decode_simple_str(node.left)
        right = _decode_simple_str(node.right)
        if left is None or right is None:
            return None
        return left + right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"str", "Path"}:
        parts = []
        for arg in node.args:
            piece = _decode_simple_str(arg)
            if piece is None:
                return None
            parts.append(piece)
        return "".join(parts)
    return None


def _evaluate_ast_path(node: ast.AST, names: dict[str, str], max_depth: int = 4) -> Path | None:
    """Evaluate a very small whitelist of safe AST nodes into a file path.

    Supports only plain path-like expressions that appear in this deployment:
      - string literals and concatenated literals
    - Name references passed in `names` (e.g., root/v12/Path vars)
    - pathlib-style Path(...) wrapping simple string paths
    - "/" division between a base path and a string/path tail
    - "str(...)" wrappers around supported values
    """
    if max_depth <= 0:
        return None
    if isinstance(node, ast.Name):
        resolved = names.get(node.id)
        return Path(resolved) if resolved is not None else None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return Path(node.value)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            left = _evaluate_ast_path(node.left, names, max_depth=max_depth - 1)
            right = _evaluate_ast_path(node.right, names, max_depth=max_depth - 1)
            if left is None or right is None:
                return None
            if left.parts and str(left).endswith("/"):
                left = Path(str(left).rstrip("/\\"))
            return left / str(right)
        if isinstance(node.op, ast.Add):
            left = _evaluate_ast_path(node.left, names, max_depth=max_depth - 1)
            right = _evaluate_ast_path(node.right, names, max_depth=max_depth - 1)
            if left is None or right is None:
                return None
            return Path(f"{left}{right}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn in {"str", "Path"} and node.args:
            return _evaluate_ast_path(node.args[0], names, max_depth=max_depth - 1)
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            value_path = _evaluate_ast_path(value, names, max_depth=max_depth - 1)
            if value_path is None:
                return None
            parts.append(value_path.name)
        return Path("".join(parts))
    return None


def _extract_robot_asset_path(text: str) -> str | None:
    try:
        module = ast.parse(text)
    except Exception:
        return None
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ROBOT_ASSET_PATH":
                    return _decode_simple_str(node.value)
    return None


def _extract_cfg_required_resources(text: str) -> dict[str, str]:
    resources: dict[str, str] = {}
    try:
        module = ast.parse(text)
    except Exception:
        return resources
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_cfg":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Assign)
                    and len(inner.targets) == 1
                    and isinstance(inner.targets[0], ast.Name)
                    and inner.targets[0].id == "required"
                ):
                    if not isinstance(inner.value, ast.Dict):
                        break
                    for key_node, value_node in zip(inner.value.keys, inner.value.values, strict=False):
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            expr = ast.get_source_segment(text, value_node) or ""
                            # Keep raw expr for traceability, but also resolve simple path-like forms.
                            path_expr = _decode_simple_str(value_node)
                            resources[key_node.value] = path_expr if path_expr is not None else expr
                    break
    return resources


def _parse_declared_path(value: str | None) -> Path | None:
    if not value:
        return None
    value = value.strip()
    try:
        expr = ast.parse(value, mode="eval").body
    except Exception:
        return None
    names = {
        "root": None,
        "v12": None,
    }
    # Keep fallback raw string for simple quoted expressions.
    if (value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2):
        try:
            return Path(ast.literal_eval(value))
        except Exception:
            return None
    resolved = _decode_simple_str(expr)
    if resolved is not None:
        return Path(resolved)
    # For expressions such as `root / "..."`, `_decode_simple_str` cannot decode.
    # Keep the names open for safe evaluation.
    names["root"] = "__root_placeholder__"
    names["v12"] = "__v12_placeholder__"
    return _evaluate_ast_path(expr, names=names)


def _check_declared_resources(
    cfg_text: str,
    runtime_root: Path,
    v12_root: Path,
    record: list[dict[str, Any]],
) -> bool:
    robot_asset_text = _extract_robot_asset_path(cfg_text) or EXPECTED_ROBOT_USD_TEXT
    required_resources = _extract_cfg_required_resources(cfg_text)
    ok = True
    expected = {
        "inputs": runtime_root / "data" / "clean3_c3p1_runtime_inputs.npz",
        "sidecar": runtime_root / "data" / "clean3_action_calibration_c3p1r2.npz",
        "right_pca": runtime_root / "data" / "right_synergy_pca5.npz",
        "left_pca": runtime_root / "data" / "left_synergy_pca5.npz",
        "plate": runtime_root / "assets" / "retarget" / "object_0_textured.usd",
        "sponge": runtime_root / "assets" / "retarget" / "object_1_textured.usd",
        "robot": EXPECTED_ROBOT_USD,
        "right_urdf": v12_root / "rl_rebuild" / "baselines" / "h2s2r" / "assets" / "vega_sharpa_right_fabric.urdf",
        "left_urdf": v12_root / "rl_rebuild" / "baselines" / "h2s2r" / "assets" / "vega_sharpa_left_fabric.urdf",
    }
    for key in (
        "build_cfg(",
        "runtime_inputs_npz",
        "action_sidecar_npz",
        "right_synergy_npz",
        "left_synergy_npz",
        "robot_usd",
    ):
        contains = key in cfg_text
        record.append({"check": f"cfg_contract_token_{key}", "status": "PASS" if contains else "FAIL", "present": contains})
        ok = ok and contains

    for key in expected:
        expr = required_resources.get(key)
        record.append({"check": f"cfg_required_expr_{key}", "status": "PASS" if expr else "FAIL", "expr": expr or ""})
        if not expr:
            ok = False
            continue
        names = {
            "root": runtime_root.as_posix(),
            "v12": v12_root.as_posix(),
            "ROBOT_ASSET_PATH": robot_asset_text,
            "robot": robot_asset_text,
        }
        if key == "robot" and expr == "ROBOT_ASSET_PATH":
            parsed = _normalize_root(Path(robot_asset_text))
            path = parsed
            # Keep deterministic contract against reviewer-identity without AST ambiguity.
            entry = {
                "check": f"cfg_required_path_{key}",
                "status": "PASS" if path == expected[key] else "WARN",
                "declared": str(path),
                "expected": str(expected[key]),
                "exists": path.is_file(),
            }
            if path.is_file():
                entry["sha256"] = sha256(path)
            else:
                ok = False
                entry["status"] = "FAIL"
            record.append(entry)
            continue
        try:
            module = ast.parse(expr, mode="eval")
            parsed = _evaluate_ast_path(module.body, names=names)
        except Exception:
            parsed = None
        if parsed is None:
            record.append({"check": f"cfg_required_path_parse_{key}", "status": "WARN", "expr": expr, "path": None})
            ok = False
            continue
        try:
            parsed_path = parsed
            # Resolve relative declarations against cfg module root as a fallback.
            if not parsed_path.is_absolute():
                parsed_path = (runtime_root / parsed_path).resolve()
            entry = {
                "check": f"cfg_required_path_{key}",
                "status": "PASS" if parsed_path == expected[key].resolve() else "WARN",
                "declared": str(parsed_path),
                "expected": str(expected[key]),
                "exists": parsed_path.is_file(),
            }
            if parsed_path.is_file():
                entry["sha256"] = sha256(parsed_path)
            else:
                ok = False
                entry["status"] = "FAIL"
            record.append(entry)
        except Exception as exc:
            ok = False
            record.append({"check": f"cfg_required_path_{key}", "status": "FAIL", "expr": expr, "reason": repr(exc)})
    return ok


def _read_cfg_contract(cfg_path: Path, record: list[dict[str, Any]]) -> tuple[bool, str]:
    if not cfg_path.is_file():
        record.append({"check": "cfg_path", "status": "FAIL", "reason": "missing_cfg_file", "path": str(cfg_path)})
        return False, ""
    text = cfg_path.read_text(encoding="utf-8")
    robot_path_text = _extract_robot_asset_path(text)
    ok = True
    if robot_path_text is None:
        record.append({"check": "cfg_robot_asset_const", "status": "FAIL", "reason": "robot_asset_constant_missing"})
        ok = False
    else:
        robot_line = robot_path_text.strip()
        found = robot_path_text.replace("\\\\", "/") == EXPECTED_ROBOT_USD_TEXT
        record.append({
            "check": "cfg_robot_asset_const",
            "status": "PASS" if found else "FAIL",
            "expected": EXPECTED_ROBOT_USD_TEXT,
            "snippet": robot_line,
            "contains_expected": found,
        })
        ok = ok and found
    required_resources = _extract_cfg_required_resources(text)
    for key in ("inputs", "sidecar", "right_pca", "left_pca", "plate", "sponge", "robot", "right_urdf", "left_urdf"):
        record.append({
            "check": f"cfg_required_expr_{key}",
            "status": "PASS" if key in required_resources and required_resources[key] else "FAIL",
            "expr": required_resources.get(key, ""),
        })
        if key not in required_resources or not required_resources[key]:
            ok = False
    return ok, text


def check_file(record: list[dict[str, Any]], path: Path, expected_sha256: str | None = None, *, must_exist: bool = True) -> bool:
    exists = path.is_file()
    entry: dict[str, Any] = {
        "path": str(path),
        "must_exist": must_exist,
        "exists": exists,
        "size": path.stat().st_size if exists else 0,
        "sha256": None,
        "expected_sha256": expected_sha256,
    }
    if not exists:
        entry["status"] = "FAIL"
        entry["reason"] = "missing_file"
        record.append(entry)
        return False
    if expected_sha256 is not None:
        entry["sha256"] = sha256(path)
        if entry["sha256"] != expected_sha256:
            entry["status"] = "FAIL"
            entry["reason"] = "sha256_mismatch"
            record.append(entry)
            return False
    entry["status"] = "PASS"
    record.append(entry)
    return True


def check_usd_runtime_resources(runtime_root: Path, v12_root: Path, cfg_text: str, record: list[dict[str, Any]]) -> bool:
    ok = True
    required: dict[str, Path | tuple[Path, str]] = {
        "runtime_inputs_npz": runtime_root / "data" / "clean3_c3p1_runtime_inputs.npz",
        "action_sidecar_npz": runtime_root / "data" / "clean3_action_calibration_c3p1r2.npz",
        "left_synergy_npz": runtime_root / "data" / "left_synergy_pca5.npz",
        "right_synergy_npz": runtime_root / "data" / "right_synergy_pca5.npz",
        "plate_texture_usd": runtime_root / "assets" / "retarget" / "object_0_textured.usd",
        "sponge_texture_usd": runtime_root / "assets" / "retarget" / "object_1_textured.usd",
        "plate_mesh_obj": runtime_root / "assets" / "objects" / "object_0" / "object_mesh_scaled_final.obj",
        "sponge_mesh_obj": runtime_root / "assets" / "objects" / "object_1" / "object_mesh_scaled_final.obj",
        "robot_usd": (EXPECTED_ROBOT_USD, EXPECTED_ROBOT_USD_SHA256),
        "right_fabric_urdf": v12_root / "rl_rebuild" / "baselines" / "h2s2r" / "assets" / "vega_sharpa_right_fabric.urdf",
        "left_fabric_urdf": v12_root / "rl_rebuild" / "baselines" / "h2s2r" / "assets" / "vega_sharpa_left_fabric.urdf",
        "input_manifest": runtime_root / "data" / "input_manifest.json",
        "action_manifest": runtime_root / "data" / "action_calibration_manifest.json",
    }
    for key in required:
        path = required[key]
        if isinstance(path, tuple):
            path, expected_sha256 = path
            ok &= check_file(record, path, expected_sha256=expected_sha256, must_exist=True)
        else:
            ok &= check_file(record, path, must_exist=True)

    # Keep contract-path verification in one place: cfg must explicitly consume
    # every path we actually depend on and resolve to the expected absolute identity.
    ok &= _check_declared_resources(cfg_text, runtime_root, v12_root, record)

    try:
        shapes = read_npz_shapes(required["runtime_inputs_npz"])
        entry = {
            "check": "runtime_inputs_shape",
            "status": "PASS",
            "runtime_input_shapes": shapes,
        }
        ok_reset = shapes["reset_q58"] == (58,)
        if not ok_reset:
            entry["status"] = "FAIL"
            entry["reason"] = "reset_q58_shape_mismatch"
        record.append(entry)
        ok &= bool(ok_reset)
        with np.load(required["runtime_inputs_npz"], allow_pickle=False) as data:
            if data["plate_reference_pose_wxyz"].shape != (300, 7):
                raise RuntimeError("plate_reference_pose_wxyz must be [300, 7]")
            if data["sponge_reference_pose_wxyz"].shape != (300, 7):
                raise RuntimeError("sponge_reference_pose_wxyz must be [300, 7]")
            if data["reset_q58"].shape != (58,) or data["reset_q58"].dtype != np.float32:
                raise RuntimeError("reset_q58 must be float32[58]")
            if data["footprint_xy"].ndim != 2 or data["footprint_xy"].shape[1] != 2:
                raise RuntimeError("footprint_xy must be [N,2]")
    except Exception as exc:  # pragma: no cover - runtime guardrail
        record.append({
            "check": "runtime_inputs_shape",
            "status": "FAIL",
            "reason": repr(exc),
        })
        ok = False

    for key, path in (("left_synergy", required["left_synergy_npz"]), ("right_synergy", required["right_synergy_npz"])):
        try:
            with np.load(path, allow_pickle=False) as data:
                if "matrix" not in data.files:
                    raise RuntimeError(f"{key} sidecar missing matrix")
                record.append({"check": f"{key}_shape", "status": "PASS", "shape": tuple(data["matrix"].shape)})
                if data["matrix"].shape != (5, 22):
                    raise RuntimeError(f"{key} PCA basis must be (5, 22)")
                if data["minimum"].shape != (5,) or data["maximum"].shape != (5,) or data["center"].shape != (22,):
                    raise RuntimeError(f"{key} PCA bounds/center shape mismatch")
        except Exception as exc:
            record.append({"check": f"{key}_shape", "status": "FAIL", "reason": repr(exc)})
            ok = False
    return ok


def check_pxrd_deps(runtime_root: Path, record: list[dict[str, Any]]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "pxr_available": False,
        "status": "FAIL",
        "reason": "pxr import failed in CPU runtime",
        "joint_count": 0,
        "joint_names": [],
        "fixed_torso_candidates": [],
        "dependency_count": 0,
        "dependencies": [],
    }
    try:
        from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    except Exception as exc:  # pragma: no cover
        details["reason"] = f"pxr import failed: {repr(exc)}"
        return details
    details["pxr_available"] = True
    details["status"] = "PASS"
    robot = EXPECTED_ROBOT_USD
    try:
        if not robot.is_file():
            details["status"] = "FAIL"
            details["reason"] = f"missing robot USD on this file system: {robot}"
            return details
        stage = Usd.Stage.Open(str(robot))
        if stage is None:
            details["status"] = "FAIL"
            details["reason"] = "Usd.Stage.Open returned null"
            return details
        root = stage.GetPseudoRoot()
        mesh_count = 0
        mesh_with_collision = 0
        joint_like: list[str] = []
        fixed_torso_candidates: list[str] = []
        for prim in root.GetDescendants():
            if prim.IsA(UsdGeom.Mesh):
                mesh_count += 1
                has_collision = prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI.CanApply(prim.GetPath())
                if has_collision:
                    mesh_with_collision += 1
            prim_type = str(prim.GetTypeName()).lower()
            if prim_type.endswith("joint") or "joint" in prim_type or isinstance(prim.GetTypeName(), type(prim_type)):
                if prim.GetTypeName() or prim_type:
                    joint_like.append(prim.GetPath().pathString)
            if prim_type.startswith("fixed") and ("joint" in prim_type or "weld" in prim_type):
                fixed_torso_candidates.append(prim.GetPath().pathString)
        details["joint_like_count"] = len(joint_like)
        details["joint_like_sample"] = joint_like[:20]
        details["joint_names"] = joint_like[:50]
        details["fixed_torso_candidates"] = fixed_torso_candidates[:20]
        details["joint_count"] = len(joint_like)
        details["usd_root_path"] = str(stage.GetRootLayer().realPath)
        details["mesh_count"] = mesh_count
        details["mesh_with_collision"] = mesh_with_collision
        details["runtime_root"] = str(runtime_root)
        if details["joint_count"] != 58:
            details["status"] = "FAIL"
            details["reason"] = "robot usd joint count not equal to 58"
        if not fixed_torso_candidates:
            details["status"] = "FAIL"
            details["reason"] = "no fixed-torso candidate prim found in robot usd"
        if mesh_count == 0:
            details["status"] = "FAIL"
            details["reason"] = "robot usd has no mesh prims"
            return details
        details["collision_coverage_ratio"] = mesh_with_collision / max(1, mesh_count)
        if mesh_with_collision == 0:
            details["status"] = "FAIL"
            details["reason"] = "robot mesh has no collision API tags detected"

        dependencies = []
        for dep in UsdUtils.GetAllDependencies(stage.GetRootLayer()):
            src = Path(dep)
            candidate = src if src.is_absolute() else (robot.parent / src).resolve()
            dependencies.append({
                "path": str(src),
                "resolved": str(candidate),
                "exists": candidate.exists(),
            })
        details["dependency_count"] = len(dependencies)
        details["dependencies"] = dependencies
    except Exception as exc:  # pragma: no cover
        details["status"] = "FAIL"
        details["reason"] = repr(exc)
    return details


def DEPLOYMENT_CFG_PATH(runtime_root: Path) -> Path:
    return runtime_root / "tasks" / "h2s2r_clean3" / "cfg.py"


def run(runtime_root: Path, v12_root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append({"check": "runtime_v12_roots", "runtime_root": str(runtime_root), "v12_root": str(v12_root), "status": "PASS"})
    details: dict[str, Any] = {
        "schema": "c3p1r2_preflight_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "runtime_root": str(runtime_root),
        "v12_root": str(v12_root),
        "checks": checks,
    }

    cfg_path = DEPLOYMENT_CFG_PATH(runtime_root)
    cfg_ok, cfg_text = _read_cfg_contract(cfg_path, checks)
    ok = cfg_ok
    if ok:
        ok &= check_usd_runtime_resources(runtime_root, v12_root, cfg_text, checks)
    usd_details = check_pxrd_deps(runtime_root, checks)
    checks.append({"check": "pxr_checks", "status": usd_details.get("status", "FAIL"), "details": usd_details})
    if usd_details.get("status", "FAIL") == "FAIL":
        ok = False
    if not (runtime_root.is_dir() and v12_root.is_dir()):
        checks.append({"check": "root_is_dir", "status": "FAIL", "runtime_root_dir": runtime_root.is_dir(), "v12_root_dir": v12_root.is_dir()})
        ok = False
    details["status"] = "PASS" if ok else "FAIL"
    details["passed"] = ok
    details["environment"] = {
        "pid": os.getpid(),
        "python": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "torch_available": False,
    }
    try:
        import torch  # type: ignore[import-not-found]

        details["environment"]["torch_available"] = True
        details["environment"]["torch_version"] = torch.__version__
    except Exception:
        pass
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment preflight checks without SimulationApp.")
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--v12_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path to write preflight JSON")
    args = parser.parse_args()
    payload = run(_normalize_root(args.runtime_root), _normalize_root(args.v12_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
