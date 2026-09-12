"""Read-only identity and live placement checks for the approved WP4 run."""
from pathlib import Path
import numpy as np
from .geometry import sha256


def identity(bundle, reference, urdf, overlay):
    paths = {"reference": Path(reference), "urdf": Path(urdf), "overlay": Path(overlay)}
    for name in ("world_manifest.json", "canonical_reset_v1.json",
                 "vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd"):
        paths[name] = Path(bundle) / "world" / name
    for path in sorted(Path(__file__).parent.glob("*.py")):
        paths["source/" + path.name] = path
    return {name: sha256(path) for name, path in paths.items()}


def readback(raw):
    transforms = raw.robot.root_physx_view.get_root_transforms().cpu().numpy()
    positions = transforms[:, :3] - raw.scene.env_origins.cpu().numpy()
    q = raw.robot.root_physx_view.get_dof_positions()[:, raw.joint_ids].cpu().numpy()
    root_error = float(np.max(np.abs(positions - [-.75, 0., 0.])))
    q_error = float(np.max(np.abs(q - raw.reset_q.cpu().numpy())))
    if root_error > 1e-6 or q_error > 1e-6:
        raise RuntimeError(f"WP4 live root/canonical mismatch: {root_error}, {q_error}")
    if raw.cfg.probe_only or raw.cfg.external_evaluator_controls_termination:
        raise RuntimeError("diagnostic termination override in training")
    return {"root_env_xyz_m": positions.tolist(), "root_max_abs_error_m": root_error,
            "canonical_q_max_abs_error_rad": q_error, "normal_termination": True,
            "reset_state": raw.reset_state()}


def compatible_resume(previous, current):
    """Allow only driver/instrumentation changes, never world or method drift."""
    if not isinstance(previous, dict):
        return {"compatible": False, "reason": "checkpoint has no development binding"}
    driver_names = {"source/train.py", "source/internal_eval.py",
                    "source/development_binding.py", "source/development_job.py",
                    "source/policy_diagnostic_eval.py", "source/development_extension_job.py",
                    "source/development_curriculum_job.py",
                    "source/robot_table_contact.py"}
    keys = set(previous) | set(current)
    changes = {key: {"previous": previous.get(key), "current": current.get(key)}
               for key in sorted(keys) if previous.get(key) != current.get(key)}
    registered_eval_only_env_change = changes.get("source/env.py") == {
        "previous": "2b2b8f7aa5ae9c618d372f1caf47efc32c9bfcb207642b6370f2f0c19671ac0f",
        "current": "fb63f2dfea1559691ae7a0c5b70aab4e431c7f48f6195d990d274ad88d20b479"}
    allowed = driver_names | ({"source/env.py"} if registered_eval_only_env_change else set())
    forbidden = {key: value for key, value in changes.items() if key not in allowed}
    return {"compatible": not forbidden, "allowed_driver_changes": changes,
            "registered_eval_only_env_change": registered_eval_only_env_change,
            "forbidden_method_or_input_changes": forbidden}
