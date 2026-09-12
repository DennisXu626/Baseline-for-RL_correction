"""Apply the single approved common fixed-root overlay, never reset the root."""
import json
from pathlib import Path
from .geometry import sha256


def apply_root_overlay(cfg, bundle_root, overlay_path):
    overlay_path = Path(overlay_path)
    record = json.loads(overlay_path.read_text())
    assert record["world_version"] == "CW_ROOT_NEGATIVE_X_075_V1"
    assert record["robot_root_env_xyz_m"] == [-.75, 0., 0.]
    assert record["robot_root_quat_wxyz"] == [1., 0., 0., 0.]
    root = Path(bundle_root) / "world"
    for name, field in (("world_manifest.json", "base_world_sha256"),
                        ("canonical_reset_v1.json", "base_canonical_sha256"),
                        ("vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd", "base_robot_usd_sha256")):
        assert sha256(root / name) == record[field], name
    # Isaac's USD articulation spawn applies the complete fixed-root pose.
    # This must be confirmed by live root/constraint readback, not assumed.
    cfg.robot.init_state.pos = tuple(record["robot_root_env_xyz_m"])
    cfg.robot.init_state.rot = tuple(record["robot_root_quat_wxyz"])
    cfg.wp4_world_version = record["world_version"]
    cfg.wp4_root_overlay_sha256 = sha256(overlay_path)
    return record
