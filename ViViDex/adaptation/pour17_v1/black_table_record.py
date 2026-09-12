"""Record the authorized visual-only diff; no simulation or trajectory edits."""
from pathlib import Path
import ast
import difflib
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    out = root / "artifacts/pour17_v1/wp2_20260912_01/black_table_correction_v1"
    rows, patches = [], []
    for old_name, new_name in (("env_before.py", "env.py"),
                               ("wp1_compat_env_before.py", "wp1_compat/env.py")):
        old, new = out / old_name, root / "adaptation/pour17_v1" / new_name
        before, after = old.read_text(), new.read_text()
        expected = before.replace("diffuse_color=(0.4, 0.3, 0.2)", "diffuse_color=(0.0, 0.0, 0.0)")
        assert before.count("diffuse_color=(0.4, 0.3, 0.2)") == 1
        assert after == expected, "color patch includes unexpected changes"
        assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(after))
        rows.append({"before": str(old), "after": str(new), "before_sha256": digest(old),
                     "after_sha256": digest(new), "only_diffuse_color_changed": True})
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), old_name, new_name))
    (out / "color_only.patch").write_text("".join(patches))
    (out / "manifest.json").write_text(json.dumps({
        "visual_version": "WP2_BLACK_TABLE_ONLY_V1", "status": "COLOR_ONLY_DIFF_VERIFIED",
        "files": rows, "snapshot_sha256": digest(out / "source_before.tar.gz"),
        "patch_sha256": digest(out / "color_only.patch"), "script_sha256": digest(Path(__file__)),
        "physics_control_reset_observation_reward_unchanged_by_color_patch": True,
        "legacy_wp1_checkpoint_for_video_only_not_wp2_resume": True,
        "review_evidence": {p.name: digest(p) for p in (out / "camera_frame0.png", out / "camera_review.json",
                                                       out / "pause_resume_record.json") if p.exists()},
    }, indent=2) + "\n")
    print("Color-only diff verified for WP2 and WP1 video compatibility copy")


if __name__ == "__main__":
    main()
