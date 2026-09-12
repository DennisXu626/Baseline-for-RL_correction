"""Bind metadata without repeating MANO, finger optimization or arm IK."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import sha256, ndarray_sha256
from .schema import schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    a = ap.parse_args()
    directory = a.directory
    with np.load(directory / "s1_operation_targets.npz") as z:
        goals = np.stack([z[f"object_{i}"][np.flatnonzero(z["stage"] == "synthetic_hold")[-1], :3] for i in (0, 1)])
    bindings = []
    for name, target in (("reference_s1_v2.npz", "reference_s1_v2_bound.npz"),
                         ("wp1_reference_schema_v2_diagnostic.npz", "wp1_reference_schema_v2_bound_diagnostic.npz")):
        with np.load(directory / name) as z:
            fields = {k: z[k] for k in z.files}
        before = ndarray_sha256(fields["joint_q_rad"])
        fields["contract_sha256"] = np.array(sha256(a.config))
        fields["final_object_goal_m"] = goals
        fields["schema_version"] = np.array("pour17_reference_s1_v2" if name.startswith("reference_s1") else "wp1_trajectory_wp2_observation_diagnostic_only")
        np.savez_compressed(directory / target, **fields)
        bindings.append({"input": name, "input_sha256": sha256(directory / name),
            "output": target, "output_sha256": sha256(directory / target), "joint_q_unchanged_sha256": before,
            "final_goal": "S1 synthetic terminal object position, not the final home-return sample"})
    (directory / "observation_action_schema_v2.json").write_text(json.dumps(schema(), indent=2) + "\n")
    (directory / "wp2_input_binding.json").write_text(json.dumps({"bindings": bindings,
        "config_sha256": sha256(a.config), "script_sha256": sha256(Path(__file__)),
        "final_object_goal_m_object0_then1": goals.tolist(), "q_recomputed": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
