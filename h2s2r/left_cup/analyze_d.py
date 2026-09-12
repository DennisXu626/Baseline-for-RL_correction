"""Recompute D replay integrity and contact timing from saved runtime arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(archive, name):
    value = archive[name]
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite values in {name}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sub_path = args.run / "substeps.npz"
    contact_path = args.run / "contacts.npz"
    sub = np.load(sub_path, allow_pickle=False)
    contacts = np.load(contact_path, allow_pickle=False)
    commands = np.load(args.commands, allow_pickle=False)

    q = finite_array(sub, "q")
    qd = finite_array(sub, "qd")
    cup = finite_array(sub, "cup")
    cup_vel = finite_array(sub, "cup_vel")
    request = finite_array(sub, "request_left")
    counts = sub["view_counts"]
    sensor_names = sub["contact_view_names"].astype(str).tolist()
    joint_names = sub["joint_names"].astype(str).tolist()
    drive_names = sub["drive_joint_names"].astype(str).tolist()
    expected_names = commands["joint_names"].astype(str).tolist()
    controls = sub["control"].astype(int)
    within = sub["within_substep"].astype(int)

    if q.shape[0] != 1200 or qd.shape != q.shape or request.shape != (1200, 29):
        raise ValueError("unexpected D substep shapes")
    expected_control = np.repeat(np.arange(1, 101), 12)
    expected_within = np.tile(np.arange(12), 100)
    target_repeated = np.repeat(commands["targets_rad"], 12, axis=0)
    checks = {
        "substeps_1200": len(q) == 1200,
        "control_index_exact": np.array_equal(controls, expected_control),
        "within_index_exact": np.array_equal(within, expected_within),
        "request_exact_frozen_targets": np.array_equal(request, target_repeated),
        "joint_names_unique": len(joint_names) == len(set(joint_names)),
        "left_joint_order_present": all(name in joint_names for name in expected_names),
        "contact_view_count_34": len(sensor_names) == 34 and counts.shape == (1200, 34),
        "contact_counts_match_rows": int(counts.sum()) == int(len(contacts["contact_substep"])),
        "all_core_arrays_finite": True,
        "early_atomic_files_12": len(list((args.run / "early_substeps").glob("substep_*.npz"))) == 12,
    }

    left_indices = [joint_names.index(name) for name in expected_names]
    right_indices = [index for index in range(len(joint_names)) if index not in set(left_indices)]
    target = finite_array(sub, "target")[:, left_indices]
    processed = finite_array(sub, "processed")[:, left_indices]
    checks["joint_target_exact_frozen_targets"] = np.array_equal(target, target_repeated)
    checks["processed_target_exact_frozen_targets"] = np.array_equal(processed, target_repeated)
    checks["right_target_constant"] = np.array_equal(sub["target"][:, right_indices], np.repeat(sub["target"][:1, right_indices], 1200, axis=0))
    checks["right_processed_target_constant"] = np.array_equal(sub["processed"][:, right_indices], np.repeat(sub["processed"][:1, right_indices], 1200, axis=0))
    if len(drive_names):
        drive = finite_array(sub, "drive")
        drive_indices = [drive_names.index(name) for name in expected_names]
        right_drive_indices = [index for index, name in enumerate(drive_names) if name not in set(expected_names)]
        checks["drive_target_exact_frozen_targets"] = np.array_equal(drive[:, drive_indices], target_repeated)
        checks["right_direct_drive_constant"] = np.array_equal(drive[:, right_drive_indices], np.repeat(drive[:1, right_drive_indices], 1200, axis=0))
    else:
        checks["drive_target_exact_frozen_targets"] = False

    contact_step = contacts["contact_substep"].astype(int)
    contact_sensor = contacts["contact_sensor"].astype(int)
    separation = finite_array(contacts, "contact_separation").reshape(-1)
    force = finite_array(contacts, "contact_force").reshape(-1)
    point = finite_array(contacts, "contact_point")
    normal = finite_array(contacts, "contact_normal")
    checks["contact_arrays_same_length"] = len({len(contact_step), len(contact_sensor), len(separation), len(force), len(point), len(normal)}) == 1
    checks["contact_indices_in_range"] = (len(contact_step) == 0 or
        (contact_step.min() >= 0 and contact_step.max() < 1200 and
         contact_sensor.min() >= 0 and contact_sensor.max() < len(sensor_names)))
    checks = {name: bool(value) for name, value in checks.items()}

    state_rows = []
    state_path = args.run / "state_writes.jsonl"
    if state_path.exists():
        state_rows = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    phase_counts = {}
    for row in state_rows:
        key = f"{row.get('phase')}|{row.get('event')}|{row.get('asset_path')}|{row.get('method')}"
        phase_counts[key] = phase_counts.get(key, 0) + 1
    active_writes = [row for row in state_rows if row.get("phase") == "control"]

    per_sensor = {}
    for index, name in enumerate(sensor_names):
        mask = contact_sensor == index
        steps = contact_step[mask]
        sep = separation[mask]
        per_sensor[name] = {
            "records": int(mask.sum()),
            "first_global_substep": None if not mask.any() else int(steps.min()),
            "first_control": None if not mask.any() else int(steps.min() // 12 + 1),
            "first_within_substep": None if not mask.any() else int(steps.min() % 12),
            "min_separation_m": None if not mask.any() else float(sep.min()),
            "max_separation_m": None if not mask.any() else float(sep.max()),
            "max_force_N": None if not mask.any() else float(force[mask].max()),
        }

    pinky_name = "left_pinky_MP"
    table_name = "cup_to_table"
    key_sensors = {name: per_sensor[name] for name in (pinky_name, table_name)}
    pinky_sensor = sensor_names.index(pinky_name)
    pinky_body = sub["left_body_names"].astype(str).tolist().index(pinky_name)
    positive_pinky = np.flatnonzero((contact_sensor == pinky_sensor) & (force > 0))
    first_pinky_record = int(positive_pinky[0])
    first_pinky_step = int(contact_step[first_pinky_record])
    first_pinky_point = point[first_pinky_record]
    first_pinky_cup = cup[first_pinky_step, :3]
    first_pinky_body_pos = sub["body_pos"][first_pinky_step, pinky_body]
    first_pinky_event = {
        "global_substep": first_pinky_step,
        "control": first_pinky_step // 12 + 1,
        "within_substep": first_pinky_step % 12,
        "time_after_first_control_start_s": (first_pinky_step + 1) / 240.0,
        "sensor": pinky_name,
        "point_w_m": first_pinky_point.tolist(),
        "force_N": float(force[first_pinky_record]),
        "separation_m": float(separation[first_pinky_record]),
        "normal_w": normal[first_pinky_record].tolist(),
        "cup_center_m": first_pinky_cup.tolist(),
        "point_minus_cup_center_m": (first_pinky_point - first_pinky_cup).tolist(),
        "cup_radial_xy_mm": float(np.linalg.norm((first_pinky_point - first_pinky_cup)[:2]) * 1000.0),
        "cup_relative_z_mm": float((first_pinky_point - first_pinky_cup)[2] * 1000.0),
        "pinky_body_position_m": first_pinky_body_pos.tolist(),
        "point_to_pinky_body_origin_mm": float(np.linalg.norm(first_pinky_point - first_pinky_body_pos) * 1000.0),
    }
    reset_cup = np.asarray(json.loads((args.run / "reset_validation.json").read_text(encoding="utf-8"))["actual"]["cup"][:3])
    first12_rows = []
    first12_positive_force = []
    for step in range(12):
        row = {
            "global_substep": step,
            "control": int(controls[step]),
            "within_substep": int(within[step]),
            "cup_vx_mps": float(cup_vel[step, 0]),
            "cup_vy_mps": float(cup_vel[step, 1]),
            "cup_vz_mps": float(cup_vel[step, 2]),
            "cup_linear_speed_mps": float(np.linalg.norm(cup_vel[step, :3])),
            "cup_x_m": float(cup[step, 0]),
            "cup_y_m": float(cup[step, 1]),
            "cup_z_m": float(cup[step, 2]),
            "cup_displacement_from_reset_mm": float(np.linalg.norm(cup[step, :3] - reset_cup) * 1000.0),
        }
        for name in (pinky_name, table_name):
            index = sensor_names.index(name)
            mask = (contact_step == step) & (contact_sensor == index)
            row[f"{name}_records"] = int(mask.sum())
            row[f"{name}_min_separation_m"] = None if not mask.any() else float(separation[mask].min())
            row[f"{name}_max_force_N"] = None if not mask.any() else float(force[mask].max())
        all_mask = contact_step == step
        row["all_left_plus_table_records"] = int(all_mask.sum())
        first12_rows.append(row)
        active = []
        for index in np.unique(contact_sensor[all_mask]):
            mask = all_mask & (contact_sensor == index)
            max_force = float(force[mask].max())
            if max_force > 0:
                active.append({"sensor": sensor_names[index], "max_force_N": max_force,
                               "min_separation_m": float(separation[mask].min())})
        first12_positive_force.append({"global_substep": step, "contacts": active})

    with (args.out / "FIRST_12_SUBSTEPS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(first12_rows[0]))
        writer.writeheader()
        writer.writerows(first12_rows)

    timeline_rows = []
    for step in sorted(set(contact_step.tolist())):
        for index, name in enumerate(sensor_names):
            mask = (contact_step == step) & (contact_sensor == index)
            if not mask.any():
                continue
            timeline_rows.append({
                "global_substep": step,
                "control": step // 12 + 1,
                "within_substep": step % 12,
                "sensor": name,
                "records": int(mask.sum()),
                "min_separation_m": float(separation[mask].min()),
                "max_separation_m": float(separation[mask].max()),
                "max_force_N": float(force[mask].max()),
            })
    with (args.out / "CONTACT_TIMELINE.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timeline_rows[0]) if timeline_rows else
                                ["global_substep", "control", "within_substep", "sensor", "records", "min_separation_m", "max_separation_m", "max_force_N"])
        writer.writeheader()
        writer.writerows(timeline_rows)

    first_motion_order = np.flatnonzero(np.linalg.norm(cup_vel[:, :3], axis=1) > 0)
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "source": {
            "run": str(args.run), "commands": str(args.commands),
            "substeps_sha256": sha256(sub_path), "contacts_sha256": sha256(contact_path),
            "state_writes_sha256": sha256(state_path),
        },
        "shapes": {name: list(sub[name].shape) for name in sub.files},
        "contact_shapes": {name: list(contacts[name].shape) for name in contacts.files},
        "key_sensors": key_sensors,
        "first_pinky_force_event": first_pinky_event,
        "global_contact_separation": {
            "records": int(len(separation)),
            "negative": int((separation < 0).sum()),
            "zero": int((separation == 0).sum()),
            "positive": int((separation > 0).sum()),
            "min_m": float(separation.min()),
            "max_m": float(separation.max()),
            "positive_force_records": int((force > 0).sum()),
        },
        "trajectory_outcome": {
            "final_cup_position_m": cup[-1, :3].tolist(),
            "net_cup_displacement_from_reset_mm": float(np.linalg.norm(cup[-1, :3] - reset_cup) * 1000.0),
            "final_cup_z_change_from_reset_mm": float((cup[-1, 2] - reset_cup[2]) * 1000.0),
            "minimum_cup_z_change_from_reset_mm": float((cup[:, 2].min() - reset_cup[2]) * 1000.0),
            "maximum_cup_z_change_from_reset_mm": float((cup[:, 2].max() - reset_cup[2]) * 1000.0),
            "peak_cup_linear_speed_mps": float(np.linalg.norm(cup_vel[:, :3], axis=1).max()),
        },
        "all_sensor_summary": per_sensor,
        "first_nonzero_cup_linear_velocity_substep": None if not len(first_motion_order) else int(first_motion_order[0]),
        "first12": first12_rows,
        "first12_positive_force_contacts": first12_positive_force,
        "state_writes": {
            "records": len(state_rows), "phase_event_asset_method_counts": phase_counts,
            "control_phase_records": len(active_writes), "control_phase_rows": active_writes,
        },
        "interpretation_guardrails": [
            "negative separation is reported as returned by the actual contact view; it is not relabeled as mesh penetration depth",
            "cup motion is temporally compared with all left-hand and cup-table contacts, not attributed to the pinky alone",
            "input collision metadata is not represented as the cooked PhysX geometry",
        ],
    }
    (args.out / "D_ANALYSIS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": checks,
                      "key_sensors": key_sensors,
                      "first_nonzero_cup_linear_velocity_substep": result["first_nonzero_cup_linear_velocity_substep"],
                      "control_state_writes": len(active_writes)}, indent=2))


if __name__ == "__main__":
    main()
