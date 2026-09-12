"""Read-only full MANO decoding and proposed local-calibration audit for M1-A.

This diagnostic performs normal MANO shape blend, pose blend, kinematic chain,
and linear-blend skinning.  It does not optimise, retarget, replay, or write a
robot reference.  Every choice loaded from the companion candidate JSON remains
PROPOSED unless its report entry names independent source evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
COLORS = dict(zip(FINGERS, ("#9c755f", "#4e79a7", "#59a14f", "#e15759", "#b07aa1")))
MANO_OUTPUT = {
    "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
# These roles are verified as the public ViViDex/DexYCB 21-point visualization
# names.  They deliberately do not translate thumb_pip/dip into a different
# anatomical CMC/MCP/IP convention.
OFFICIAL_21_LABELS = (
    "wrist", "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
)
# Independently transcribed from the trusted remote official ManoLayer selector
# (source SHA-256 is recorded in the runtime evidence), then checked against
# the local source parser below.  Do not derive this expectation from the
# generated `final_sources` object under test.
EXPECTED_TIP_VERTEX_BY_FINGER = {
    "thumb": 745, "index": 317, "middle": 445, "ring": 556, "pinky": 673,
}


class _ChumpyNode:
    """Data holder used only to inspect legacy chumpy nodes in MANO pkl files."""


class _AuditUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("chumpy"):
            return _ChumpyNode
        return super().find_class(module, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_chumpy(value):
    """Resolve only the Select/Ch graph representation present in this asset."""
    if isinstance(value, np.ndarray):
        return value
    if not isinstance(value, _ChumpyNode):
        return value
    state = value.__dict__
    if {"a", "idxs", "preferred_shape"}.issubset(state):
        source = np.asarray(resolve_chumpy(state["a"]))
        return source.reshape(-1)[np.asarray(state["idxs"], dtype=np.int64)].reshape(tuple(state["preferred_shape"]))
    if "x" in state:
        return resolve_chumpy(state["x"])
    if "a" in state:
        return resolve_chumpy(state["a"])
    raise RuntimeError(f"unsupported legacy chumpy node fields: {sorted(state)}")


def load_model(path: Path) -> dict:
    with path.open("rb") as stream:
        model = _AuditUnpickler(stream, encoding="latin1").load()
    model["shapedirs"] = np.asarray(resolve_chumpy(model["shapedirs"]), dtype=np.float64)
    for name in ("v_template", "posedirs", "weights", "hands_mean", "hands_components", "kintree_table"):
        model[name] = np.asarray(model[name], dtype=np.float64)
    model["J_regressor"] = model["J_regressor"].astype(np.float64)
    return model


def parse_source_indices(source: str, expression: str) -> tuple[int, ...]:
    match = re.search(expression, source, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"could not locate official ManoLayer expression: {expression}")
    return tuple(int(value) for value in re.findall(r"\d+", match.group(1)))


def official_output_contract(source_path: Path, kintree_table: np.ndarray) -> dict:
    """Recover the official output contract from source, separate from decode_mano.

    ManoLayer first concatenates breadth levels, then applies `reorder_idxs` to
    restore raw MANO joint order.  Its final 21-point selector is applied only
    after the five tip vertices are appended.  This routine independently reads
    those selectors from the unmodified official source and validates the raw
    chain against the asset kinematic tree.
    """
    source = source_path.read_text(encoding="utf-8")
    level_selector = parse_source_indices(source, r"reorder_idxs\s*=\s*\[([^\]]+)\]")
    final_selector = parse_source_indices(
        source,
        r"Reorder joints to match visualization utilities\s+th_jtr\s*=\s*th_jtr\[:,\s*\[([^\]]+)\]\]",
    )
    tip_vertices = parse_source_indices(
        source,
        r"else:\s+tips\s*=\s*th_verts\[:,\s*\[([^\]]+)\]\]",
    )
    parents = np.asarray(kintree_table[0], dtype=np.int64).copy()
    parents[0] = -1
    levels = [[0]]
    while sum(len(level) for level in levels) < len(parents):
        previous = {joint for level in levels for joint in level}
        next_level = [joint for joint, parent in enumerate(parents) if joint not in previous and parent in levels[-1]]
        if not next_level:
            raise RuntimeError("kinematic tree cannot be level-ordered")
        levels.append(next_level)
    breadth_raw = tuple(joint for level in levels for joint in level)
    restored_raw = tuple(breadth_raw[index] for index in level_selector)
    if restored_raw != tuple(range(16)):
        raise RuntimeError(f"official transform reorder does not restore raw order: {restored_raw}")
    pre_final = tuple(("joint", joint) for joint in restored_raw) + tuple(("tip_vertex", vertex) for vertex in tip_vertices)
    final_sources = tuple(pre_final[index] for index in final_selector)
    expected_joint_sources = (0, 13, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9)
    if tuple(source[1] for source in final_sources if source[0] == "joint") != expected_joint_sources:
        raise RuntimeError("official 21-point selector has an unexpected joint sequence")
    return {
        "official_source_sha256": sha256(source_path),
        "breadth_raw_order": breadth_raw,
        "official_transform_reorder": level_selector,
        "restored_raw_transform_order": restored_raw,
        "official_final_selector": final_selector,
        "tip_vertices_left": tip_vertices,
        "final_sources": final_sources,
        "parents": parents,
    }


def output_mapping_and_structure(contract: dict) -> tuple[list[dict], dict]:
    """Report output-index sources and validate finger chains against raw parents."""
    raw_chains = {
        "index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9),
        "ring": (10, 11, 12), "thumb": (13, 14, 15),
    }
    expected_outputs = {
        "thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
        "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20),
    }
    parents = contract["parents"]
    chain_results = {}
    for finger, chain in raw_chains.items():
        parent_ok = parents[chain[0]] == 0 and all(parents[child] == parent for parent, child in zip(chain, chain[1:]))
        sources = [contract["final_sources"][index] for index in expected_outputs[finger]]
        actual_tip = sources[3][1] if sources[3][0] == "tip_vertex" else None
        source_ok = tuple(source[1] for source in sources[:3]) == chain
        tip_ok = actual_tip == EXPECTED_TIP_VERTEX_BY_FINGER[finger]
        chain_results[finger] = {
            "raw_joint_chain": list(chain), "output_indices": list(expected_outputs[finger]),
            "expected_tip_vertex": EXPECTED_TIP_VERTEX_BY_FINGER[finger], "actual_tip_vertex": actual_tip,
            "same_parent_chain": bool(parent_ok), "same_finger_joint_sources": bool(source_ok),
            "same_finger_tip_connected": bool(tip_ok),
        }
    mapping = []
    for output_index, (kind, source) in enumerate(contract["final_sources"]):
        mapping.append({
            "output_index": output_index,
            "source": f"raw_joint[{source}]" if kind == "joint" else f"tip_vertex[{source}]",
            "official_visualization_label": OFFICIAL_21_LABELS[output_index],
        })
    return mapping, {"all_chains_pass": bool(all(item["same_parent_chain"] and item["same_finger_joint_sources"] and item["same_finger_tip_connected"] for item in chain_results.values())),
                     "chains": chain_results}


def skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))


def rodrigues(rotvecs: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvecs, dtype=np.float64).reshape(-1, 3)
    result = np.empty((len(values), 3, 3), dtype=np.float64)
    for i, vector in enumerate(values):
        angle = float(np.linalg.norm(vector))
        if angle < 1e-12:
            result[i] = np.eye(3)
        else:
            unit = vector / angle
            cross = skew(unit)
            result[i] = np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)
    return result


def transform(rot: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = translation
    return out


def decode_mano(model: dict, *, transl: np.ndarray, global_aa: np.ndarray, pose45: np.ndarray,
                betas: np.ndarray, add_hands_mean: bool, left_shapedirs_x_flip: bool,
                final_sources: tuple[tuple[str, int], ...]) -> dict:
    """Full MANO LBS in the documented asset convention; output is meters."""
    if pose45.shape != (45,) or global_aa.shape != (3,) or betas.shape != (10,) or transl.shape != (3,):
        raise ValueError("unexpected MANO input shape")
    shapedirs = model["shapedirs"].copy()
    if left_shapedirs_x_flip:
        shapedirs[:, 0, :] *= -1.0
    vertices_shaped = model["v_template"] + np.einsum("vck,k->vc", shapedirs, betas)
    joints_rest = np.asarray(model["J_regressor"].dot(vertices_shaped), dtype=np.float64)
    local_aa = pose45 + model["hands_mean"] if add_hands_mean else pose45.copy()
    local_rot = rodrigues(local_aa).reshape(15, 3, 3)
    global_rot = rodrigues(global_aa)[0]
    pose_feature = (local_rot - np.eye(3)).reshape(-1)
    vertices_posed = vertices_shaped + np.einsum("vcp,p->vc", model["posedirs"], pose_feature)

    parents = np.asarray(model["kintree_table"][0], dtype=np.int64).copy()
    parents[0] = -1
    transforms = [np.eye(4) for _ in range(16)]
    for joint in range(16):
        rot = global_rot if joint == 0 else local_rot[joint - 1]
        offset = joints_rest[joint] if parents[joint] < 0 else joints_rest[joint] - joints_rest[parents[joint]]
        local = transform(rot, offset)
        transforms[joint] = local if parents[joint] < 0 else transforms[parents[joint]] @ local
    skin = np.stack([transforms[i] @ transform(np.eye(3), -joints_rest[i]) for i in range(16)])
    homogeneous_vertices = np.c_[vertices_posed, np.ones(len(vertices_posed))]
    vertices = np.einsum("vi,ijk,vk->vj", model["weights"], skin, homogeneous_vertices)[:, :3] + transl
    # `transforms` is already in raw MANO kinematic-tree order.  Official
    # ManoLayer's earlier reorder only restores that order after it temporarily
    # concatenates breadth levels; applying it here would scramble the input to
    # the final 21-point visualization selector.
    joints16_raw = np.array([item[:3, 3] for item in transforms]) + transl
    tips = vertices[[source for kind, source in final_sources if kind == "tip_vertex"]]
    tip_by_vertex = {vertex: point for (kind, vertex), point in zip((source for source in final_sources if source[0] == "tip_vertex"), tips)}
    joints21 = np.array([joints16_raw[source] if kind == "joint" else tip_by_vertex[source] for kind, source in final_sources])
    return {"vertices_m": vertices, "joints16_raw_m": joints16_raw, "tips_m": tips,
            "joints21_m": joints21, "joints_rest_m": joints_rest,
            "local_aa_used": local_aa, "pose_feature_norm": float(np.linalg.norm(pose_feature))}


def normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError(f"degenerate calibration vector: {label}, norm={norm}")
    return vector / norm


def mano_palm_frame(joints21: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    wrist = joints21[0]
    index, middle, ring, pinky = (joints21[5], joints21[9], joints21[13], joints21[17])
    origin = (index + middle + ring + pinky) / 4.0
    x = normalize(middle - wrist, "MANO distal axis")
    y_raw = index - pinky
    z = normalize(np.cross(x, y_raw), "MANO palm normal")
    y = normalize(np.cross(z, x), "MANO radial axis after orthogonalisation")
    basis = np.column_stack((x, y, z))
    return origin, basis, {"wrist": wrist, "index_mcp": index, "middle_mcp": middle, "ring_mcp": ring, "pinky_mcp": pinky}


def parse_urdf(path: Path) -> dict:
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    edges = {}
    for element in root.findall("joint"):
        origin = element.find("origin")
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
        axis = element.find("axis")
        edges[element.attrib["name"]] = {
            "parent": element.find("parent").attrib["link"], "child": element.find("child").attrib["link"],
            "type": element.attrib["type"], "xyz": xyz, "rpy": rpy,
            "axis": np.fromstring(axis.attrib.get("xyz", "0 0 1") if axis is not None else "0 0 1", sep=" "),
        }
    return edges


def rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cx, sx, cy, sy, cz, sz = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array(((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
                     (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
                     (-sy, cy * sx, cy * cx)))


def sharpa_zero_frames(edges: dict) -> dict[str, np.ndarray]:
    pending = {name: edge for name, edge in edges.items() if edge["parent"] == "left_hand_C_MC" or edge["parent"].startswith("left_")}
    frames = {"left_hand_C_MC": np.eye(4)}
    while pending:
        completed = []
        for name, edge in pending.items():
            if edge["parent"] not in frames:
                continue
            frames[edge["child"]] = frames[edge["parent"]] @ transform(rpy_rotation(edge["rpy"]), edge["xyz"])
            completed.append(name)
        if not completed:
            raise RuntimeError(f"unresolved left URDF nodes: {sorted(pending)}")
        for name in completed:
            del pending[name]
    return frames


def sharpa_palm_frame(frames: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    root = frames["left_hand_C_MC"][:3, 3]
    index = frames["left_index_MCP_VL"][:3, 3]
    middle = frames["left_middle_MCP_VL"][:3, 3]
    ring = frames["left_ring_MCP_VL"][:3, 3]
    pinky = frames["left_pinky_MCP_VL"][:3, 3]
    origin = (index + middle + ring + pinky) / 4.0
    x = normalize(middle - root, "Sharpa distal axis")
    y_raw = index - pinky
    z = normalize(np.cross(x, y_raw), "Sharpa palm normal")
    y = normalize(np.cross(z, x), "Sharpa radial axis after orthogonalisation")
    basis = np.column_stack((x, y, z))
    return origin, basis, {"root": root, "index_mcp": index, "middle_mcp": middle, "ring_mcp": ring, "pinky_mcp": pinky}


def landmarks_mano(joints21: np.ndarray) -> dict[str, np.ndarray]:
    values = {}
    for finger, ids in MANO_OUTPUT.items():
        values[f"{finger}_tip"] = joints21[ids[-1]]
        # The public output names establish thumb_mcp/pip/dip, not the separate
        # CMC/MCP/IP anatomical interpretation requested for the proposed thumb
        # convention.  Do not silently select a thumb midpoint in this audit.
        if finger != "thumb":
            values[f"{finger}_middle"] = (joints21[ids[1]] + joints21[ids[2]]) / 2.0
    return values


def sharpa_segment_midpoint(frames: dict, edges: dict, finger: str) -> np.ndarray:
    if finger == "thumb":
        parent_link, outgoing = "left_thumb_PP", "left_thumb_IP"
    else:
        parent_link, outgoing = f"left_{finger}_MP", f"left_{finger}_DIP"
    edge = edges[outgoing]
    if edge["parent"] != parent_link:
        raise RuntimeError(f"unexpected Sharpa topology for {finger}: {outgoing}")
    return (frames[parent_link] @ np.r_[edge["xyz"] / 2.0, 1.0])[:3]


def landmarks_sharpa(frames: dict, edges: dict) -> dict[str, np.ndarray]:
    values = {}
    for finger in FINGERS:
        values[f"{finger}_tip"] = frames[f"left_{finger}_fingertip"][:3, 3]
        values[f"{finger}_middle"] = sharpa_segment_midpoint(frames, edges, finger)
    return values


def draw_sharpa_skeleton(ax, frames: dict, edges: dict) -> None:
    """Draw only zero-q URDF link-origin connectivity, never COM geometry."""
    for edge in edges.values():
        parent, child = edge["parent"], edge["child"]
        if parent not in frames or child not in frames:
            continue
        if not (parent == "left_hand_C_MC" or parent.startswith("left_")):
            continue
        start, end = frames[parent][:3, 3], frames[child][:3, 3]
        ax.plot((start[0], end[0]), (start[1], end[1]), (start[2], end[2]), color="0.55", linewidth=0.5, alpha=0.7)


def draw_axes(ax, origin: np.ndarray, basis: np.ndarray, prefix: str) -> None:
    for vector, color, label in zip(basis.T, ("r", "g", "b"), ("x", "y", "z")):
        endpoint = origin + vector * 0.025
        ax.plot((origin[0], endpoint[0]), (origin[1], endpoint[1]), (origin[2], endpoint[2]), color=color, linewidth=1.3)
        ax.text(*endpoint, f"{prefix}-{label}", fontsize=6, color=color)


def set_3d(ax, title: str) -> None:
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.set_zlabel("z (m)", fontsize=7)
    ax.view_init(elev=24, azim=-60)
    ax.set_box_aspect((1, 1, 1))


def set_equal_limits(axes: list, point_sets: list[np.ndarray]) -> None:
    """Use one physical-meter scale for every panel in one static figure."""
    points = np.concatenate([np.asarray(values, dtype=float).reshape(-1, 3) for values in point_sets])
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    half_span = max(float((upper - lower).max()) / 2.0, 0.03) * 1.12
    for axis in axes:
        axis.set_xlim(center[0] - half_span, center[0] + half_span)
        axis.set_ylim(center[1] - half_span, center[1] + half_span)
        axis.set_zlim(center[2] - half_span, center[2] + half_span)


def draw_hand(ax, joints: np.ndarray, *, title: str, landmark_only: bool = False) -> None:
    set_3d(ax, title)
    ax.scatter(*joints[0], color="black", s=14)
    ax.text(*joints[0], "J0 wrist", fontsize=5)
    for finger, ids in MANO_OUTPUT.items():
        chain = np.vstack((joints[0], joints[list(ids)]))
        ax.plot(chain[:, 0], chain[:, 1], chain[:, 2], color=COLORS[finger], linewidth=1.5)
        ax.scatter(chain[1:, 0], chain[1:, 1], chain[1:, 2], color=COLORS[finger], s=10)
        for index in ids[:-1]:
            ax.text(*joints[index], f"J{index}", fontsize=5)
        ax.text(*joints[ids[-1]], f"{finger} J{ids[-1]}", fontsize=5)
        if landmark_only:
            if finger != "thumb":
                mid = (joints[ids[1]] + joints[ids[2]]) / 2.0
                ax.scatter(*mid, color=COLORS[finger], marker="^", s=28)


def draw_calibration(ax, mano_points: dict, sharpa_points: dict, mano_origin: np.ndarray, mano_basis: np.ndarray,
                     sharpa_origin: np.ndarray, sharpa_basis: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> None:
    set_3d(ax, "PROPOSED fixed local calibration; s=1 m")
    for name in mano_points:
        finger = name.split("_")[0]
        mapped = rotation @ mano_points[name] + translation
        target = sharpa_points[name]
        ax.scatter(*mapped, color=COLORS[finger], marker="o", s=26)
        ax.scatter(*target, color=COLORS[finger], marker="x", s=30)
        ax.plot((mapped[0], target[0]), (mapped[1], target[1]), (mapped[2], target[2]), color=COLORS[finger], alpha=0.5, linewidth=0.8)
        ax.text(*mapped, name, fontsize=5)
    draw_axes(ax, sharpa_origin, sharpa_basis, "S")
    ax.text2D(0.02, 0.98, "o = A(MANO candidate); x = zero-q Sharpa candidate; connectors are not errors", transform=ax.transAxes, fontsize=6, va="top")


def check_calibration(rotation: np.ndarray, translation: np.ndarray, points: dict[str, np.ndarray]) -> dict:
    inv = rotation.T
    round_trip = [np.linalg.norm(inv @ (rotation @ point + translation - translation) - point) for point in points.values()]
    return {"orthogonality_fro": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
            "determinant": float(np.linalg.det(rotation)), "round_trip_max_m": float(max(round_trip)),
            "scale": 1.0, "reflection": bool(np.linalg.det(rotation) < 0.0)}


def serialize_array(value: np.ndarray) -> list:
    return np.asarray(value, dtype=float).round(9).tolist()


def max_pair_distance(points: list[np.ndarray]) -> float:
    values = np.asarray(points, dtype=float)
    return float(max(np.linalg.norm(a - b) for a in values for b in values))


def decoder_smoke_cases(model: dict, final_sources: tuple[tuple[str, int], ...], variants: list[dict], raw: np.lib.npyio.NpzFile,
                        selected: tuple[int, ...]) -> list[dict]:
    """Exercise fixed inputs; this is not a trusted-layer equivalence test."""
    variant = variants[0]
    shared = {"add_hands_mean": bool(variant["add_hands_mean"]),
              "left_shapedirs_x_flip": bool(variant["left_shapedirs_x_flip"]), "final_sources": final_sources}
    cases = [
        ("neutral", np.zeros(3), np.zeros(3), np.zeros(45), np.zeros(10)),
        ("index_mcp_local_rotation", np.zeros(3), np.zeros(3), np.r_[np.array([0.0, 0.0, 0.35]), np.zeros(42)], np.zeros(10)),
        ("nonzero_betas", np.zeros(3), np.zeros(3), np.zeros(45), np.array([0.5, -0.25] + [0.0] * 8)),
        ("global_rotation_translation", np.array([0.02, -0.03, 0.04]), np.array([0.2, -0.1, 0.15]), np.zeros(45), np.zeros(10)),
    ]
    cases.extend((f"real_frame_{frame}", np.asarray(raw["left_mano_trans"][frame], dtype=float),
                  np.asarray(raw["left_mano_rot"][frame], dtype=float), np.asarray(raw["left_mano_pose45"][frame], dtype=float),
                  np.asarray(raw["left_mano_betas"][frame], dtype=float)) for frame in selected)
    records = []
    for name, translation, root, local, betas in cases:
        result = decode_mano(model, transl=translation, global_aa=root, pose45=local, betas=betas, **shared)
        records.append({"name": name, "finite_vertices_778": bool(np.isfinite(result["vertices_m"]).all()),
                        "finite_joints16": bool(np.isfinite(result["joints16_raw_m"]).all()),
                        "finite_landmarks21": bool(np.isfinite(result["joints21_m"]).all())})
    return records


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo / "configs/m1a_left_landmark_calibration_candidate_v1.json")
    parser.add_argument("--trusted-runtime-evidence", type=Path,
                        default=repo / "artifacts/m1a_trusted_layer_validation_v4/trusted_layer_runtime_evidence.json")
    parser.add_argument("--out-dir", type=Path, default=repo / "artifacts/m1a_full_mano_decode_audit_v4")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    trusted_runtime = json.loads(args.trusted_runtime_evidence.read_text(encoding="utf-8"))
    trusted_layer_validation = {
        **trusted_runtime["trusted_layer_validation"],
        "comparison_status": trusted_runtime["comparison_status"],
        "comparison_result": trusted_runtime["comparison_result"],
        "local_environment_checks": trusted_runtime["local_environment_checks"],
    }
    paths = {name: Path(value) for name, value in config["paths"].items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(paths["mano"])
    output_contract = official_output_contract(paths["mano_layer_source"], model["kintree_table"])
    output_mapping, structure_checks = output_mapping_and_structure(output_contract)
    if not structure_checks["all_chains_pass"]:
        raise RuntimeError("official 21-point output does not pass raw kinematic-chain checks")
    raw = np.load(paths["perception"], allow_pickle=True)
    edges = parse_urdf(paths["urdf"])
    sharpa_frames = sharpa_zero_frames(edges)
    valid_indices = np.flatnonzero(np.asarray(raw["left_mano_valid"], dtype=bool))
    if not len(valid_indices):
        raise RuntimeError("no valid left MANO frame")
    # Predeclared, result-independent selection: first, median-valid, last valid source frame.
    selected = (int(valid_indices[0]), int(valid_indices[len(valid_indices) // 2]), int(valid_indices[-1]))
    variants = config["decode_variants"]
    decoded: dict[str, dict[int, dict]] = {}
    for variant in variants:
        decoded[variant["id"]] = {}
        for frame in selected:
            decoded[variant["id"]][frame] = decode_mano(
                model, transl=np.asarray(raw["left_mano_trans"][frame], dtype=float),
                global_aa=np.asarray(raw["left_mano_rot"][frame], dtype=float),
                pose45=np.asarray(raw["left_mano_pose45"][frame], dtype=float),
                betas=np.asarray(raw["left_mano_betas"][frame], dtype=float),
                add_hands_mean=bool(variant["add_hands_mean"]),
                left_shapedirs_x_flip=bool(variant["left_shapedirs_x_flip"]),
                final_sources=output_contract["final_sources"],
            )

    # These validate that this candidate decoder uses each field once.  They do
    # not prove that either candidate matches the historical HaWoR layer.
    check_frame = selected[0]
    check_variant = variants[0]
    check_kwargs = {
        "global_aa": np.asarray(raw["left_mano_rot"][check_frame], dtype=float),
        "pose45": np.asarray(raw["left_mano_pose45"][check_frame], dtype=float),
        "betas": np.asarray(raw["left_mano_betas"][check_frame], dtype=float),
        "add_hands_mean": bool(check_variant["add_hands_mean"]),
        "left_shapedirs_x_flip": bool(check_variant["left_shapedirs_x_flip"]),
    }
    input_translation = np.asarray(raw["left_mano_trans"][check_frame], dtype=float)
    with_translation = decode_mano(model, transl=input_translation, final_sources=output_contract["final_sources"], **check_kwargs)
    without_translation = decode_mano(model, transl=np.zeros(3), final_sources=output_contract["final_sources"], **check_kwargs)
    zero_shape = decode_mano(model, transl=input_translation, global_aa=check_kwargs["global_aa"],
                             pose45=check_kwargs["pose45"], betas=np.zeros(10),
                             add_hands_mean=check_kwargs["add_hands_mean"],
                             left_shapedirs_x_flip=check_kwargs["left_shapedirs_x_flip"],
                             final_sources=output_contract["final_sources"])
    decoder_checks = {
        "all_selected_outputs_finite": bool(all(np.isfinite(item["joints21_m"]).all() and np.isfinite(item["vertices_m"]).all()
                                                  for variant in decoded.values() for item in variant.values())),
        "translation_applied_once_max_abs_m": float(np.max(np.abs((with_translation["joints21_m"] - without_translation["joints21_m"]) - input_translation))),
        "nonzero_beta_changes_vertices_l2_m": float(np.linalg.norm(with_translation["vertices_m"] - zero_shape["vertices_m"])),
        "global_rotation_input_norm_rad": float(np.linalg.norm(check_kwargs["global_aa"])),
        "local_pose_input_norm_rad": float(np.linalg.norm(check_kwargs["pose45"])),
        "note": "Field-use and finiteness checks only; not proof of equivalence to a trusted MANO layer or either historical mean-pose candidate.",
    }
    smoke_records = decoder_smoke_cases(model, output_contract["final_sources"], variants, raw, selected)

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    real_axes = []
    for row, frame in enumerate(selected):
        for col, variant in enumerate(variants):
            result = decoded[variant["id"]][frame]
            axis = fig.add_subplot(3, 2, row * 2 + col + 1, projection="3d")
            draw_hand(axis, result["joints21_m"],
                      title=f"{variant['id']} — source frame {frame}\nmean={variant['add_hands_mean']}; x-shape-flip={variant['left_shapedirs_x_flip']}", landmark_only=True)
            real_axes.append(axis)
    set_equal_limits(real_axes, [item["joints21_m"] for variant in decoded.values() for item in variant.values()])
    fig.suptitle("FULL MANO LBS — corrected official 21-point order; historical mean-pose semantics UNKNOWN", fontsize=11)
    real_figure = args.out_dir / "m1a_full_mano_real_frames_candidates.png"
    fig.savefig(real_figure, dpi=220)
    plt.close(fig)

    calibration_variant = next(item for item in variants if item["id"] == config["calibration_variant"])
    neutral = decode_mano(model, transl=np.zeros(3), global_aa=np.zeros(3), pose45=np.zeros(45), betas=np.zeros(10),
                          add_hands_mean=bool(calibration_variant["add_hands_mean"]),
                          left_shapedirs_x_flip=bool(calibration_variant["left_shapedirs_x_flip"]),
                          final_sources=output_contract["final_sources"])
    mano_origin, mano_basis, mano_anchors = mano_palm_frame(neutral["joints21_m"])
    sharpa_origin, sharpa_basis, sharpa_anchors = sharpa_palm_frame(sharpa_frames)
    rotation = sharpa_basis @ mano_basis.T
    translation = sharpa_origin - rotation @ mano_origin
    mano_landmarks = landmarks_mano(neutral["joints21_m"])
    sharpa_landmarks = landmarks_sharpa(sharpa_frames, edges)
    checks = check_calibration(rotation, translation, mano_landmarks)

    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    ax_mano = fig.add_subplot(1, 3, 1, projection="3d")
    draw_hand(ax_mano, neutral["joints21_m"], title=f"MANO canonical ({calibration_variant['id']})", landmark_only=True)
    draw_axes(ax_mano, mano_origin, mano_basis, "H")
    ax_sharpa = fig.add_subplot(1, 3, 2, projection="3d")
    set_3d(ax_sharpa, "Sharpa left zero-q candidate")
    draw_sharpa_skeleton(ax_sharpa, sharpa_frames, edges)
    for name, point in sharpa_landmarks.items():
        finger = name.split("_")[0]
        ax_sharpa.scatter(*point, color=COLORS[finger], marker="*" if name.endswith("tip") else "^", s=30)
        label = "thumb_middle target only; MANO UNKNOWN" if name == "thumb_middle" else name
        ax_sharpa.text(*point, label, fontsize=5)
    draw_axes(ax_sharpa, sharpa_origin, sharpa_basis, "S")
    draw_calibration(fig.add_subplot(1, 3, 3, projection="3d"), mano_landmarks, sharpa_landmarks, mano_origin, mano_basis,
                     sharpa_origin, sharpa_basis, rotation, translation)
    set_equal_limits(fig.axes, [neutral["joints21_m"], np.asarray(list(sharpa_landmarks.values())),
                                np.asarray([rotation @ point + translation for point in mano_landmarks.values()])])
    fig.suptitle("PROPOSED anatomical palm calibration — fixed canonical poses, no fit, s=1 m", fontsize=10)
    calibration_figure = args.out_dir / "m1a_proposed_local_calibration.png"
    fig.savefig(calibration_figure, dpi=240)
    plt.close(fig)

    frame_records = []
    for frame in selected:
        entry = {"source_frame": frame,
                 "input": {name: serialize_array(raw[name][frame]) for name in ("left_mano_trans", "left_mano_rot", "left_mano_pose45", "left_mano_betas")},
                 "variants": {}}
        for variant in variants:
            value = decoded[variant["id"]][frame]
            entry["variants"][variant["id"]] = {"joints21_m": serialize_array(value["joints21_m"]),
                                                    "pose_feature_norm": value["pose_feature_norm"]}
        frame_records.append(entry)
    report = {
        "status": {"full_mano_decode": f"Corrected output order executed; trusted-layer equivalence {trusted_runtime['comparison_status']}; historical mean-pose/layer semantics UNKNOWN",
                   "landmark_convention": "Long-finger convention PROPOSED; thumb midpoint UNRESOLVED; not original-author mapping",
                   "m1a_retargeting": "NOT TESTED"},
        "source_hashes": {name: sha256(path) for name, path in paths.items()},
        "candidate_config_sha256": sha256(args.config),
        "trusted_runtime_evidence_sha256": sha256(args.trusted_runtime_evidence),
        "diagnostic_source_sha256": sha256(Path(__file__)),
        "runtime": {"python": sys.version, "numpy": np.__version__},
        "source_frame_selection": {"rule": "first, median-valid, last valid index; fixed before decode", "selected": selected},
        "decode_variants": variants,
        "official_output_order": {"mapping": output_mapping, "structure_checks": structure_checks,
                                    "source_contract": {key: value for key, value in output_contract.items() if key not in {"parents", "final_sources"}}},
        "full_decoder": {"shape_blend": True, "pose_blend": True, "linear_blend_skinning": True,
                         "source_equivalence_scope": "Numerical reconstruction; trusted-layer availability and comparison status are loaded from versioned runtime evidence, not inferred from local package checks.",
                         "explicit_parameters": {"root_rotation": "axis-angle -> Rodrigues", "local_pose": "15 axis-angle rotations -> Rodrigues",
                                                 "pca": "not used", "mean_pose": "separate false/true PROPOSED variants", "translation": "added once after LBS",
                                                 "betas": "applied once through shapedirs", "left_shapedirs_x_flip": "per candidate config"},
                         "field_use_checks": decoder_checks, "smoke_cases": smoke_records},
        "trusted_layer_validation": trusted_layer_validation,
        "trusted_layer_comparison": {"status": trusted_runtime["comparison_status"],
                                     "pre_registered_tolerances_m": config["independent_layer_comparison"]["pre_registered_tolerances_m"],
                                     "basis": config["independent_layer_comparison"]["basis"],
                                     "results": trusted_runtime["comparison_result"]},
        "candidate_landmarks": {name: {"mano_expression": ("tip output index" if name.endswith("tip") else "PIP-DIP midpoint"),
                                         "mano_canonical_m": serialize_array(mano_landmarks[name]),
                                         "mapped_mano_canonical_m": serialize_array(rotation @ mano_landmarks[name] + translation),
                                         "sharpa_canonical_m": serialize_array(sharpa_landmarks[name]),
                                         "unfitted_static_separation_m": float(np.linalg.norm(rotation @ mano_landmarks[name] + translation - sharpa_landmarks[name]))}
                                for name in mano_landmarks},
        "unresolved_thumb_middle": {"status": "UNKNOWN", "public_output_labels": {"J1": "thumb_mcp", "J2": "thumb_pip", "J3": "thumb_dip"},
                                      "reason": "The public visualization labels do not establish the distinct CMC/MCP/IP anatomy required to choose (J1,J2) versus (J2,J3). No thumb MANO midpoint is emitted."},
        "model_size_comparison": {"metric": "maximum pairwise distance among the five tip landmarks; descriptive only, no scale fit",
                                   "mano_tip_span_m": max_pair_distance([mano_landmarks[f"{finger}_tip"] for finger in FINGERS]),
                                   "sharpa_tip_span_m": max_pair_distance([sharpa_landmarks[f"{finger}_tip"] for finger in FINGERS]),
                                   "applied_scale": 1.0},
        "proposed_calibration": {"canonical_pose": "MANO beta=0, global_aa=0, pose45=0; Sharpa q=0",
                                   "mano_origin_m": serialize_array(mano_origin), "mano_basis_columns": serialize_array(mano_basis),
                                   "sharpa_origin_m": serialize_array(sharpa_origin), "sharpa_basis_columns": serialize_array(sharpa_basis),
                                   "R_S_from_H": serialize_array(rotation), "t_S_from_H_m": serialize_array(translation),
                                   "formula": "A(p_H)=R_S_from_H @ (1.0*p_H) + t_S_from_H_m", "checks": checks,
                                   "anchors": {"mano": {key: serialize_array(value) for key, value in mano_anchors.items()},
                                               "sharpa": {key: serialize_array(value) for key, value in sharpa_anchors.items()}}},
        "selected_real_frames": frame_records,
        "figures": {"real_frames": str(real_figure), "calibration": str(calibration_figure)},
    }
    report_path = args.out_dir / "m1a_full_mano_decode_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(real_figure)
    print(calibration_figure)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
