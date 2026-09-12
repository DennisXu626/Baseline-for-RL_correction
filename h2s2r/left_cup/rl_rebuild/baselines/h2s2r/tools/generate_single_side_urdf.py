"""Generate a kinematics-only Vega + Sharpa URDF for FABRICS.

The Isaac scene continues to use the complete dual-arm robot asset. FABRICS only
supports fixed/revolute kinematic trees, so this tool extracts one arm and hand
from the same source URDF and adds the seven palm helper frames expected by the
official H2S2R pose fabric.
"""

from __future__ import annotations

import argparse
import copy
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from ..contract import ControlledSide, FABRIC_DOF_DIM, joint_names_for_side


PALM_FRAME = "h2s2r_palm"
PALM_HELPERS = {
    "h2s2r_palm_x": (0.25, 0.0, 0.0),
    "h2s2r_palm_x_neg": (-0.25, 0.0, 0.0),
    "h2s2r_palm_y": (0.0, 0.25, 0.0),
    "h2s2r_palm_y_neg": (0.0, -0.25, 0.0),
    "h2s2r_palm_z": (0.0, 0.0, 0.25),
    "h2s2r_palm_z_neg": (0.0, 0.0, -0.25),
}


def _clone_link_without_geometry(link: ET.Element) -> ET.Element:
    result = copy.deepcopy(link)
    for child in list(result):
        if child.tag in {"visual", "collision"}:
            result.remove(child)
    return result


def _append_fixed_joint(
    robot: ET.Element,
    *,
    name: str,
    parent: str,
    child: str,
    xyz: tuple[float, float, float],
) -> None:
    robot.append(ET.Element("link", {"name": child}))
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(
        joint,
        "origin",
        {"xyz": " ".join(f"{value:.8g}" for value in xyz), "rpy": "0 0 0"},
    )


def extract_single_side(source: str | Path, side: ControlledSide | str) -> ET.ElementTree:
    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    source_root = ET.parse(source).getroot()
    links = {link.attrib["name"]: link for link in source_root.findall("link")}
    joints_by_parent: dict[str, list[ET.Element]] = defaultdict(list)
    for joint in source_root.findall("joint"):
        joints_by_parent[joint.find("parent").attrib["link"]].append(joint)

    first_joint_name = f"{side.arm_prefix}_arm_j1"
    first_joint = next(
        (
            joint
            for joint in joints_by_parent["arm_center"]
            if joint.attrib["name"] == first_joint_name
        ),
        None,
    )
    if first_joint is None:
        raise ValueError(f"source URDF has no {first_joint_name}")

    selected_joints: list[ET.Element] = []
    selected_links = {"arm_center"}

    def visit(joint: ET.Element) -> None:
        selected_joints.append(joint)
        child = joint.find("child").attrib["link"]
        selected_links.add(child)
        for descendant in joints_by_parent.get(child, ()):
            visit(descendant)

    visit(first_joint)

    output = ET.Element(
        "robot",
        {"name": f"vega_sharpa_{side.value}_h2s2r_fabric"},
    )
    # Keep root first, then append each child link directly after its tree joint.
    output.append(_clone_link_without_geometry(links["arm_center"]))
    appended_links = {"arm_center"}
    for joint in selected_joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        if parent not in appended_links:
            raise RuntimeError(f"joint tree is not topological at {joint.attrib['name']}")
        output.append(copy.deepcopy(joint))
        output.append(_clone_link_without_geometry(links[child]))
        appended_links.add(child)

    hand_root = f"{side.value}_hand_C_MC"
    _append_fixed_joint(
        output,
        name="h2s2r_palm_joint",
        parent=hand_root,
        child=PALM_FRAME,
        xyz=(0.0, 0.0, 0.0),
    )
    for frame, xyz in PALM_HELPERS.items():
        _append_fixed_joint(
            output,
            name=f"{frame}_joint",
            parent=PALM_FRAME,
            child=frame,
            xyz=xyz,
        )

    tree = ET.ElementTree(output)
    validate_generated_urdf(tree, side)
    return tree


def validate_generated_urdf(tree: ET.ElementTree, side: ControlledSide | str) -> None:
    side = side if isinstance(side, ControlledSide) else ControlledSide.parse(side)
    root = tree.getroot()
    joints = root.findall("joint")
    active = [joint for joint in joints if joint.attrib["type"] == "revolute"]
    names = tuple(joint.attrib["name"] for joint in active)
    expected = joint_names_for_side(side)
    if names != expected:
        raise ValueError(
            "generated revolute-joint order differs from the adapter contract:\n"
            f"expected={expected}\nactual={names}"
        )
    if len(active) != FABRIC_DOF_DIM:
        raise ValueError(f"expected {FABRIC_DOF_DIM} revolute joints, got {len(active)}")
    unsupported = [
        joint.attrib["name"]
        for joint in joints
        if joint.attrib["type"] not in {"fixed", "revolute"}
    ]
    if unsupported:
        raise ValueError(f"FABRICS-unsupported joints remain: {unsupported}")
    link_names = {link.attrib["name"] for link in root.findall("link")}
    required_frames = {PALM_FRAME, *PALM_HELPERS}
    missing = required_frames - link_names
    if missing:
        raise ValueError(f"missing palm helper frames: {sorted(missing)}")


def write_single_side_urdf(
    source: str | Path,
    destination: str | Path,
    side: ControlledSide | str,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree = extract_single_side(source, side)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    args = parser.parse_args()
    sides = (
        (ControlledSide.LEFT, ControlledSide.RIGHT)
        if args.side == "both"
        else (ControlledSide.parse(args.side),)
    )
    for side in sides:
        destination = args.output_dir / f"vega_sharpa_{side.value}_fabric.urdf"
        write_single_side_urdf(args.source, destination, side)
        print(destination)


if __name__ == "__main__":
    main()

