"""Bounded WP2 source-formula and independent-slot regression checks."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

from .geometry import sha256, quat_to_matrix_wxyz, matrix_to_quat_wxyz, axis_angle_matrix
from .reward import object_mimic_bimanual
from .curriculum import CurriculumState
from .batch_eval import load_evaluator, criteria_overlay, Slots


def source_class(upstream):
    """Execute the pinned methods themselves, mocking only simulator access."""
    from pyquaternion import Quaternion
    path = upstream / "hand_imitation/env/models/rewards.py"
    tree = ast.parse(path.read_text())
    source = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ObjectMimic")
    source.bases = []
    norm = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "norm2")
    util = upstream / "hand_imitation/env/models/motion_util.py"
    functions = [n for n in ast.parse(util.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ("to_quat", "rotation_distance")]
    module = ast.fix_missing_locations(ast.Module(body=[*functions, norm, source], type_ignores=[]))
    namespace = {"np": np, "Quaternion": Quaternion}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["ObjectMimic"], {str(path): sha256(path), str(util): sha256(util)}


def reward_comparison(upstream):
    cls, sources = source_class(upstream)
    cases = [
        dict(name="pregrasp_palm_only", phase=False, palm=.06),
        dict(name="pregrasp_tips", phase=False, tip=.02),
        dict(name="no_contact", hand=False),
        dict(name="non_tip_hand_contact", hand=True),
        *[dict(name=f"finger_{i}", hand=True, contacts=[i]) for i in range(5)],
        dict(name="all_fingers", hand=True, contacts=list(range(5))),
        dict(name="target_lift_only", hand=True, target_z=.03),
        dict(name="actual_lift_only", hand=True, actual_z=.03),
        dict(name="both_lift", hand=True, target_z=.03, actual_z=.03),
        dict(name="table_blocks_lift", hand=True, target_z=.03, actual_z=.03, table=True),
        dict(name="hand_bonus_outside_gate", hand=True, object_dx=.02, tip=.09),
        dict(name="hand_bonus_inside_gate", hand=True, object_dx=.005, tip=.09),
        dict(name="orientation", hand=True, angle=.7),
        dict(name="imitate_success_branch", hand=True, imitate=True),
    ]
    results = []
    for c in cases:
        s = cls()
        for key, value in s.DEFAULT_HPARAMS.items():
            setattr(s, key, value)
        s.pregrasp_steps, s.start_step, s._mode, s.stage = 40, 0, "train", 0
        s.imitate_steps, s.manipulation_steps = 150, 190
        s.pregrasp_success, s.imitate_success = c.get("phase", True), c.get("imitate", False)
        s.relocate, s.obj_reward_start, s.floor_z, s._lift_z = True, 0, 1., 1.02
        actual = np.zeros((6, 3), dtype=np.float32)
        actual[0, 0] = c.get("palm", 0.)
        actual[1:, 0] = c.get("tip", 0.)
        target = np.zeros((6, 3), dtype=np.float32)
        op = np.array([c.get("object_dx", 0), 0., 1. + c.get("actual_z", 0)])
        tp = np.array([0., 0., 1. + c.get("target_z", 0)])
        oq = np.array([1., 0., 0., 0.])
        tq = matrix_to_quat_wxyz(axis_angle_matrix(np.array([c.get("angle", 0.), 0., 0.])))
        s._object_name = "object"
        s._body_names = ["palm", "thumb", "index", "middle", "ring", "pinky"]
        s.robot_geom_names = [f"g{i}" for i in range(20)]
        s.object_geom_names = ["object_geom"]
        active = set(c.get("contacts", []))
        tip_lookup = {19: 0, 6: 1, 9: 2, 12: 3, 16: 4}
        def contacts(_, g1, g2):
            if g1 is s.robot_geom_names:
                return c.get("hand", False)
            if g1 is s.object_geom_names:
                return c.get("table", False)
            return tip_lookup.get(s.robot_geom_names.index(g1), -1) in active
        s.check_contacts = contacts
        tracker = NS(append=lambda value: None, reset=lambda: None, error=0., success=0., success_goal=False)
        s._obj = s._hand = tracker
        s._reference_motion = NS(final_goal=tp, object_pos=tp, object_rot=tq, hand_jnts=target, time=.1, length=100)
        s.retarget_robot_jpos = np.repeat(target[None], 41, axis=0)
        physics = NS(data=NS(qpos=np.zeros(30)), named=NS(data=NS(xipos={"object": op}, xquat={"object": oq}, xpos=dict(zip(s._body_names, actual)))))
        expected, _ = s.get_reward(physics, 2)
        def tensor(x):
            return torch.tensor(np.asarray(x), dtype=torch.float64)
        port_inputs = dict(actual_hand_points=tensor(np.tile(actual, (1, 2, 1, 1))),
            target_hand_points=tensor(np.tile(target, (1, 2, 1, 1))),
            actual_object_pose=tensor(np.tile(np.r_[op, oq], (1, 2, 1))),
            target_object_pose=tensor(np.tile(np.r_[tp, tq], (1, 2, 1))),
            fingertip_contacts=torch.tensor([[[i in active for i in range(5)]] * 2]),
            hand_object_contacts=torch.tensor([[c.get("hand", False)] * 2]),
            object_table_contacts=torch.tensor([[c.get("table", False)] * 2]),
            pregrasp_success=torch.tensor([c.get("phase", True)]), initial_object_z=tensor([[1., 1.]]))
        got, terms = object_mimic_bimanual(**port_inputs)
        error = abs(float(got[0]) - expected)
        results.append({"case": c, "source": float(expected), "port": float(got[0]), "abs_error": error,
                        "port_inputs": {k: v.tolist() for k, v in port_inputs.items()}})
        assert error < 1e-6, results[-1]
        # Termination branch compares source itself against the registered
        # dual-side combination, not simulator-derived task outcomes.
        for step in (1, 39, 40):
            s._step_count = step
            src_term = s.check_termination(physics)
            if not s.pregrasp_success:
                expected_term = step >= 40 or c.get("hand", False)
            elif s.imitate_success:
                expected_term = step >= 190 or not c.get("hand", False)
            else:
                expected_term = np.linalg.norm(op-tp) >= .25 or step >= 150
            assert src_term == expected_term
    return {"status": "PASS", "source_hashes": sources, "rows": results,
            "termination_checks": len(results) * 3, "scope": "unmodified pinned reward AST; only physical observations/contact queries mocked"}


def slot_comparison(bundle, reference, output):
    e = load_evaluator(bundle)
    tape, mb, mc = criteria_overlay(bundle, reference, output)
    batch = Slots(e, tape, mb, mc, [10, 11, 12])
    scalar = [e.PourProgress(str(tape), mb, mc, mouth_gate=e.MOUTH_GATE) for _ in range(3)]
    comparisons, first_gates, negative_fail = 0, {}, None
    fixed_trace = []
    for step in range(903):
        infos = []
        for i, p in enumerate(scalar):
            o0, o1 = p.rest[0].copy(), p.rest[1].copy()
            # Genuine state-machine inputs; no gate/clock presets. Slot0 walks
            # G1/G2/G3/placed/G4, slot1 never contacts, slot2 drops below table.
            pads = i == 0
            if i == 0 and p.cert_phase:
                for obj in (o0, o1):
                    obj[2] += .015 * e.cert_alpha(p.cert_phase, p.cert_t)
            if i == 0 and p.g[2] and not p.g[3]:
                # Tilt120deg, away from the exactly90deg floating-point boundary.
                rr = axis_angle_matrix(np.array([7 * np.pi / 6, 0., 0.]))
                o1[3:] = matrix_to_quat_wxyz(rr)
                o1[:3] = o0[:3] + quat_to_matrix_wxyz(o0[3:]) @ mc - rr @ mb
            if i == 2:
                o0[2] = .80
            infos.append({"object_0_pose": o0, "object_1_pose": o1,
                "arm_q_right": p.stance["right"].copy(), "arm_q_left": p.stance["left"].copy(),
                "pads3": pads, "wrist_right": o1[:3] + [.1, 0., 0.], "wrist_left": o0[:3] + [.1, 0., 0.]})
        was_finished = batch.finished.copy()
        fixed_trace.append([{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()} for info in infos])
        outputs = batch.step(infos)
        for i, (p, info) in enumerate(zip(scalar, infos)):
            if was_finished[i]:
                continue
            out = p.step(info["object_0_pose"], info["object_1_pose"], info["arm_q_right"], info["arm_q_left"], info["pads3"], info["wrist_right"], info["wrist_left"])
            assert out == outputs[i], (step, i, out, outputs[i])
            assert p.g == batch.progress[i].g
            assert p.cert_phase == batch.progress[i].cert_phase and p.cert_t == batch.progress[i].cert_t
            assert set(e.classify_failure(p, info["object_0_pose"], info["object_1_pose"])) <= batch.deadlines[i]
            comparisons += 1
            if i == 0:
                for gate, reached in p.g.items():
                    if reached:
                        first_gates.setdefault(str(gate), step + 1)
            if i == 2 and out["fail"]:
                negative_fail = out["fail"]
    assert scalar[0].g[4] and not scalar[1].g[1], {"first_gates": first_gates, "results": batch.results,
        "cert_try": scalar[0].cert_try, "m2_run": scalar[0].m2_run, "m3_run": scalar[0].m3_run}
    assert batch.results[1]["fail_reason_progress"] == "timeout"
    assert negative_fail and "D1" in negative_fail
    trace_path = output / "criteria_fixed_trace.json"
    trace_path.write_text(json.dumps(fixed_trace) + "\n")
    return {"status": "PASS", "synthetic_only_not_SR": True, "scalar_step_comparisons": comparisons,
            "trace_sha256": sha256(trace_path),
            "first_positive_gate_steps": first_gates, "results": batch.results,
            "independent_progress_object_count": len({id(p) for p in batch.progress}),
            "trace_rules": ["own canonical objects + physical certification rise", "no contacts903", "drop below table"]}


def main():
    ap = argparse.ArgumentParser()
    for key in ("upstream", "bundle-root", "reference", "output-dir"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--port-fixture", type=Path)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if a.port_fixture:
        fixture = json.loads(a.port_fixture.read_text())
        comparisons = []
        for row in fixture["rows"]:
            inputs = {k: torch.tensor(v, dtype=torch.bool if k.endswith("contacts") or k == "pregrasp_success" else torch.float64)
                      for k, v in row["port_inputs"].items()}
            got = float(object_mimic_bimanual(**inputs)[0][0])
            error = abs(got - row["source"])
            assert error < 1e-6
            comparisons.append({"case": row["case"]["name"], "abs_error": error})
        (a.output_dir / "runtime_torch_reward_comparison.json").write_text(json.dumps({"status": "PASS", "torch": torch.__version__,
            "source_fixture_sha256": sha256(a.port_fixture), "rows": comparisons}, indent=2) + "\n")
        return
    reward = reward_comparison(a.upstream)
    (a.output_dir / "reward_source_comparison.json").write_text(json.dumps(reward, indent=2) + "\n")
    batch = slot_comparison(a.bundle_root, a.reference, a.output_dir)
    (a.output_dir / "batch_eval_validation.json").write_text(json.dumps(batch, indent=2) + "\n")
    c = CurriculumState()
    assert not c.update([1.] * 23 + [0.] * 2)
    assert c.update([1.] * 24 + [0.]) and c.stage == 1
    assert c.update([1.] * 25) and c.stage == 2
    assert not c.update([1.] * 25)
    (a.output_dir / "curriculum_branch_test.json").write_text(json.dumps({"status": "PASS", "synthetic_only": True, "no_live_promotion_claim": True}) + "\n")
    print("WP2 CPU source, batch-slot, curriculum checks PASS", flush=True)


if __name__ == "__main__":
    main()
