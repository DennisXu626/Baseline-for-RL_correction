"""CPU behavior tests call the exact support functions used by the D replay."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from camera_delivery import CAMERAS, ThreeView, camera_config, validate_camera_config
from contact_capture import float_close, unpack_six
from runtime_support import (AtomicSubstepJournal, StateWriteRecorder,
                             close_media_then_app, jsonable, process_verdict,
                             quat_angle, register_observation_sensor,
                             route_fixed_target, sha256_file,
                             supervise_process)


B = Path(__file__).resolve().parent
tests = {}

# Frozen commands and the production routing helper.
frozen = np.load(B / "command_inputs.npz")
tests["frozen_100x29"] = frozen["targets_rad"].shape == (100, 29)
tests["trace0_not_command"] = not np.array_equal(frozen["historical_initial_q_rad"], frozen["targets_rad"][0])
names = [f"joint_{i}" for i in range(58)]
left_ids = torch.arange(0, 29)
right_ids = torch.arange(29, 58)
current = torch.arange(58, dtype=torch.float32).reshape(1, 58)
left = torch.full((1, 29), 7.0)
right = torch.full((1, 29), -3.0)
routed = route_fixed_target(current_q=current, left_target=left, left_ids=left_ids,
                            right_ids=right_ids, right_reset_q=right,
                            joint_names=names, expected_left_names=names[:29])
tests["real_route_left_right"] = torch.equal(routed[:, :29], left) and torch.equal(routed[:, 29:], right)
routed_1d = route_fixed_target(current_q=current, left_target=left, left_ids=left_ids,
                               right_ids=right_ids, right_reset_q=right[0],
                               joint_names=names, expected_left_names=names[:29])
tests["real_route_accepts_runtime_1d_right"] = torch.equal(routed_1d[:, 29:], right)


class _FakeScene:
    def __init__(self):
        self.sensors = {}


created = []
collection = []
sensor_names = []
scene = _FakeScene()


def _fake_sensor_factory(cfg):
    sensor = {"cfg": cfg}
    created.append(sensor)
    return sensor


registered = register_observation_sensor(
    sensor_factory=_fake_sensor_factory, cfg={"pair": "cup_to_table"},
    collection=collection, names=sensor_names, scene_sensors=scene.sensors,
    label="cup_to_table", key="full_cup_to_table")
tests["real_contact_sensor_registration"] = (
    len(created) == 1 and collection == [registered]
    and sensor_names == ["cup_to_table"]
    and scene.sensors == {"full_cup_to_table": registered})
try:
    route_fixed_target(current_q=current, left_target=left, left_ids=left_ids,
                       right_ids=right_ids, right_reset_q=right,
                       joint_names=names, expected_left_names=list(reversed(names[:29])))
except ValueError:
    tests["real_route_wrong_order_rejected"] = True

# Real torch values nested exactly through dict/list/kwargs, plus wrapper return/raise behavior.
nested = {"env_ids": torch.tensor([0]), "payload": [torch.arange(3), {"q": torch.ones((1, 2))}]}
tests["real_torch_nested_jsonable"] = json.loads(json.dumps(jsonable(nested))) == {
    "env_ids": [0], "payload": [[0, 1, 2], {"q": [[1.0, 1.0]]}]}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    context = lambda: {"phase": "explicit_reset", "started_control": 0,
                       "completed_controls": 0, "within_substep": -1}
    recorder = StateWriteRecorder(root / "writes.jsonl", context)
    received = []
    def original(*args, **kwargs):
        received.append((args, kwargs))
        return {"returned": torch.tensor([9.0])}
    wrapped = recorder.wrap(asset_path="/World/envs/env_.*/Aux", method_name="write_root_pose_to_sim",
                            original=original, snapshot=lambda: {"pose": torch.zeros((1, 7))})
    arg = torch.arange(7, dtype=torch.float32).reshape(1, 7)
    result = wrapped(arg, env_ids=torch.tensor([0]))
    rows = [json.loads(line) for line in (root / "writes.jsonl").read_text().splitlines()]
    tests["real_write_wrapper_same_args_return"] = received[0][0][0] is arg and result["returned"].item() == 9 and [x["event"] for x in rows] == ["CALL", "RETURNED"]
    marker = RuntimeError("expected-marker")
    def raising(*args, **kwargs):
        raise marker
    bad = recorder.wrap(asset_path="/World/envs/env_.*/Object", method_name="write_root_pose_to_sim",
                        original=raising, snapshot=lambda: {})
    try:
        bad(arg, env_ids=torch.tensor([0]))
    except RuntimeError as caught:
        tests["real_write_wrapper_same_exception"] = caught is marker
    rows = [json.loads(line) for line in (root / "writes.jsonl").read_text().splitlines()]
    tests["write_asset_paths_and_outcomes"] = rows[0]["asset_path"].endswith("/Aux") and rows[-1]["event"] == "RAISED" and rows[-1]["asset_path"].endswith("/Object")

# Quaternion behavior uses the production function.
q = np.array([0.30993634, -0.29788399, 0.724361, -0.53898561], np.float32)
tests["quat_self_zero"] = quat_angle(q, q) <= 1e-12
tests["quat_sign_zero"] = quat_angle(q, -q) <= 1e-12
tests["quat_scaled_normalized"] = quat_angle(q * 3.2, q * 0.2) <= 1e-12
for label, value in (("zero", np.zeros(4)), ("nan", np.array([1.0, 0.0, np.nan, 0.0]))):
    try:
        quat_angle(value, q)
    except ValueError:
        tests[f"quat_{label}_rejected"] = True

# Exact six-array interface shapes: scalar normal force and separation, vector point/normal.
empty = (np.zeros((8, 1), np.float32), np.full((8, 3), np.nan),
         np.full((8, 3), np.nan), np.full((8, 1), np.nan),
         np.array([[0]]), np.array([[0]]))
snapshot = unpack_six(empty, dt=1 / 240)
tests["six_empty_schema"] = snapshot["ranges"][0]["force"].shape == (0, 1)
nonempty = (np.array([[2.0], [3.0], [0.0]], np.float32), np.ones((3, 3), np.float32),
            np.ones((3, 3), np.float32), np.array([[-0.002], [0.001], [0.0]], np.float32),
            np.array([[2]]), np.array([[0]]))
snapshot = unpack_six(nonempty, dt=1 / 240)
tests["six_nonempty_sign_and_shape"] = snapshot["ranges"][0]["force"].shape == (2, 1) and snapshot["ranges"][0]["separation"][0, 0] < 0 < snapshot["ranges"][0]["separation"][1, 0]
nonempty[0][0, 0] = 999
tests["six_snapshot_breaks_alias"] = snapshot["ranges"][0]["force"][0, 0] == 2.0
try:
    unpack_six((np.zeros((2, 1)),) * 5, dt=1 / 240)
except ValueError:
    tests["six_five_array_rejected"] = True
try:
    unpack_six((np.zeros((2, 1)), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)), np.array([2]), np.array([1])), dt=1 / 240)
except OverflowError:
    tests["six_overflow_rejected"] = True
try:
    unpack_six((np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 1)), np.array([0]), np.array([0])), dt=1 / 240)
except ValueError:
    tests["fake_vector_force_rejected"] = True
tests["dtype_aware_float"] = float_close(np.array([0.05], np.float32), np.array([0.05], np.float64))[0]

# Actual atomic journal preserves 0..k-1 when the kth publish fails.
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    def inject(index, tmp, final):
        if index == 5:
            raise OSError("injected-before-publish")
    journal = AtomicSubstepJournal(root, atomic_until=12, before_publish=inject)
    for index in range(5):
        journal.save(index, {"global_substep": index, "contact_force": np.empty((0, 1), np.float32)})
    try:
        journal.save(5, {"global_substep": 5, "contact_force": np.empty((0, 1), np.float32)})
    except OSError:
        pass
    completed = sorted(root.glob("*.npz"))
    tests["real_substep_fault_preserves_prefix"] = len(completed) == 5 and all(int(np.load(path)["global_substep"]) == i for i, path in enumerate(completed))

# Actual media append/close functions with injected writer failures.
class FakeWriter:
    def __init__(self, events, *, fail_append=False, fail_close=False):
        self.events = events; self.fail_append = fail_append; self.fail_close = fail_close
    def append_data(self, frame):
        self.events.append("append")
        if self.fail_append: raise OSError("append-failure")
    def close(self):
        self.events.append("media_close")
        if self.fail_close: raise OSError("close-failure")
class FakeImageIO:
    def __init__(self, writer): self.writer = writer
    def imwrite(self, path, frame): Path(path).write_bytes(frame.tobytes())
    def get_writer(self, *args, **kwargs): return self.writer
class FakeApp:
    def __init__(self, events, fail=False): self.events = events; self.fail = fail
    def close(self):
        self.events.append("app_close")
        if self.fail: raise OSError("app-close-failure")
with tempfile.TemporaryDirectory() as temporary:
    events = []
    media = object.__new__(ThreeView); media.out = Path(temporary); media.imageio = FakeImageIO(FakeWriter(events, fail_append=True)); media.writers = {}; media.counts = {view: 0 for view in CAMERAS}; media.closed = False
    media._frame = lambda view, refreshes=2: (np.ones((480, 640, 3), np.uint8), {"view": view})
    try: media.append(0, "test")
    except OSError: pass
    tests["real_media_write_failure_saved"] = json.loads((Path(temporary) / "media_status.json").read_text())["state"] == "FAILED"
with tempfile.TemporaryDirectory() as temporary:
    events = []
    media = object.__new__(ThreeView); media.out = Path(temporary); media.writers = {"front": FakeWriter(events, fail_close=True)}; media.counts = {view: 1 for view in CAMERAS}; media.closed = False
    app = FakeApp(events)
    errors = close_media_then_app(media, app)
    tests["real_close_failure_preserved_and_ordered"] = [x["stage"] for x in errors] == ["media.close_media"] and events == ["media_close", "app_close"] and json.loads((Path(temporary) / "media_status.json").read_text())["state"] == "FAILED"
events = []
media = type("Media", (), {"close_media": lambda self: events.append("media_close")})()
errors = close_media_then_app(media, FakeApp(events, fail=True))
tests["real_app_close_failure_preserved"] = events == ["media_close", "app_close"] and errors[0]["stage"] == "app.close"

# Frozen camera contract and no approval/SCP dependency.
tests["frozen_camera_contract"] = validate_camera_config(camera_config())["resolution"] == [640, 480]

# Actual supervisor function: final log hashes, exit-0 FAILED, traceback, and missing completion.
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    def run_case(label, child_code):
        output = root / label
        return supervise_process(cmd=[sys.executable, "-c", child_code, str(output)], cwd=root,
                                 launch_dir=root / (label + "_launch"), output_dir=output,
                                 timeout_s=10, metadata={"case": label})
    success_code = "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'status.json').write_text(json.dumps({'state':'COMPLETE'}));(p/'completion.json').write_text(json.dumps({'status':'COMPLETE'}));print('final-line',flush=True)"
    success = run_case("success", success_code)
    tests["real_supervisor_success_and_final_hash"] = success["status"] == "COMPLETE" and success["stdout_sha256"] == sha256_file(root / "success_launch/stdout.log")
    fail_code = "import json,pathlib,sys,os;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'status.json').write_text(json.dumps({'state':'FAILED'}));os._exit(0)"
    failed = run_case("exit0_failed", fail_code)
    tests["real_supervisor_exit0_failed"] = failed["status"] == "FAILED" and "child_status_not_complete" in failed["reasons"]
    trace_code = "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'status.json').write_text(json.dumps({'state':'COMPLETE'}));(p/'completion.json').write_text(json.dumps({'status':'COMPLETE'}));sys.stderr.write('Traceback (most recent call last)\\n')"
    traced = run_case("traceback", trace_code)
    tests["real_supervisor_traceback_failed"] = traced["status"] == "FAILED" and "traceback" in traced["reasons"]
    missing_code = "import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'status.json').write_text(json.dumps({'state':'COMPLETE'}))"
    missing = run_case("missing_completion", missing_code)
    tests["real_supervisor_missing_completion_failed"] = missing["status"] == "FAILED" and "completion_missing_or_not_complete" in missing["reasons"]

tests = {key: bool(value) for key, value in tests.items()}
if not all(tests.values()):
    raise AssertionError({key: value for key, value in tests.items() if not value})
(B / "cpu_behavior_validation.json").write_text(json.dumps({"status": "PASS", "torch": torch.__version__, "tests": tests}, indent=2) + "\n")
print(json.dumps(tests, indent=2))
