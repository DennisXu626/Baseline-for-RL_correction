from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployment_c3p1r2"


def test_runtime_wiring_contract_source():
    env = (DEPLOY / "tasks/h2s2r_clean3/env.py").read_text(encoding="utf-8")
    cfg = (DEPLOY / "tasks/h2s2r_clean3/cfg.py").read_text(encoding="utf-8")
    ast.parse(env)
    ast.parse(cfg)
    assert "set_bimanual_targets(self.controller, self._actions)" in env
    assert "advance_bimanual(self.controller)" in env
    assert "timestep=cfg.sim.dt, right_timestep=cfg.sim.dt" in env
    assert "install_bimanual(self.controller, cfg.action_sidecar_npz)" in env
    assert "offsets.abs() > 0.0050001" in env
    assert "cfg.warmup_clamp_steps != 0" in env
    assert "action_space = ACTION_DIM" in cfg and "observation_space = OBSERVATION_DIM" in cfg
    assert "cfg.sim.dt = 1.0 / 240.0" in cfg and "cfg.episode_length_s = 400" in cfg
    assert "plate_mass_kg = 0.30" in cfg and "sponge_mass_kg = 0.05" in cfg
    assert "object_friction = 1.0" in cfg and "table_top_z = 0.87" in cfg
    assert "filter_prim_paths_expr=[\"/World/envs/env_.*/Aux\"]" in cfg
    assert "filter_prim_paths_expr=[\"/World/envs/env_.*/Object\"]" in cfg


def test_train_and_eval_frozen_sizes_and_no_ours_control_import():
    train = (DEPLOY / "tasks/h2s2r_clean3/train_lstm.py").read_text(encoding="utf-8")
    evaluate = (DEPLOY / "tasks/h2s2r_clean3/evaluate.py").read_text(encoding="utf-8")
    ppo = (DEPLOY / "tasks/h2s2r_clean3/ppo_h2s2r_lstm.yaml").read_text(encoding="utf-8")
    for source in (train, evaluate):
        ast.parse(source)
        assert "GraspPose" not in source
        assert "progress_batch" not in source
        assert "ours" not in source.lower()
    assert "(args.num_envs, args.seed, args.max_epochs) != (256, 42, 512)" in train
    assert "minibatch_size: 4096" in ppo and "horizon_length: 16" in ppo
    assert "range(42, 58)" in evaluate and "range(400)" in evaluate
    assert "Clean3ExternalEvaluatorV1" in evaluate


def test_deployment_manifest_is_independent_and_complete():
    manifest = json.loads((DEPLOY / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["remote_target"] == "/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/deployment_v3"
    assert manifest["total_files"] >= 28
    assert manifest["total_bytes"] > 50_000_000
    assert all("pour17" not in name.lower() for name in manifest["files"])
    for command in manifest["launch_commands"].values():
        assert "CUDA_VISIBLE_DEVICES=0" in command
        assert "/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912" in command


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"PASS {len(tests)} tests")
