"""Official H2S2R PPO training entry for direct-58D Unscrew17."""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, os, signal, sys, traceback
from datetime import datetime
from pathlib import Path
from isaaclab.app import AppLauncher

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OFFICIAL = ROOT / "third_party/h2s2r_official"
UPSTREAM_COMMIT = "894eae2ec3ae39a573b81bd1860d14cc6bdfa6df"

parser = argparse.ArgumentParser()
parser.add_argument("--output_root", type=Path, required=True)
parser.add_argument("--name", default="H2S2R_Unscrew17_Direct58D")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int)
parser.add_argument("--max_epochs", type=int)
parser.add_argument("--load_path", type=Path)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--smoke_test", action="store_true")
parser.add_argument("--preflight", action="store_true")
parser.add_argument("--milestone_steps", type=int, default=0)
parser.add_argument("--pause_at_agent_steps", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0:
    raise SystemExit("--num_envs must be positive")
if args.max_agent_steps is not None and args.max_epochs is not None:
    raise SystemExit("choose either --max_agent_steps or --max_epochs")
if args.smoke_test and args.preflight:
    raise SystemExit("--smoke_test and --preflight are mutually exclusive")
if args.resume and args.load_path is None:
    raise SystemExit("--resume requires --load_path")
if args.milestone_steps < 0 or args.pause_at_agent_steps < 0:
    raise SystemExit("milestone and pause steps must be non-negative")
if args.pause_at_agent_steps and (
    not args.milestone_steps or args.pause_at_agent_steps % args.milestone_steps
):
    raise SystemExit("pause step must be an exact positive milestone")

app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch, yaml  # noqa: E402
sys.path.insert(0, str(OFFICIAL))
from human2sim2robot.ppo.ppo_agent import PpoAgent, PpoConfig  # noqa: E402
from human2sim2robot.ppo.utils.dict_to_dataclass import dict_to_dataclass  # noqa: E402
from human2sim2robot.ppo.utils.network import NetworkConfig  # noqa: E402
from rl_rebuild.baselines.h2s2r.official_ppo_adapter import OfficialPpoEnvAdapter  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.h2s2r_unscrew17.cfg import (CRITIC_STATE_DIM, OBSERVATION_DIM,  # noqa: E402
    UNSCREW17_ENV, build_cfg, unscrew_task_module)
from tasks.h2s2r_unscrew17.env import Unscrew17H2S2REnv  # noqa: E402
TASK = unscrew_task_module()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class UnscrewPpoAgent(PpoAgent):
    """Upstream PPO plus the unchanged Clip17 RSI curriculum."""
    milestone_steps = 0
    _next_milestone = 0
    _pause_sent = False

    def __init__(self, *a, raw_env=None, **kw):
        super().__init__(*a, **kw)
        self.raw = raw_env
        self.ema = {i: 0.0 for i in range(1, 5)}
        self.ema_t0 = {i: 0.0 for i in range(1, 4)}
        self.ema_cert = 0.0
        self.sustain = {i: 0 for i in range(1, 4)}
        self.phase_b = bool(getattr(raw_env, "phase_b", False))

    def train_epoch(self):
        value = super().train_epoch()
        if not self.milestone_steps:
            return value
        completed = int(self.epoch_num) * args.num_envs * int(self.cfg.horizon_length)
        while self._next_milestone <= completed:
            requested = self._next_milestone
            ckpt = self.nn_dir / f"milestone_{requested // 1_000_000:04d}M.pth"
            self.save(ckpt)
            atomic_json(ckpt.with_suffix(".json"), {
                "requested_step": requested, "completed_steps": completed,
                "completed_epochs": int(self.epoch_num), "checkpoint_sha256": sha256(ckpt)})
            print(f"[H2S2R-UNSCREW17] milestone -> {ckpt.name}", flush=True)
            self._next_milestone += self.milestone_steps
            if requested == args.pause_at_agent_steps and not self._pause_sent:
                self._pause_sent = True
                print(f"[H2S2R-UNSCREW17] pausing pid={os.getpid()} @ {completed}", flush=True)
                os.kill(os.getpid(), signal.SIGSTOP)
        return value

    def update_curriculum(self, frame: int) -> None:
        rates = self.raw.PB.pop_rates()
        for name, value in rates.items():
            if np.isfinite(value):
                self.writer.add_scalar(name, value, frame)
        for name, value in self.raw.pop_racc().items():
            if np.isfinite(value):
                self.writer.add_scalar(name, value, frame)
        rate = lambda name: TASK.finite_rate(rates, name)
        for gate in range(1, 5):
            self.ema[gate] = 0.98 * self.ema[gate] + 0.02 * rate(f"sr/gate{gate}")
        self.ema_cert = 0.98 * self.ema_cert + 0.02 * rate("sr/cert_pass")
        for gate in range(1, 4):
            self.ema_t0[gate] = 0.98 * self.ema_t0[gate] + 0.02 * rate(f"sr_t0/gate{gate}")
            ready = self.ema_t0[gate] >= 0.30
            if gate == 2:
                ready = ready and self.ema_cert >= 0.15
            self.sustain[gate] = self.sustain[gate] + 1 if ready else 0
            if (self.sustain[gate] >= 20 and gate not in self.raw.unlocked
                    and gate <= int(os.environ["UNSCREW_MAX_RSI_GATE"])):
                self.raw.unlocked.add(gate)
                self.raw._rebuild_entries()
                print(f"[H2S2R-UNSCREW17] RSI gate {gate} unlocked @ {frame}", flush=True)
        self.raw.p_t0 = max(0.5, 0.2 + 0.6 * self.ema[4])
        if not self.phase_b and self.ema[1] >= 0.7:
            self.phase_b = self.raw.phase_b = True
            print(f"[H2S2R-UNSCREW17] phase B unlocked @ {frame}", flush=True)
        values = {**{f"curr/ema_gate{i}": self.ema[i] for i in range(1, 5)},
                  "curr/ema_cert": self.ema_cert, "curr/p_t0": self.raw.p_t0,
                  "curr/n_entries": len(self.raw.entries), "curr/phase_b": float(self.phase_b)}
        for name, value in values.items():
            self.writer.add_scalar(name, value, frame)

    def write_stats(self, *a, **kw):
        super().write_stats(*a, **kw)
        if self.writer is None:
            return
        frame = int(kw.get("frame", self.epoch_num * args.num_envs * self.cfg.horizon_length))
        for name, value in self.env.last_extras.items():
            if torch.is_tensor(value) and value.numel() == 1:
                value = float(value.item())
                if np.isfinite(value):
                    self.writer.add_scalar(name, value, frame)
            elif isinstance(value, (int, float)) and np.isfinite(value):
                self.writer.add_scalar(name, float(value), frame)
        self.update_curriculum(frame)
        self.writer.flush()

    def train(self):
        try:
            return super().train()
        finally:
            if self.writer is not None:
                self.writer.close()


def load_configs():
    values = yaml.safe_load((HERE / "ppo_h2s2r_lstm.yaml").read_text())
    net, ppo = values["network"], values["ppo"]
    ppo["num_actors"], ppo["device"] = args.num_envs, args.device
    rollout, minibatch = args.num_envs * int(ppo["horizon_length"]), int(ppo["minibatch_size"])
    if args.smoke_test or args.preflight:
        ppo.update(max_epochs=3 if args.smoke_test else 1, max_frames=-1,
                   minibatch_size=rollout, save_frequency=1, save_best_after=0)
        ppo["asymmetric_critic"]["minibatch_size"] = rollout
    elif rollout < minibatch or rollout % minibatch:
        raise ValueError(f"formal rollout {rollout} must be a multiple of {minibatch}")
    if args.max_agent_steps is not None:
        ppo.update(max_frames=int(args.max_agent_steps), max_epochs=-1)
    elif args.max_epochs is not None:
        ppo.update(max_epochs=int(args.max_epochs), max_frames=-1)
    return dict_to_dataclass(net, NetworkConfig), dict_to_dataclass(ppo, PpoConfig)


def main():
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    network, ppo = load_configs()
    cfg = build_cfg(num_envs=args.num_envs, seed=args.seed); cfg.sim.device = args.device
    log_dir = args.output_root.resolve() / args.name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    kind = "smoke" if args.smoke_test else "preflight" if args.preflight else "formal"
    manifest = {
        "schema": "h2s2r_unscrew17_direct58d_lstm_v1", "run_kind": kind,
        "task": "Unscrew17 Clip17 V82 world", "seed": args.seed, "num_envs": args.num_envs,
        "action": {"width": 58, "order": "Rarm,Larm,Rhand,Lhand", "pca": False,
                   "fabrics": False, "decode": "-1/0/+1 -> lower/per-env reset/upper"},
        "actor_observation": {"width": OBSERVATION_DIM, "contract": "Pour17 direct58D"},
        "asymmetric_critic": {"width": CRITIC_STATE_DIM, "pour_base": 509,
            "extra": ["screw_angle", "engaged", "has_depth", "released"],
            "contacts": "ten fingertip-object sensors; D6 sensors excluded"},
        "task_behavior": "frozen Clip17 physics, reward, RSI curriculum, reset and success gates",
        "official_ppo": {"commit": UPSTREAM_COMMIT, "network": dataclasses.asdict(network),
                         "ppo": dataclasses.asdict(ppo)}, "task_environment": UNSCREW17_ENV,
        "reference": {"path": str(Path(TASK.MASTER).resolve()), "sha256": sha256(Path(TASK.MASTER))}}
    (log_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[H2S2R-UNSCREW17] log_dir={log_dir} envs={args.num_envs} kind={kind}", flush=True)
    print("[H2S2R-UNSCREW17] stage=construct_env begin", flush=True)
    try:
        raw = Unscrew17H2S2REnv(cfg)
    except BaseException as error:
        print(f"[H2S2R-UNSCREW17] construct failed: {error!r}", flush=True)
        traceback.print_exc(); raise
    env = OfficialPpoEnvAdapter(GymStyleEnvWrapper(raw, clip_actions=1.0))
    print("[H2S2R-UNSCREW17] stage=official_adapter complete", flush=True)
    agent = UnscrewPpoAgent(experiment_dir=log_dir, ppo_config=ppo,
        network_config=network, env=env, raw_env=raw)
    agent.milestone_steps = args.milestone_steps
    agent._next_milestone = args.milestone_steps
    if args.resume:
        agent.restore(args.load_path.resolve())
    print("[H2S2R-UNSCREW17] stage=train begin", flush=True)
    try:
        agent.train()
    finally:
        env.close()


try:
    main()
except BaseException as error:
    print(f"[H2S2R-UNSCREW17] fatal: {error!r}", flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
