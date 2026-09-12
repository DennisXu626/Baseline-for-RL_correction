"""Support-only runtime auditing and raw recording for the bounded pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import torch

from rl_rebuild.baselines.h2s2r.contract import ControlledSide
from .right_env import RightBottleH2S2REnv


from runtime.training_pilot_20260912.camera_delivery import CAMERAS, CameraDelivery


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class RawChunkWriter:
    """Write exact control-time arrays in bounded chunks without dropping failures."""

    def __init__(self, root: Path, *, controls_per_chunk: int) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.controls_per_chunk = int(controls_per_chunk)
        if self.controls_per_chunk <= 0:
            raise ValueError("controls_per_chunk must be positive")
        self._rows: list[dict[str, np.ndarray]] = []
        self.chunk_index = 0
        self.controls_written = 0

    def append(self, row: dict[str, np.ndarray]) -> None:
        if not all(np.isfinite(value).all() for value in row.values() if value.dtype.kind in "fcu"):
            raise RuntimeError("non-finite raw pilot trajectory value")
        self._rows.append(row)
        if len(self._rows) >= self.controls_per_chunk:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        keys = tuple(self._rows[0])
        if any(tuple(row) != keys for row in self._rows):
            raise RuntimeError("raw trajectory schema changed inside a chunk")
        arrays = {key: np.stack([row[key] for row in self._rows]) for key in keys}
        start = self.controls_written
        end = start + len(self._rows) - 1
        path = self.root / f"controls_{start:06d}_{end:06d}.npz"
        temporary = path.with_suffix('.npz.partial')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        self.controls_written += len(self._rows)
        self.chunk_index += 1
        self._rows.clear()


def v13_first_rollout_contract(num_envs: int) -> dict:
    if int(num_envs) != 256:
        raise ValueError('V13 first rollout requires 256 environments')
    return dict(controls=16, physics_substeps=192, right_set_targets=16,
                right_advance=192, position_target_submit=192,
                environment_transitions=4096)


class AuditedRightBottleEnv(RightBottleH2S2REnv):
    """Real training environment with counters and fail-closed object-write guards."""

    def __init__(
        self,
        *args,
        audit_root: Path,
        controls_per_chunk: int,
        video_epochs: Iterable[int] = (),
        capture_every_control: bool = False,
        capture_env0_substeps: bool = False,
        suppress_auto_reset: bool = False,
        **kwargs,
    ) -> None:
        self._audit_enabled = False
        self._audit_in_reset = False
        self._suppress_auto_reset = bool(suppress_auto_reset)
        super().__init__(*args, **kwargs)
        self.audit_root = Path(audit_root)
        self.audit_root.mkdir(parents=True, exist_ok=False)
        self.raw_writer = RawChunkWriter(
            self.audit_root / "raw_chunks", controls_per_chunk=controls_per_chunk
        )
        self.audit_epoch = 0
        self.audit_control = 0
        self.audit_counts = {
            "controls": 0,
            "physics_substeps": 0,
            "right_set_targets": 0,
            "right_advance": 0,
            "position_target_submit": 0,
            "reset_calls": 0,
            "reset_envs": 0,
            "reset_object_pose_writes": 0,
            "reset_object_velocity_writes": 0,
            "active_object_pose_writes": 0,
            "active_object_velocity_writes": 0,
        }
        self._audit_reset_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._terminal_before_reset: dict[str, torch.Tensor] = {}
        self._video_epochs = frozenset(int(value) for value in video_epochs)
        self._capture_every_control = bool(capture_every_control)
        self._capture_env0_substeps = bool(capture_env0_substeps)
        self._substep_rows: list[dict[str, np.ndarray]] = []
        self._current_action: torch.Tensor | None = None
        self._current_substep = 0
        self._frames: dict[int, dict[str, list[np.ndarray]]] = {}  # Legacy partial fixture compatibility; never stores runtime frames.
        self._media_episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self.delivery = CameraDelivery(self, self.audit_root / 'delivery')
        self.left_delivery = CameraDelivery(self, self.audit_root / 'left_delivery', side='left')
        self._reward_diagnostics = {}
        atomic_json(self.audit_root / 'left_sensor_contract.json', {
            'sensors': [{'prim': s.cfg.prim_path, 'filters': list(s.cfg.filter_prim_paths_expr)}
                        for s in self._contact_sensors[5:10]],
            'left_sim_joint_ids': self._cpu(self.side_joint_ids[ControlledSide.LEFT]).tolist(),
            'all_recorded_sim_joint_ids': self._cpu(self.sim_joint_ids).tolist(),
            'scope': 'existing filtered left fingertip versus Aux/cup, not all contact surfaces'})
        if any(list(s.cfg.filter_prim_paths_expr) != ['/World/envs/env_.*/Aux']
               for s in self._contact_sensors[5:10]):
            raise RuntimeError('left force filter is not the cup; no inferred contact substitution')
        self._wrap_counter(self.controller.right, "set_targets", "right_set_targets")
        self._wrap_counter(self.controller.right, "advance", "right_advance")
        self._wrap_counter(self.hand, "set_joint_position_target", "position_target_submit")
        self._wrap_object_write(self.object, "write_root_pose_to_sim", "pose")
        self._wrap_object_write(self.aux, "write_root_pose_to_sim", "pose")
        self._wrap_object_write(self.object, "write_root_velocity_to_sim", "velocity")
        self._wrap_object_write(self.aux, "write_root_velocity_to_sim", "velocity")
        self._wrap_sim_step()
        self._audit_enabled = True

    def _support_state(self):
        """Read-only left/cup and robot state; same temporal phase as the raw row."""
        cup = self._object_poses()[0]
        pos, quat = self._wrist(ControlledSide.LEFT)
        ids = self.side_joint_ids[ControlledSide.LEFT]
        force = torch.stack([torch.linalg.vector_norm(
            s.data.force_matrix_w.reshape(self.num_envs, -1, 3), dim=-1).max(dim=-1).values
            for s in self._contact_sensors[5:10]], dim=1)
        return {
            'cup_pose_env_xyz_wxyz': cup,
            'left_palm_pose_env_xyz_wxyz': torch.cat((pos, quat), dim=1),
            'cup_linear_velocity_world_m_s': self.aux.data.root_lin_vel_w,
            'cup_angular_velocity_world_rad_s': self.aux.data.root_ang_vel_w,
            'left_finger_cup_contact_force_n': force,
            'left_actual_minus_fixed_target_rad': self.hand.data.joint_pos[:, ids] - self.controller.left.q,
            'robot_root_pose_world_xyz_wxyz': torch.cat(
                (self.hand.data.root_pos_w, self.hand.data.root_quat_w), dim=1),
        }

    def _get_rewards(self):
        # Observe the exact pre-clock reward inputs, then return the unchanged core result.
        if not self._audit_enabled:
            return super()._get_rewards()
        from rl_rebuild.baselines.h2s2r.reward import object_tracking_reward
        values = self._right_quantities()
        _, terms = object_tracking_reward(values['bottle_error'], values['right_tip_distance'],
            previous_actions=self._previous_actions, actions=self._actions,
            smoothing_weight=self.cfg.action_smoothing_weight,
            close_threshold=self.cfg.fingertips_close_threshold_m,
            stop_reference_threshold=self.cfg.stop_reference_threshold_m)
        self._reward_diagnostics = {
            'reward_tracking': terms['object_tracking'].detach().clone(),
            'reward_close_gate': terms['fingertips_close'].detach().clone(),
            'reward_action_smoothing_unweighted': terms['action_smoothing_penalty'].detach().clone(),
            'bottle_error_reward_time_m': values['bottle_error'].detach().clone(),
            'right_tip_distance_reward_time_m': values['right_tip_distance'].detach().clone(),
            'reference_float_reward_time': self.clock.float_indices.detach().clone(),
            'action_boundary_fraction': (self._actions.abs() >= 0.999).float().mean(dim=1),
        }
        result = super()._get_rewards()
        self._reward_diagnostics['reward_returned'] = result.detach().clone()
        return result

    def verify_first_rollout(self):
        contract = v13_first_rollout_contract(self.num_envs)
        expected = {key: contract[key] for key in ('controls', 'physics_substeps',
                    'right_set_targets', 'right_advance', 'position_target_submit')}
        if any(self.audit_counts[k] != v for k, v in expected.items()):
            raise RuntimeError('first rollout cadence mismatch before optimizer')
        if self.cfg.warmup_clamp_steps != 0 or self.audit_counts['active_object_pose_writes'] or self.audit_counts['active_object_velocity_writes']:
            raise RuntimeError('first rollout hold/object-write violation')
        atomic_json(self.audit_root / 'first_rollout.json',
            dict(state='NUMERIC_PASS_AWAITING_LEFT_AND_RIGHT_VISUAL', counts=self.audit_counts,
                 optimizer_updates=0, environment_transitions=4096,
                 phase='inside_env_step_before_rollout_returns'))

    def _wrap_sim_step(self) -> None:
        original = self.sim.step

        def audited_step(*args, **kwargs):
            value = original(*args, **kwargs)
            if (
                self._audit_enabled
                and not self._audit_in_reset
                and self._capture_env0_substeps
                and self._current_action is not None
            ):
                bottle = self._object_poses()[1]
                palm_position, palm_quaternion = self._wrist(ControlledSide.RIGHT)
                row = {
                    "control_index": np.asarray(self.audit_control, dtype=np.int64),
                    "substep_index": np.asarray(self._current_substep, dtype=np.int64),
                    "action": self._cpu(self._current_action[0], np.float32),
                    "q_rad": self._cpu(self.hand.data.joint_pos[0, self.sim_joint_ids], np.float32),
                    "qd_rad_s": self._cpu(self.hand.data.joint_vel[0, self.sim_joint_ids], np.float32),
                    "target_q_rad": self._cpu(self._target_q[0, self.sim_joint_ids], np.float32),
                    "bottle_pose_env_xyz_wxyz": self._cpu(bottle[0], np.float32),
                    "right_palm_pose_env_xyz_wxyz": self._cpu(
                        torch.cat((palm_position[0], palm_quaternion[0])), np.float32
                    ),
                    "right_finger_contact_force_n": self._cpu(
                        self._contact_force()[0], np.float32
                    ),
                }
                if not all(
                    np.isfinite(item).all()
                    for item in row.values()
                    if item.dtype.kind in "fcu"
                ):
                    raise RuntimeError("non-finite env0 substep trajectory value")
                self._substep_rows.append(row)
                self._current_substep += 1
            return value

        self.sim.step = audited_step

    def _wrap_counter(self, owner: object, name: str, key: str) -> None:
        original = getattr(owner, name)

        def counted(*args, **kwargs):
            if self._audit_enabled and not self._audit_in_reset:
                self.audit_counts[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, counted)

    def _wrap_object_write(self, owner: object, name: str, kind: str) -> None:
        original = getattr(owner, name)

        def guarded(*args, **kwargs):
            if self._audit_enabled:
                scope = "reset" if self._audit_in_reset else "active"
                key = f"{scope}_object_{kind}_writes"
                self.audit_counts[key] += 1
                if scope == "active":
                    raise RuntimeError(f"forbidden active-rollout object {kind} write")
            return original(*args, **kwargs)

        setattr(owner, name, guarded)

    def _reset_idx(self, env_ids) -> None:
        if not self._audit_enabled:
            return super()._reset_idx(env_ids)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        elif torch.is_tensor(env_ids):
            ids = env_ids
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        self.audit_counts["reset_calls"] += 1
        self.audit_counts["reset_envs"] += int(len(ids))
        self._audit_reset_mask[ids] = True
        bottle = self._object_poses()[1]
        palm_position, palm_quaternion = self._wrist(ControlledSide.RIGHT)
        self._terminal_before_reset = {
            "ids": ids.detach().clone(),
            "q_rad": self.hand.data.joint_pos[ids][:, self.sim_joint_ids].detach().clone(),
            "qd_rad_s": self.hand.data.joint_vel[ids][:, self.sim_joint_ids].detach().clone(),
            "target_q_rad": self._target_q[ids][:, self.sim_joint_ids].detach().clone(),
            "bottle_pose_env_xyz_wxyz": bottle[ids].detach().clone(),
            "right_palm_pose_env_xyz_wxyz": torch.cat(
                (palm_position[ids], palm_quaternion[ids]), dim=1
            ).detach().clone(),
            "right_finger_contact_force_n": self._contact_force()[ids].detach().clone(),
            "reference_index": self.clock.indices[ids].detach().clone(),
            "episode_length": self.episode_length_buf[ids].detach().clone(),
        }
        self._terminal_before_reset.update({
            key: value[ids].detach().clone() for key, value in self._support_state().items()})
        media_ids = ids.detach().cpu().numpy()
        if self._current_action is not None and 0 in media_ids and (
            self._capture_every_control or self.audit_epoch in self._video_epochs
        ):
            boundary = self.delivery.capture(0, int(self._media_episode_ids[0]),
                self.audit_control, 'pre_reset_terminal', reset=True)
            self.delivery.save_frames(self.audit_root / 'reset_boundaries' /
                f'control_{self.audit_control:06d}_pre', boundary)
        self._audit_in_reset = True
        try:
            if self._suppress_auto_reset:
                return None
            value = super()._reset_idx(ids)
            self._media_episode_ids[media_ids] += 1
            return value
        finally:
            self._audit_in_reset = False

    @staticmethod
    def _cpu(value: torch.Tensor, dtype=None) -> np.ndarray:
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=True) if dtype is not None else array.copy()

    def _contact_force(self) -> torch.Tensor:
        values = [
            torch.linalg.vector_norm(
                sensor.data.force_matrix_w.reshape(self.num_envs, -1, 3), dim=-1
            ).max(dim=-1).values
            for sensor in self._contact_sensors[:5]
        ]
        return torch.stack(values, dim=1)

    def _snapshot_row(self, action: torch.Tensor, result: tuple) -> dict[str, np.ndarray]:
        bottle = self._object_poses()[1]
        palm_position, palm_quaternion = self._wrist(ControlledSide.RIGHT)
        terminated = result[2]
        time_out = result[3]
        row = {
            "epoch": np.asarray(self.audit_epoch, dtype=np.int64),
            "control_index": np.asarray(self.audit_control, dtype=np.int64),
            "action": self._cpu(action, np.float32),
            "q_rad": self._cpu(self.hand.data.joint_pos[:, self.sim_joint_ids], np.float32),
            "qd_rad_s": self._cpu(self.hand.data.joint_vel[:, self.sim_joint_ids], np.float32),
            "target_q_rad": self._cpu(self._target_q[:, self.sim_joint_ids], np.float32),
            "bottle_pose_env_xyz_wxyz": self._cpu(bottle, np.float32),
            "right_palm_pose_env_xyz_wxyz": self._cpu(
                torch.cat((palm_position, palm_quaternion), dim=1), np.float32
            ),
            "right_finger_contact_force_n": self._cpu(self._contact_force(), np.float32),
            "terminated": self._cpu(terminated, np.bool_),
            "time_out": self._cpu(time_out, np.bool_),
            "reset": self._cpu(self._audit_reset_mask, np.bool_),
            "reference_index": self._cpu(self.clock.indices, np.int64),
            "episode_length": self._cpu(self.episode_length_buf, np.int64),
        }
        row.update({key: self._cpu(value, np.float32) for key, value in self._support_state().items()})
        row.update({key: self._cpu(value, np.float32) for key, value in self._reward_diagnostics.items()})
        if self._terminal_before_reset and bool(self._audit_reset_mask.any()):
            ids = self._cpu(self._terminal_before_reset["ids"], np.int64)
            for key in (
                "q_rad", "qd_rad_s", "target_q_rad",
                "bottle_pose_env_xyz_wxyz", "right_palm_pose_env_xyz_wxyz",
                "right_finger_contact_force_n", "reference_index", "episode_length",
            ):
                row[key][ids] = self._cpu(
                    self._terminal_before_reset[key], row[key].dtype
                )
            for key in self._support_state():
                row[key][ids] = self._cpu(self._terminal_before_reset[key], row[key].dtype)
        return row

    def _capture_views(self) -> None:
        if not (self._capture_every_control or self.audit_epoch in self._video_epochs):
            return
        reset = bool(self._audit_reset_mask[0])
        phase = 'post_reset' if reset else 'post_control'
        frames = self.delivery.capture(0, int(self._media_episode_ids[0]),
                                       self.audit_control, phase, reset=reset)
        directory = self.audit_root / "milestones" / f"epoch_{self.audit_epoch:04d}"
        self.delivery.append(directory, frames)
        left_frames = self.left_delivery.capture(0, int(self._media_episode_ids[0]),
                                               self.audit_control, phase, reset=reset)
        self.left_delivery.append(directory / 'left_cup', left_frames)
        first_training_gate = self.audit_epoch == 1 and self.audit_counts['controls'] == 16
        if first_training_gate:
            self.verify_first_rollout()
        if (self._capture_every_control and self.audit_counts['controls'] in (1, 16)) or first_training_gate:
            self.raw_writer.flush()
            if self._substep_rows:
                np.savez_compressed(self.audit_root / f"substeps_checkpoint_{self.audit_counts['controls']:03d}.npz",
                    **{k: np.stack([r[k] for r in self._substep_rows]) for k in self._substep_rows[0]})
            self.delivery.wait_signoff(f"control_{self.audit_counts['controls']:03d}", {0: frames})
            self.left_delivery.wait_signoff(f"control_{self.audit_counts['controls']:03d}", {0: left_frames})
            if first_training_gate:
                atomic_json(self.audit_root / 'first_rollout.json',
                    dict(state='PASS_BEFORE_OPTIMIZER', counts=self.audit_counts, optimizer_updates=0,
                         environment_transitions=4096,
                         left_scope='current fixed holding/shared scene only; not historical dynamic certification'))

    def step(self, action: torch.Tensor):
        self.left_delivery.preview()
        self.delivery.preview()  # Fail closed before applying the first policy control.
        if action.shape != (self.num_envs, 11) or not bool(torch.isfinite(action).all()):
            raise RuntimeError(f"invalid policy action shape/value: {tuple(action.shape)}")
        self._audit_reset_mask.zero_()
        self._terminal_before_reset = {}
        before = dict(self.audit_counts)
        sim_before = int(self._sim_step_counter)
        self._current_action = action.detach().clone()
        self._current_substep = 0
        try:
            result = super().step(action)
        finally:
            self._current_action = None
        self.audit_counts["controls"] += 1
        self.audit_counts["physics_substeps"] += int(self._sim_step_counter) - sim_before
        deltas = {
            key: self.audit_counts[key] - before[key]
            for key in ("right_set_targets", "right_advance", "position_target_submit")
        }
        if deltas != {
            "right_set_targets": 1,
            "right_advance": 12,
            "position_target_submit": 12,
        }:
            raise RuntimeError(f"per-control cadence contradiction: {deltas}")
        if int(self._sim_step_counter) - sim_before != 12:
            raise RuntimeError("per-control physics substep count is not 12")
        if self._capture_env0_substeps and self._current_substep != 12:
            raise RuntimeError(
                f"env0 substep recorder saw {self._current_substep}, expected 12"
            )
        row = self._snapshot_row(action, result)
        if self.audit_control == 0:
            schema = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'bytes': int(v.nbytes)}
                      for k, v in row.items()}
            atomic_json(self.audit_root / 'raw_schema.json', schema)
            if sum(v.nbytes for v in row.values()) > self.num_envs * 1152 + 16:
                raise RuntimeError('raw schema exceeds predeclared capacity envelope')
        self.raw_writer.append(row)
        self._capture_views()
        self.audit_control += 1
        return result

    def finish_epoch(self, epoch: int) -> None:
        if epoch == 1:
            expected = {
                "controls": 16,
                "physics_substeps": 192,
                "right_set_targets": 16,
                "right_advance": 192,
                "position_target_submit": 192,
            }
            actual = {key: self.audit_counts[key] for key in expected}
            if actual != expected:
                raise RuntimeError(f"first-rollout interface contradiction: {actual}")
            if self.audit_counts["active_object_pose_writes"] or self.audit_counts[
                "active_object_velocity_writes"
            ]:
                raise RuntimeError("active rollout wrote an object state")
        directory = self.audit_root / 'milestones' / f'epoch_{epoch:04d}'
        self.delivery.close(directory)
        self.left_delivery.close(directory / 'left_cup')
        self.raw_writer.flush()
        if epoch in self._video_epochs and epoch != 1:
            self.delivery.wait_signoff(f'epoch_{epoch:04d}', {0: self.delivery.last[0]})
            self.left_delivery.wait_signoff(f'epoch_{epoch:04d}', {0: self.left_delivery.last[0]})
        atomic_json(self.audit_root / "runtime_counts.json", self.audit_counts)

    def close_audit(self) -> None:
        self.raw_writer.flush()
        if self._capture_env0_substeps and self._substep_rows:
            keys = tuple(self._substep_rows[0])
            arrays = {
                key: np.stack([row[key] for row in self._substep_rows])
                for key in keys
            }
            np.savez_compressed(self.audit_root / "env0_substeps.npz", **arrays)
            self._substep_rows.clear()
        atomic_json(self.audit_root / "runtime_counts.json", self.audit_counts)
        self.delivery.close()
        self.left_delivery.close()
