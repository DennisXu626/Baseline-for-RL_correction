# Method identity and intentional deviations

## Accurate name

This implementation is:

> H2S2R official PPO/LSTM + Pour-style direct-58D policy interface +
> Clip17/Ours Unscrew task environment.

It must not be reported as a strict end-to-end reproduction of the upstream
H2S2R environment.

Upstream reference: `tylerlum/human2sim2robot`, commit
`894eae2ec3ae39a573b81bd1860d14cc6bdfa6df`.

## Kept from upstream H2S2R

- released PPO implementation;
- actor MLP `[512, 512]` and one-layer 1024-unit LSTM;
- normalized observations, values and advantages;
- `gamma=0.998`, `tau=0.95`, learning rate `1e-4`;
- linear LR schedule, KL threshold `0.01`, PPO clip `0.2`;
- horizon 16, minibatch 16384, four mini-epochs;
- critic coefficient 4, sequence length 16 and bounds coefficient 0.005;
- asymmetric critic network `[1024, 512]`, learning rate `5e-5` and four
  mini-epochs;
- observation and action clipping at 5 and 1 respectively;
- formal vectorization of 4096 environments.

## Approved embodiment/task adaptations

- Fabrics and PCA are deliberately retired.
- A centralized 58D policy controls both 7-DoF arms and both 22-DoF Sharpa
  hands directly.
- The actor is 342D: the established 171D Pour-side observation duplicated for
  cap/right and bottle/left. `command_q` and `command_qd` are retained because
  the direct controller is stateful.
- The 513D critic is the 509D Pour-equivalent state plus normalized screw angle,
  engaged, has-depth and released truth fields.
- Ten fingertip-object sensors enter the critic. Four D6 inter-arm sensors are
  retained for task safety/reward but excluded from the critic state.
- Clip17 assets, two-body screw projection and the verified 30-degree physical
  screw model are retained.

## Material differences from the upstream H2S2R task

These differences are intentional for the current runnable baseline, but they
change the method and must be disclosed in comparisons.

### Reward

Upstream forced-reference training gives nonzero weight only to object tracking
(plus an optional action-smoothing term). Unscrew17 retains the Clip17/Ours
multi-term task reward: reference advance, leash, milestones, contact wages,
pinch, force/torque, screw progress, separation and earn-only contact bonuses.

Reason retained: the supplied Clip17 task was already physically validated
around staged bottle hold, cap pinch, screw release and separation. Replacing
the reward after smoke success would create a new experimental method rather
than a packaging change.

### Reference and reset protocol

Upstream samples reference playback speed in `[0.5, 1.0]` and applies per-run
XY/yaw offsets. Unscrew17 uses the fixed 451-row Clip17 reference clock with
contact-conditioned clock holding, cap rebase and task stages.

Unscrew17 also retains RSI entry points, 15-step object hold, task-specific
failure deadlines, success termination and the RSI/Phase-B curriculum. These
are not upstream H2S2R reset semantics.

Reason retained: these mechanisms are coupled to the verified screw/contact
state machine. Removing them requires a new physics and learning study.

### Simulation cadence and physics

Upstream's documented training command uses 15 Hz control (`dt=1/60`,
`controlFrequencyInv=4`, two substeps). This task stays at 20 Hz (`dt=1/240`,
decimation 12).

At 15 Hz each action is held for 66.7 ms instead of 50 ms, changing contact
impulse, screw torque, reward-per-second and every step-count threshold. The
Clip17 reference and its gates were authored at 20 Hz, so changing only the
frequency would be invalid; all timing thresholds and the reference would need
joint revalidation.

Other retained Clip17 physics differences include velocity iterations, contact
offset, maximum depenetration velocity, friction/mass choices, environment
spacing and `replicate_physics=False`.

### Domain randomization

The upstream README training recipe enables medium domain randomization. This
baseline currently preserves the deterministic Clip17 world and does not add
that upstream randomization package.

## Operational differences that do not change PPO updates

- checkpoint save frequency is 500 rather than upstream 1000, by user decision;
- a 5M agent-step gate is used before any longer run;
- smoke/preflight runs shrink both actor and critic minibatches to their small
  rollout size; formal runs keep 16384;
- non-finite frame-zero "no samples yet" diagnostics are omitted from
  TensorBoard only. Curriculum math already uses finite-safe rates.
