# Method boundary and deviations

## What remains H2S2R

The comparison target is the H2S2R learning/task stage: its observation design,
asymmetric critic route, PPO implementation/configuration, moving task reference,
and task reward. Those components belong to Stage-2 and are not replaced by
Ours reward after the handoff.

## Why this differs from upstream H2S2R

### Direct 58D replaces FABRICS and PCA

The Clean3 robot has two arms and two 22-DoF hands that do not match the upstream
H2S2R embodiment. Porting FABRICS collision geometry and rebuilding a meaningful
hand PCA basis would be a separate adaptation project. The approved adaptation
therefore controls `right arm 7 + left arm 7 + right hand 22 + left hand 22`.
Normalized `-1/0/+1` maps to the loaded articulation lower limit, reset angle,
and upper limit. There is no hidden PCA projection or FABRICS repulsion. This is
a disclosed embodiment/control adaptation, not an unchanged paper reproduction.

### Ours Stage-1 precedes H2S2R Stage-2

Unassisted H2S2R adaptations repeatedly failed to acquire the initial grasp:
5M runs stayed at row 50 with roughly 90--100% post-release drop rates. The
teammate-proven Ours curriculum supplies the exploration window. While objects
are pinned, all 58 residual coordinates remain trainable; after release, success
measures stable grasp/hold. This stage uses Ours 367D observation, 22D privileged
critic input, reward, PPO network, and annealing controller. It is isolated by
`ours_stage1_contract=True`; the default H2S2R 342D/509D contract is unchanged.

Stage-1 is not baseline reward training. It is a disclosed initializer because
H2S2R cannot acquire this grasp from scratch on the adapted embodiment.

### Hard Stage-1/Stage-2 boundary

The approved handoff is: certify Stage-1; capture successful post-release states;
set each captured joint pose as the Stage-2 residual origin; zero cumulative
residual and previous action; remove/freeze Stage-1; then train H2S2R Stage-2.
This snapshot does not improvise that handoff. `train_lstm.py` is included for
source review but is not a released Stage-2 launch command.

Stage-1 uses contact/hold/certification and fixed initial-world-pose stability.
Stage-2 must use H2S2R task reward/reference; Stage-1 world-pose holding must not
leak into wiping, where object motion is expected.

## Timing and recording

- Physics 240 Hz; policy control 20 Hz (12 substeps per action).
- Videos write one frame per control at 20 fps and stop at first `done`.
- Launchers must source `/ssd/sy/kailang/tmp/cuda_env.sh` and set NVIDIA Vulkan
  ICD variables. Omitting these caused rejected PhysX CPU-fallback recordings.
