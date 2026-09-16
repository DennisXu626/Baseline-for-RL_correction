# Clean3 current package (2026-09-16)

This is the current A800 handoff package for training H2S2R on the Clean3
plate-and-sponge task. It supersedes the old C3P1/C3P1R2 execution path for new
runs. Historical material remains in the parent directory for audit only.

## Method identity

The experiment is an **H2S2R comparison baseline**, not an Ours evaluation.
Repeated attempts showed that the adapted H2S2R policy could not learn the
initial bimanual grasp from scratch on this embodiment. The approved pipeline
therefore has two explicitly separated stages:

1. **Ours Stage-1 bootstrap:** reproduce the teammate-proven release-row grasp
   curriculum and train a stable grasp initializer.
2. **H2S2R Stage-2:** initialize from captured successful grasp states, rebase
   the 58D residual origin, then train with the H2S2R observation, critic, PPO,
   task reference, and reward.

Ours Stage-1 is an initialization adaptation and must be disclosed. It must not
be described as part of original H2S2R. The strict state-bank/rebased-controller
Stage-2 route remains unreleased. Because the paper deadline no longer permits
another Stage-1 run, a separately authorized checkpoint-compatible continuation
is included. It keeps the Ours 367D/22D network and residual controller, then
switches to the H2S2R task reward after grasp success. See
[TIME_LIMITED_STAGE2_HANDOFF.md](TIME_LIMITED_STAGE2_HANDOFF.md); this exception
must not be reported as a strict Stage-1 gate pass or as original H2S2R.

## Current contract

- Action order: right arm 7, left arm 7, right hand 22, left hand 22.
- No FABRICS or hand PCA; the project hands differ from the upstream embodiment.
- Stage-1: bounded cumulative 58D residual, 367D actor input, 22D privileged
  critic, release rows `50 -> 45 -> ... -> 10`, and random 0--4 row jitter.
- Anneal rule: `EMA=0.95*old+0.05*success`; EMA >= 0.70 for ten consecutive
  windows lowers the release row by five and resets the EMA.
- Strict completion: row 10, ten qualifying windows at row 10, then deterministic
  video validation.
- Planned Stage-2: H2S2R-derived 342D actor observation, 509D asymmetric critic
  state, direct 58D control, and H2S2R task reward.

See [METHOD_AND_DEVIATIONS.md](METHOD_AND_DEVIATIONS.md) and
[DEPLOYMENT_AND_RUNBOOK.md](DEPLOYMENT_AND_RUNBOOK.md).

The `overlay` contains the current task code, compact NPZ inputs, and exact Ours
PPO modules from source commit `43d5747587619d5538ec4503089da30ff9ccaafc`.
Large USD/texture assets, Isaac Sim, and the shared Pour17 direct58D runtime are
not duplicated here.
