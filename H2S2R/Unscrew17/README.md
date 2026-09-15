# Unscrew17 direct-58D baseline handoff

Status: implementation and smoke validation complete; formal training is
stopped and must not be resumed from the failed launch.

This directory transfers the H2S2R-PPO + Clip17/Ours Unscrew17 baseline built
on 2026-09-16. It contains the verified Clip17 runtime-input archive plus the
H2S2R overlay; it is not a strict reproduction of the upstream H2S2R task
environment.

Read in this order:

1. `METHOD_AND_DEVIATIONS.md` — what is upstream H2S2R and what is adapted.
2. `DEPLOYMENT_AND_RUNBOOK.md` — input preparation, smoke tests and 5M launch.
3. `VALIDATION.md` — evidence already collected and the unresolved disk blocker.
4. `SOURCE_IDENTITY.json` and `FILES.sha256` — immutable source identities.

## Frozen policy contract

| Item | Value |
|---|---|
| Task | Clip17 bottle-cap unscrewing |
| Environments | 4096 by default and for formal training |
| Action | 58D: right arm 7, left arm 7, right hand 22, left hand 22 |
| Controller | Direct joint-position; `-1/0/+1` maps to hard lower / per-env reset / hard upper |
| Fabrics / PCA | Disabled / disabled |
| Actor observation | 342D (171D per side) |
| Asymmetric critic | 513D |
| Control rate | 20 Hz (`dt=1/240`, decimation 12) |
| PPO | Upstream H2S2R commit `894eae2`, LSTM, horizon 16 |
| PPO minibatch | 16384, four mini-epochs |
| Checkpoint cadence | 500 epochs; 5M milestone for the first gate |

## What is and is not included

Included:

- `runtime_assets/unscrew17_clip17_runtime_inputs_20260916.tar.gz`, containing
  the USD/NPZ task inputs and the verified base runtime tree;
- the local ground USD used by the validated A800 environment;
- the complete policy adapter and training entry;
- the two Clip17 runtime files changed for local/offline asset loading;
- the exact upstream H2S2R PPO snapshot and MIT license;
- the successful 1-env and 1024-env smoke logs;
- a bounded first/last-lines excerpt of the failed 4096-env launch log;
- deployment, launch, monitoring and method-identity documentation.

Not included:

- the 6.58 GB raw failed-launch log, whose repeated filesystem errors exceed
  GitHub's file limits and add no evidence beyond the included excerpt;
- a formal-training checkpoint, because the formal launch never reached PPO;
- the optional 1024-env smoke checkpoint. Its verified identity is recorded in
  `SOURCE_IDENTITY.json`; it is not required to start formal training.

Do not replace included or externally identified files with similarly named
files without checking their SHA256 identities.

## Current stop condition

The 1024-env PPO smoke passed three epochs. The later 4096-env 5M launch never
reached PPO because the A800 host root filesystem (including `/tmp`) had zero
free space. That PID was stopped. No valid formal checkpoint exists.

Do not launch until the disk preflight in `DEPLOYMENT_AND_RUNBOOK.md` passes.
