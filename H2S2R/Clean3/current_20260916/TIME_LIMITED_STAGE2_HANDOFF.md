# Clean3 time-limited Stage-2 handoff

## Decision

Use `latest_complete.pth` at `release_row=35`, not the early `best.pth`.

The run reached 5,013,504 agent steps and annealed 50 -> 45 -> 40 -> 35.
Its final window reported 61.21% grasp success, 98.28% certification, 0.86%
post-release drop, and 37.93% held-but-short.  The strict Stage-1 gate was not
passed: the final EMA was 46.32%, sustain count was 0, and row 10 was not
reached.

`best.pth` was selected by reward while the curriculum was still at the easier
row 50.  Reward values from different release rows are not comparable, so it is
kept only as a rollback artifact.  The recommended checkpoint is:

```text
latest_complete.pth
SHA256 34fc3873612f17d9041fe75986b87ce1ddac5d0fd6968d95fcd82476c3a5f9e5
release_row 35
```

Rollback only:

```text
best.pth
SHA256 d1777d8c981c700e463728048c2f96f694441bc9710fbd77d2472c41bca132d5
release_row 50
```

## What the new entry does

The Stage-1 checkpoint cannot be restored into the existing H2S2R LSTM entry:
Stage-1 is a 367D actor with 22D privileged input, while that entry is a 342D
actor with a 509D asymmetric critic and a different network topology.

`train_ours_stage2_from_stage1.py` therefore keeps the checkpoint-compatible
367D/22D Ours policy and the same bounded cumulative residual 58D controller.
It restores model weights and input/value normalizers.  Optimizer state is not
available in the Stage-1 checkpoint and starts fresh.  Within each episode:

1. before grasp certification, the disclosed Ours contact/hold objective is used;
2. after the grasp survives certification and the hold window, reward switches
   to the existing H2S2R bimanual reference-tracking reward;
3. the shared reference clock stays frozen until that switch.

The two existing 6D object-error slots remain fixed-hold errors before grasp
success, preserving the Stage-1 checkpoint semantics.  After success, those
same slots become current task-goal translation/rotation errors so the 367D
actor can observe the wiping reference without changing its input shape.

This is an explicitly disclosed bootstrap adaptation, not original H2S2R and
not evidence that the strict Stage-1 gate passed.  It also does not implement a
state-bank or transfer weights into the incompatible H2S2R LSTM.

## Deploy and launch

These commands assume the same directory contract as the validated A800 host.
All paths must remain below `/ssd/sy/kailang`.

```bash
cd Baseline-for-RL_correction/H2S2R/Clean3/current_20260916
bash scripts/deploy.sh /ssd/sy/kailang/clean3/curriculum_20260915

mkdir -p /ssd/sy/kailang/clean3/checkpoints
cp /path/received/latest_complete.pth \
  /ssd/sy/kailang/clean3/checkpoints/ours_stage1_latest_row35.pth

bash scripts/run_stage2_from_stage1.sh smoke \
  /ssd/sy/kailang/clean3/checkpoints/ours_stage1_latest_row35.pth \
  /ssd/sy/kailang/clean3/gates/stage2_latest_smoke_RUN 0
```

The smoke must produce `training_exit.json`, must not produce
`entry_failure.json`, and its log must show 367D observations, 22D privileged
input, and a restored checkpoint without missing/unexpected keys.  Only then:

```bash
bash scripts/run_stage2_from_stage1.sh formal \
  /ssd/sy/kailang/clean3/checkpoints/ours_stage1_latest_row35.pth \
  /ssd/sy/kailang/clean3/gates/stage2_latest_formal_150m_RUN 0
```

The formal default is 4096 environments and 150,000,000 agent steps.  The
launcher is one process on one selected GPU.  The package has not validated
DDP or eight-GPU synchronous PPO; on an eight-card machine, select one free GPU
instead of inventing untested multi-GPU behavior.

To use the rollback checkpoint deliberately, pass its SHA and row:

```bash
bash scripts/run_stage2_from_stage1.sh smoke /path/best.pth /path/output 0 \
  d1777d8c981c700e463728048c2f96f694441bc9710fbd77d2472c41bca132d5 50
```

Monitor with:

```bash
bash scripts/status.sh /path/output PID
```

After the process is demonstrably healthy but still running, stop repeated
polling and let the operator use the status command.
