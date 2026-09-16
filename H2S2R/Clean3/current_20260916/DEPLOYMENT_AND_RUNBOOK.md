# Deployment and runbook

Validated server layout:

```text
/ssd/sy/kailang/clean3/curriculum_20260915
/ssd/sy/kailang/pour17/direct58d
/ssd/sy/kailang/env/rl-correction-pour/bin/python
/ssd/sy/kailang/tmp/cuda_env.sh
```

All writes must stay below `/ssd/sy/kailang`. The Clean3 runtime must already
contain `assets/retarget/object_0_textured.usd` and `object_1_textured.usd`.
The shared direct58D root supplies the robot USD, common modules, and official
H2S2R PPO dependency.

Deploy from this directory:

```bash
bash scripts/deploy.sh /ssd/sy/kailang/clean3/curriculum_20260915
```

Then run, in order, after GPU-policy admission:

```bash
bash scripts/run_stage1.sh gate  /ssd/sy/kailang/clean3/gates/contract_RUN 1
bash scripts/run_stage1.sh smoke /ssd/sy/kailang/clean3/gates/smoke_RUN 1
bash scripts/run_stage1.sh 5m    /ssd/sy/kailang/clean3/gates/stage1_RUN 1
```

The contract gate must report 367D/22D/58D, fixed row-110 deadline, correct
left/plate then right/sponge force order, and unchanged default H2S2R contract.
The smoke must write `training_exit.json`, not `entry_failure.json`. The 5M run
must be fresh, not resumed from smoke.

Monitor with `bash scripts/status.sh OUTPUT PID`. After three checks that still
show waiting, stop polling and hand the command to the operator. A process may
occasionally linger after writing terminal JSON; terminate only the exact PID
after verifying the terminal file.

Record with:

```bash
bash scripts/record_stage1.sh CHECKPOINT OUTPUT RELEASE_ROW PHYSICAL_GPU
```

Reject logs containing `GPU solver pipeline failed` or `GPU Bp pipeline failed`.

Do not start Stage-2 merely because 5M ended. Require row 10, EMA >= 0.70 for ten
windows at row 10, deterministic video PASS, state-bank/rebase tests, and a
Stage-2 smoke proving zero-action continuity plus 342D actor/509D critic shapes.
The state-bank/rebase implementation is pending in this snapshot.

For the separately authorized time-limited handoff that proceeds despite this
failed strict gate, read `TIME_LIMITED_STAGE2_HANDOFF.md`.  It adds a
checkpoint-compatible 367D/22D continuation entry and defaults to the final
row-35 `latest_complete.pth`; it does not make the original strict gate PASS.
