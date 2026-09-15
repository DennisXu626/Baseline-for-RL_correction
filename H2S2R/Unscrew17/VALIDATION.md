# Validation record

## Source handoff

The original Clip17 package verifier passed:

- 2,374 files;
- 140 source-contract checks;
- 388 NPZ checks;
- 228 PTH CRC checks;
- 99 evaluation records;
- zero verifier errors.

## Static checks

- Python compilation passed for the task adapter and PPO bridge.
- The policy path imports neither Fabrics nor PCA.
- The deployed source hashes matched the local implementation snapshot at the
  end of the implementation turn.

## 1-env preflight

Remote evidence root:

```text
/ssd/sy/kailang/unscrew17/gates/shape_runtime_preflight_1env_gpu1_retry8_20260916
```

Result:

- environment constructed;
- actor `RunningMeanStd: (342,)`;
- asymmetric critic `RunningMeanStd: (513,)`;
- one rollout and both PPO updates completed;
- checkpoint size approximately 86.2 MB;
- safe weights-only inspection found 78 tensors, all finite;
- normal exit at `MAX EPOCHS NUM!`.

## 1024-env PPO smoke

Remote evidence root:

```text
/ssd/sy/kailang/unscrew17/gates/ppo_smoke_1024_gpu1_20260916
```

Result:

- 3/3 epochs completed;
- total throughput approximately 303–325 FPS;
- finite episodic rewards were printed each epoch;
- epoch 1/2/3, best and final checkpoints were saved;
- no fatal traceback.

One TensorBoard warning at frame zero was traced to four diagnostics whose
denominator had no samples yet: `sr/cert_pass`, `diag/certfail_rise`,
`diag/certfail_slip` and `diag/certfail_pads`. It was not model input, reward or
gradient divergence. The training entry now omits only non-finite no-sample
diagnostics from TensorBoard; the existing finite-safe curriculum math is
unchanged.

## Invalid formal launch

Remote evidence root:

```text
/ssd/sy/kailang/unscrew17/gates/formal_5m_4096env_gpu1_20260916
```

The process remained in 4096-env scene construction and emitted
`OSError: [Errno 28] No space left on device`. Read-only inspection showed:

```text
/      100% used, 0 available
/tmp   same root filesystem
/ssd   36% used, about 4.3 TB available
```

It never reached `stage=official_adapter complete` or `stage=train begin`, so
it produced no valid formal training steps, reward or checkpoint. PID 645712
was stopped and no replacement process was launched.

This is an infrastructure failure, not a failed 4096-env PPO update. A future
run still needs a fresh 4096-env construction plus at least one PPO update to
prove full-scale resource sufficiency.
