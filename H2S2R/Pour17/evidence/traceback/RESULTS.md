# H2S2R periodic-traceback intervention result

Status: **INIT_RETURNED_WITH_PERIODIC_DUMP_DISABLED / IMPLEMENTATION_PASS**.

The isolated support change disabled only the repeating 30-second Python traceback dump. The actual V13/256 environment then constructed successfully under GDB and the unchanged line-585 guard exited before policy work. This is an engineering stability mitigation PASS for this initialization run; it is not proof of a unique historical root cause or long-term training stability.

## Support acceptance

- Original `resume_observation.py` SHA-256: `7fe7b1371a4c2aa493454ac74e27a84c8292f6f78e084673e674744f3030b9ab`.
- Deployed isolated overlay SHA-256: `23122aad8cdb8d7ebfd1e31de584f9ded43ac14b34e45cd55fdccef0381dda30`.
- The sole source difference gates `faulthandler.dump_traceback_later(30, repeat=True)` behind `H2S2R_DISABLE_PERIODIC_TRACEBACK=1`. Default behavior, `faulthandler.enable`, SIGUSR1 registration, ordinary event logs, resource records, initialization tracing and timeout handling remain present.
- The deployed overlay was actually imported from the remote overlay path. Both its 31.5-second CPU test and the real initialization produced a zero-byte `stacks.log`, demonstrating that construction did not re-register the periodic timer.
- The exact deployable GDB script was tested with a real CPU SIGSEGV. It saved full native backtrace, registers, shared libraries, all-thread stacks and a core. A deliberately failing single command was contained and later sections still completed.
- The first CPU test's collector worked, but its test assertion looked for the signal only in the redirected log while GDB printed that line to its console. The corrected tester checked both channels and passed; no Isaac allowance was involved.

## Initialization validation

- Frozen V13 identity: 32/32 PASS; scientific command fields PASS.
- Shared resource gate: both GPUs eligible; GPU1 selected at 3% mean/maximum utilization and 48,063 MiB minimum free during admission.
- Actual operational environment: physical GPU1, internal `cuda:0`, `RL_ISAAC_NO_GUARD=1`, `H2S2R_DISABLE_PERIODIC_TRACEBACK=1`.
- GDB/Isaac wall time: 141.872 seconds. Environment construction completed at observation time 135.312 seconds.
- Actual loaded `resume_observation` path and SHA match the overlay.
- Environment returned: yes. Native SIGSEGV/SIGABRT: none.
- The unchanged initialization-return guard wrote its marker at line 585 and exited code 86 before executing that line.
- Recorded initialization `SimulationContext.step` calls: entered 0, returned 0. Policy controls 0; PPO updates 0; evaluation 0.

Runtime supervision collected 14 samples. GPU1 maximum total memory use was 5,383 MiB, minimum free memory 43,156 MiB, maximum utilization 13%, and minimum host `MemAvailable` 221,858,729,984 bytes. No resource threshold, OOM, or accessible kernel Xid/NVRM evidence appeared. Both task PIDs were absent after exit.

## Decision and limits

The package's only Isaac start is consumed: 1/1. No retry, training, evaluation, video, physics rollout, trajectory or checkpoint was produced. The result establishes that V13/256 initialization can return once when the nonessential periodic traceback sampler is disabled under this diagnostic timing. It does not prove that all prior crashes came from that sampler; older runs differed, and 16-environment runs previously used periodic dumps successfully.

Remote evidence root: `/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01/`. CPU-test cores remain remote and are indexed in `remote_results/REMOTE_CORE_INDEX.md`.
