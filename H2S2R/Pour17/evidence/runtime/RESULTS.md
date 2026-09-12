# H2S2R zero-physics native runtime evidence package

## Outcome

Status: `COMPLETE_NO_NATIVE_DUMP / ROOT_CAUSE_UNRESOLVED`.

This package performed read-only evidence collection only. It did not start Isaac Sim, physics, policy execution, training, an import probe, or a debugger-assisted reproduction.

The two abnormal exits that are relevant to the native-crash branch remain established by their parent launch manifests as `return_code=-11` (SIGSEGV), but no Kit crash report, minidump, core file, apport report, or native backtrace was found in the scoped existing evidence. Therefore the faulting native thread, instruction, and shared library are unknown. The last Python log line and the periodic Python faulthandler stack are localization evidence only and are not a root-cause finding.

## Existing native evidence search

Read-only searches were limited to the `kailang` user's Isaac/Omniverse logs and caches, the V12/V13 runtime temporary trees, `/tmp` files owned by `kailang`, and `/var/crash` files owned by `kailang`.

- Relevant-time search (`2026-09-12 10:30–15:30 UTC`) found no `core`, `core.*`, `*.core`, `*.dmp`, `*.mdmp`, `*.minidump`, `*.crash`, or `*.dump` file.
- An all-time filename search over the same user-scoped Isaac/Omniverse/V12/V13 trees also found no such artifact or file under a crash-named directory.
- `/var/crash` contained no matching file owned by `kailang` in the relevant time window.
- `coredumpctl` is not installed/available on the host.
- `/proc/sys/kernel/core_pattern` is an apport pipe, but the launch user's `ulimit -c` is `0`; all four Kit commands also contain `--/app/installSignalHandlers=0`.
- The four Kit logs contain no SIGSEGV, fatal-error, minidump, crash-reporter, crash-dump, core-dump, or received-signal record. Matches for the word “Segmentation” were only Replicator node names such as `InstanceSegmentationLegacy`, not crashes.

These facts explain why no ordinary core was retained under the observed launch configuration; they do not explain why SIGSEGV occurred.

## Run comparison

| Run | Entry / envs | Parent result | Furthest recorded boundary | Policy / PPO | Kit log end |
|---|---|---|---|---:|---|
| successful short02 | `short_physics_validity.py`, 16 | rc 0, 695.910035 s | environment returned; 256 controls; normal `app.close` | 256 / 0 | normal stage close and shutdown |
| failed resume01 | `train_right_lstm.py`, 1024 | rc -9, initialization timeout, 487.082299 s | entered `SimulationContext.reset` at 147.482812 s; did not return | 0 / 0 | ordinary USD/material population records, then cutoff |
| failed resume02 | `train_right_lstm.py`, 1024 | rc -11, 122.233336 s | entered `DirectRLEnv.__init__` at 13.752175 s; no reset-enter event | 0 / 0 | ordinary asset opens; no shutdown or native fault record |
| failed V13 train01 | `train_right_lstm.py`, 256 | rc -11, 32.311839 s | entered `DirectRLEnv.__init__` at 13.553506 s; no reset-enter event | 0 / 0 | ordinary asset opens; no shutdown or native fault record |

The 1024 timeout is not the same recorded stop as either SIGSEGV. The two SIGSEGV runs have the same coarse observation boundary, but that boundary encloses native USD/scene construction and does not identify a faulting function.

## Runtime and startup identity

The following items are the same in all four Kit logs/manifests:

- host `mscauto-Lambda-Vector`;
- interpreter `/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python` resolving to Python 3.11;
- Isaac Sim 5.1.0, Kit `107.3.3+production.229672.69cbf6ad.gl`, kernel `206.6+release.9587.07f17b1b.gl`;
- driver 580.173.02 and reported CUDA 13.0;
- `isaaclab.python.headless.rendering.kit`, headless/no-window rendering, fast shutdown, Kit signal handlers disabled;
- `CUDA_VISIBLE_DEVICES=1`, with the child selecting logical `cuda:0` and Kit `/renderer/activeGpu=0`, `/physics/cudaDevice=0`;
- the normalized set of 649 registered Kit extensions, SHA-256 `972bb8f3a3f48766fc2f89b255bf39b3e2e9c7dd5da1fc26741c1a0f81db2afd`;
- core environment/controller identities: `env.py` `8d5cb5…`, `right_env.py` `ebe50f…`, controller `65ed5d…`, `init_trace.py` `bca30f…`.

The warning that `CUDA_VISIBLE_DEVICES` can cause undesired behavior appears in all four runs, including the successful 16-environment run, so it is not a discriminator in this evidence set. Likewise the startup “skipping NVIDIA GPU due CUDA being in bad state” warnings appear in the successful run as well as the failures; they cannot by themselves explain the later SIGSEGV.

Actual differences are:

- `short` versus `train` entry and 16 versus 1024/256 environments;
- V12 versus isolated V13 import root and corresponding `TMPDIR`;
- wrapper deadline/resource-recording versions;
- training-only PPO imports/configuration and `AuditedRightBottleEnv` support;
- later V12/V13 audit support adds left-cup read-only fields, reward diagnostics, a second camera delivery, atomic chunk writes, and first-rollout gates;
- V13 changes the approved batch contract to 256×16 and minibatch 4096.

The observation wrapper used by the successful 16 run and the failed V12 training runs is semantically identical; its saved SHA difference is only a trailing newline. V13 changes only the V12/V13 wording in that file. All entries create the same observation object, wrap `AppLauncher`, call the same `install_initialization()` boundaries, and then construct `AuditedRightBottleEnv`.

The later camera/left-cup/reward objects and recording hooks are installed after `AuditedRightBottleEnv.__init__` calls `super().__init__`. Both SIGSEGV state files show that the base `DirectRLEnv.__init__` had not returned, and no audit directory was created. Consequently those post-super support operations had not executed at the recorded stop. Their modules had been imported and the initialization wrappers were active, so this comparison does not prove the whole support layer harmless; it only rules out claiming that post-construction video/raw/reward writes themselves were executing when the process died.

The successful 16 run's `actual_imports.json` resolves the official H2S2R package through the V10 target path. Read-only `readlink -f` shows that both V12 and V13 `third_party/h2s2r_official` resolve to that same V10 target, and representative PPO source SHA values match. This is a shared deployment layout, not a success/failure difference.

## Resource evidence

The failed 1024 SIGSEGV samples show host available memory about 231.6–236.2 GiB, child RSS up to about 5.66 GiB, zero swap, and GPU1 memory 2073 MiB used with 46466 MiB free. The failed 256 SIGSEGV samples show host available memory about 232.5–236.2 GiB, child RSS about 4.69 GiB, zero swap, and the same 2073/46466 MiB GPU reading. No OOM/fatal marker was recorded. Thirty-second sampling cannot exclude an unobserved transient or a non-capacity CUDA fault, but there is no evidence of memory-capacity exhaustion.

## Exact missing evidence

- no native core/minidump/apport/Kit crash report;
- no native faulting-thread stack or register state;
- no loaded-library map captured at the fault;
- no kernel/NVIDIA Xid record tied to either process in the current package;
- no same-entry, same-support, single-variable comparison between 16 and 256 environments;
- no `actual_imports.json` for either SIGSEGV run because construction never returned;
- no proof that the last visible USD/material activity is the crashing call.

## Minimal future reproduction recommendation (not executed)

If reviewer authorizes a separate native-debug package, use exactly one reproduction of the already-failed V13 256 command and keep entry, support, assets, GPU mapping, physics, algorithm, and seed unchanged. Before launch, arrange approved native crash retention (nonzero core limit or a configured Kit/apport destination) and collect a debugger/native backtrace, module map, and process-correlated NVIDIA/kernel error record. Do not treat disabling observation support, changing environment count, or changing physics as the first intervention: each would confound the reproduction. If the exact run does not reproduce, stop with the captured evidence rather than adding a scale sweep.

## Decision

The zero-physics evidence package is complete within scope. The V13/256 campaign remains closed and runtime-blocked. There is evidence of two native abnormal exits, but there is still no native root-cause evidence. No training or evaluation result exists, so success rate remains undefined rather than 0%.
