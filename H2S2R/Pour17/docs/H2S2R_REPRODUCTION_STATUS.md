> **交接同步 2026-09-13 01:45:00 Asia/Shanghai (UTC+08:00)**：旧Run00–11及StageB结果为历史；当前V13/256关闭周期dump后一次初始化成功，但零PPO/无模型/无自主成功率。请读最新任务README及H2S2R_REVIEWER_HANDOFF_20260912.md。旧训练禁令/放行只能按对应已关闭包理解，新训练预算待决策。

# H2S2R reproduction status and handoff

> **Current handoff — 2026-09-12:** The Run00–11 audit below remains historical evidence. For the post-Run11 Stage A/B campaign, validated Sharpa limit fix, unresolved fixed-C grasp failure and environment/code map, start with [H2S2R_REVIEWER_HANDOFF_20260912.md](H2S2R_REVIEWER_HANDOFF_20260912.md). This is the current reviewer entry point; the 2026-09-07 status below does not describe the latest diagnostic state.

Last evidence audit: 2026-09-07  
Scope: official Human2Sim2Robot bring-up and the local Pour17 Run00--Run11 investigation.

> **Scientific status.** The official H2S2R/FABRICS software stack and the local
> Pour17 integration can run, but none of Run00--Run11 is a successful H2S2R
> reproduction or a valid result for comparison with ours. Runs03--11 are
> diagnostic adaptations. Run10/11 are the cleanest current local single-hand
> hold-0 experiments, but both have zero demonstrated task success.

This document indexes the historical Run00–11 audit. Earlier setup and run reports
remain immutable evidence of what was known at that time. For current post-Run11
status, use the current handoff above and its cited latest artifacts.

## Implementation pointer — 2026-09-08

The approved post-Run11 implementation is documented in
`reports/H2S2R_BOUNDED_IMPLEMENTATION_RESULTS.md`, with source/runtime hashes
in `reports/H2S2R_STAGE_A_REMOTE_RUNTIME_MANIFEST.json` and change decisions in
`reports/H2S2R_ADAPTATION_CHANGELOG.md`. It uses a separately copied remote
working tree and does not alter Run01--11 evidence. Gate A has passed bounded
remote validation; Stage B diagnostic results, including failures, determine
whether either side may enter Stage C. No PPO is authorized before that gate.

## 1. Evidence and provenance

Evidence was ranked in this order: pinned source code and run manifests; logs,
TensorBoard data, checkpoints and replay metadata; local audit reports; visual
observations and chat history. Conclusions below are deliberately conservative.

### Official source pins

| Component | Location / revision |
|---|---|
| H2S2R official FABRICS checkout | `/media/msc-auto/HDD/users/kailang/baselines/human2sim2robot/repo_official_fabrics`, commit `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df` |
| FABRICS H2S2R branch | commit `b56c9b1d7c927f5f8b7ca9464a3b5607a3777dae` |
| Local adapter base revision | Git HEAD `ead0333`; task directories contain preserved, uncommitted run-specific adaptations |
| Official paper | `C:/Users/Dennis/Downloads/Crossing the Human-Robot Embodiment Gap with.pdf` |

### Local and remote evidence

- Monitor manifests: `/home/kailang/experiments/baselines/h2s2r_monitor_records/manifests/01.json` through `11.json`.
- Monitor registrations: `/home/kailang/experiments/baselines/h2s2r_monitor_records/runs/01.json` through `11.json`.
- Preserved adapter generations: `rl_correction_h2s2r_adapter`, then `_v2_lstm_jointprogress` through `_v9_hold0` under `/home/kailang/experiments/baselines/`.
- Setup audit: `reports/h2s2r_setup_status_2026-08-27.md`.
- Run04 physics audit: `reports/h2s2r_run04_black_table_physics_audit_2026-09-03.md`.
- Run06/07 diagnostics: `reports/h2s2r_run06_diagnostics/RUN06_RUN07_DIAGNOSTIC_REPORT.md` and `analysis_summary.json`.
- Collision/penetration audit: `reports/h2s2r_v7_collision_integrity_20260905/summary.json`.
- Run-specific local snapshots: `h2s2r_right_only_03`, `h2s2r_right_only_04`, `h2s2r_left_only_05`, `h2s2r_v8_patch`, and `h2s2r_v9_hold0_patch`.

Run00 is a reconstructed label: no formal `00.json` manifest was found. It
denotes the family of official-stack and adapted-environment smoke tests, not a
comparable training run.

## 2. Run lineage

```text
Run00 official/adapted smoke
  -> Run01 bimanual MLP
  -> Run02 bimanual official recurrent stack + local joint rewards
  -> Run03 right-only isolation
  -> Run04 right bounds expansion
  -> Run05 left-only + then-current ours physics
  -> Run06/07 released-code clock and termination, one side per run
  -> Run08/09 side-specific start, zero center, and uniform physics
  -> Run10/11 remove the local 15-step object hold and retrain each side
```

| Run | Parent | Meaningful change and reason | Outcome, support and limitation |
|---|---|---|---|
| **00** | None | Official H2S2R/FABRICS/cuRobo smoke plus a 16-env Pour17 PPO integration smoke. Official smoke verified 11-D FABRICS actions, 144-D observations and one PPO update; adapted smoke used 342-D bimanual observations. | Infrastructure only. It proves that the basic controller/simulator/PPO path executes, not that an H2S2R policy learns or reproduces published results. Exact mapping from the conversational `Run00` label is not recoverable. |
| **01** | Run00 integration | Bimanual, 512 envs, seed 42, feed-forward MLP `[512,256,128]`, 342-D observation and 22-D action. Per-side released-style terms were averaged and a shared AND clock was used. Training used a 15-step object hold; early replays used hold 0. | Reached epoch 5000 / roughly 20M agent steps, then exit 137. Left cup behavior improved temporarily near epoch 3000 and later degraded; right bottle grasp failed. Obsolete as a baseline because the network/PPO and bimanual construction are nonofficial. |
| **02** | Run01 | Switched to the public recurrent stack: MLP `[512,512]`, LSTM 1024, horizon/sequence 16, 1024 envs. Still bimanual and added local joint-progress and joint-success rewards to discourage sacrificing one side. | Stopped near epoch 1789; one epoch-1000 checkpoint. Demonstrated that the recurrent stack could run, but the reward and centralized bimanual method differ from the official single-hand method. |
| **03** | Run02 code line | Isolated right/bottle learning; fixed the left side and ignored the cup. Returned to single-hand dense reward to test whether the official-style method could learn the difficult side alone. Still used start frame 32 and the earlier simplified clock/action mapping. | Stopped at epoch 2056 with checkpoints every 500. Failed to grasp or lift the bottle. Diagnostic only because the right reference should start at frame 14. |
| **04** | Run03 | Expanded right physical action bounds to cover measured near-grasp palm/PCA values, testing whether an unrepresentable reset workspace caused failure. Absolute 6-D palm + 5-D PCA semantics were retained. Start remained 32 and hold remained 15. | Completed epoch 1000. No successful bottle grasp. Establishes that range coverage was necessary but not sufficient; it did not test the corrected right start or fully restored clock. |
| **05** | Run04 adapter line, switched side | Isolated left/cup learning, expanded left bounds, used the black table and the then-provided physics values (`0.1 kg`, object/pad friction 5, body friction 0.2 and supplied PD gains). Start 32 and hold 15. This tested whether the previously easier side remained learnable under ours-aligned physics. | Completed epoch 1000 with no success. This is the first preserved run with conspicuous left visual penetration. Because side, bounds and physics changed together, it is not a one-variable causal experiment. |
| **06** | Run05 | Left-only. Restored the released-code floating reference index, per-episode speed in `[0.5,1.0]`, clock freeze during warmup, and reference-end termination. Removed joint rewards fail-closed and added diagnostics to distinguish task learning from survival. Start 32, hold 15. | Completed epoch 1000; success remained zero. Tracking error improved modestly, but per-step tracking reward decreased while episode length and return grew. This is evidence of partial/local optimization, not task success. |
| **07** | Run05, sibling of Run06 | Applied the same released-clock/termination and diagnostic changes to the right side. The monitor label contains `Run06`, but its manifest identifier is 07. It still used start 32 and hold 15. | Completed epoch 1000; success zero, bottle height stayed near reset, tracking error did not improve materially and fingertip distance worsened. The rising return primarily followed longer episodes. |
| **08** | Run06 plus the v7 uniform-physics code generation | Left-only, side-specific start 32, separately calibrated zero-action center at the left near-grasp, uniform local comparison physics (`0.5 kg`, friction 1), restored clock/horizon, hold 15. These changes tested start/action alignment under the final shared physical constants. | Completed epoch 2000. Tracking error improved, but all milestone replays were unsuccessful. Hold-15 training and replay still contained the local object-overwrite artifact. |
| **09** | Run07 plus the same v7/v8 corrections | Right-only with corrected start 14 and an independently calibrated right zero-action center. Uniform physics and hold 15. | Stopped near epoch 1584 with checkpoints through 1500. Replays remained near reference 38--39/127 with no bottle grasp. Exit 134 followed the requested stop/Isaac teardown and is not evidence of PPO numerical failure. |
| **10** | Run08 | Removed the local object hold from the first training step while retaining the left single-hand setup. This isolated whether the post-physics object rewrite caused penetration and prevented learning. | Stopped near epoch 1881; checkpoints at 500/1000/1500. The sustained penetration artifact disappeared, fingertip distance and tracking error improved, but the policy did not grasp or lift the cup. If resumed, the agreed restart point is epoch 1000. |
| **11** | Run09 | Right-side hold-zero counterpart of Run10. | Stopped near epoch 1382; checkpoints at 500/1000. Tracking error changed only slightly, fingertip distance worsened and no grasp/lift occurred. Exit 134 was operational teardown after stop, not a scientific result. |

All milestone videos are deterministic, single-episode qualitative diagnostics.
They restore the recurrent player and observation normalization, but they are
not substitutes for the planned 128-episode diagnostic or 512-episode formal
evaluation. Run10 videos exist for epochs 500/1000/1500; Run11 videos exist for
epochs 500/1000.

## 3. Durable technical findings

### 3.1 Three specifications must not be conflated — VERIFIED

The paper presents the object-centric reward as
`exp(-10 * object_pose_distance)` against `T_target[tau+t]`, and describes
`Dmax=0.25 m` object-target and palm-object termination. Appendix E reports
4096 environments, 15 Hz control, approximately 0.6B action frames and 5--24
hours on one A100 40GB.

The pinned released code contains additional operational behavior:

- reward is gated by mean fingertip-object distance below 0.3 m;
- reward is multiplied by 0.1 when object-target error exceeds 0.2 m;
- a floating reference index advances only while error is at most 0.2 m;
- reference speed is sampled in `[0.5,1.0]` and scaled from 30 Hz reference time;
- far fingertips can terminate an episode;
- the configured object-goal early-reset threshold is effectively disabled by a very large value.

The baseline target is the pinned released implementation, not a reimplementation
of the paper equation alone. The discrepancy must nevertheless be disclosed.
There is also an unresolved official-source discrepancy: Appendix E states a
learning rate of `5e-4`, while the audited released recurrent configuration uses
`1e-4`. A fresh native reproduction must record which source is followed.

### 3.2 Object hold introduced an invalid overwrite path; general penetration remains unresolved

**2026-09-12 scope correction:** the user reports visible penetration in later hold-zero runs, despite its absence in the initial version. The historical controlled observation below does not establish a unique cause or a complete fix. Exact affected versions/frames and residual penetration remain to be reconciled. Presence of collision shapes/materials/contact forces is not proof that all physical trajectories are valid. See `H2S2R_TRAINING_READINESS_REVIEW_20260912.md`.

The hold was not part of official H2S2R. The local implementation performed a
physics/contact step and then rewrote the object root pose to its reset pose and
zeroed its velocity. This can undo PhysX's collision response and relocate a
fixed object into fingers that have already moved.

Collision shapes, materials and contact responses were present, so the primary
problem was not globally disabled collision. Controlled hold-15 versus hold-0
replay of the same failed policy removed the conspicuous sustained penetration
when the overwrite was disabled. Policy wrist motion could exacerbate the
artifact, but action magnitude is not the criterion: any visible penetration
created by the post-physics rewrite is unacceptable.

### 3.3 Necessary corrections did not produce task success — VERIFIED

- The sides require distinct starts: left/cup frame 32; right/bottle frame 14.
- The old action ranges did not cover every near-grasp palm/PCA value.
- Run08+ independently centered each side's normalized zero action at its own
  measured near-grasp; the two calibrations were not shared.
- Run10/11 removed the object hold from training.

Despite these corrections, task success remained zero. Therefore wrong start,
range coverage, zero-action centering and hold removal are individually
important but not sufficient explanations or fixes.

### 3.4 Return growth did not demonstrate desired learning — STRONGLY SUPPORTED

Run06 left first-to-last-window statistics showed return and episode length
increasing while success stayed zero; Run07 was stronger evidence, with return
and episode length increasing while bottle height stayed near reset and
fingertip distance worsened. The policies could obtain more cumulative reward by
remaining alive or tracking permissive parts of the reference without grasping.

Reaching reference end is also not synonymous with success: the released clock
can advance through successive 0.2 m gates, while the diagnostic final-success
condition is stricter. Mean reward, episode duration and reference index must
never be reported as task success.

### 3.5 Training budget is an unresolved confound, not a conclusion — UNKNOWN

The paper reports roughly 0.6B frames per policy. Local diagnostic runs used a
small fraction of that budget. It is therefore invalid to conclude that enough
training would certainly fail. Conversely, zero success, plateaus and known
parity gaps make it equally invalid to claim that simply extending these runs
would solve the problem.

## 4. Rejected or downgraded hypotheses

| Hypothesis | Evidence-based disposition |
|---|---|
| The reset pose itself caused penetration | Rejected as the primary cause. Frame zero was generally a visually valid near-grasp; penetration appeared after actions and the object rewrite. |
| The dashboard/MP4 used the wrong environment | Rejected as the primary cause. Replay metadata, source hashes and runtime settings matched their runs, and the controlled hold change altered the observed behavior. |
| Collision geometry was absent | Rejected as the primary cause. Collision prims, material bindings and contact forces were verified. Exact cooked hulls and signed penetration depth remain unmeasured. |
| A 1.10 m decoded target/action saturation was the root cause | Downgraded. It is a symptom of a failed aggressive policy and can worsen contact, but it does not explain the post-physics rewrite mechanism. Action geometry can still affect learnability. |
| Right failure was entirely caused by using frame 32 | Rejected as sufficient. Run09/11 used frame 14 and still failed. |
| Extending action bounds alone would fix learning | Rejected as sufficient. Run04 covered the reset, and Run08/09 additionally centered each side; none succeeded. |
| High friction/low mass explains left success or failure | Not established. Run05 changed multiple variables, and later uniform-physics runs still failed. |
| A stricter clock threshold necessarily improves learning | Untested and method-changing. It could close a loophole or make the signal too sparse; it does not belong in the first released-code baseline. |
| Missing grasp/contact/penetration reward is an implementation bug | Rejected as an official-parity claim. The official forced-reference method intentionally lacks these terms. Adding them would define an adapted method. |
| More epochs alone will fix the current runs | Unsupported. Official scale is much larger, but current parity gaps and zero-success plateaus prevent extrapolation. |

## 5. Official versus local modification map

| Component | Official paper / pinned released code | Local Pour17 status | Classification for a fresh reproduction |
|---|---|---|---|
| Object reward | Object-pose tracking; released code adds the 0.3 m fingertip gate and 0.2 m downweight | Single-hand Run03+ largely preserves the released-code term | Preserve released code; disclose paper/code difference |
| Reference progression | Paper shows `tau+t`; released code uses a floating, conditional, randomized-speed clock | Simplified through Run05; restored in Run06+ | Preserve released code first |
| Policy | Single-hand MLP 512x512 + LSTM 1024 | Recurrent stack used from Run02 | Preserve |
| PPO | Paper and released config disagree on learning rate | Current local recurrent runs followed released config | Resolve and record before native run |
| Embodiment | KUKA arm + Allegro hand | Dual-arm Vega + Sharpa | Necessary, high-impact adaptation |
| Observation/action | Official 144-D actor observation and 11-D absolute FABRICS action | 171-D observation and 11-D action per side due to 29 DOF | Requires field-by-field parity audit |
| Pre-manipulation pose | HaMeR/depth estimate retargeted with cuRobo | Ours G2/canonical near-grasp | Major input replacement; do not call it the official pose |
| Object trajectory | FoundationPose estimate from the demonstration | EgoDex/ours estimate, constant-SE(3) anchored to local reset | Major data adaptation |
| Initial randomization | Object XY/yaw sampling and relative wrist transform; extensive domain randomization | No complete equivalent was verified | Restore/verify in native reproduction before interpreting learning |
| Bimanual training | Not provided | Central policy, mean reward, shared AND clock and joint event rewards in Run01/02 | Archived method exploration; not main baseline |
| Single-hand isolation | Official task is single arm/hand/object | From Run03, inactive arm fixed and other object ignored | Reasonable diagnostic bridge, not final bimanual result |
| Start selection | Derived from object velocity and a pre-offset | Manual left 32/right 14 | Necessary dataset adaptation; disclose |
| Zero-action map | Linear normalized range; zero is the range midpoint | Run08+ piecewise map places each side's near-grasp at zero | Local methodological modification; do not auto-carry |
| Physics | Large domain randomization | Later runs fix `0.5 kg`, friction 1 and ours PD/table | Fair external-condition experiment, not an official native configuration |
| Warmup hold | None | Hold 15 in Run01--09; removed in Run10/11 | Do not carry forward |
| Success | Task-specific outcome definitions | Diagnostic reference-end + final error <5 cm; external G3 not formally evaluated | Replace with an agreed task evaluator for formal comparison |
| Instrumentation | Standard training logs/checkpoints | Extra reset, clock, error, distance, height, action, video and monitor outputs | Retain when nonintrusive |

## 6. Unresolved questions and discriminating checks

All items in this section are **UNKNOWN** unless stated otherwise.

| Issue | Current evidence | Leading explanations | Already ruled out / downgraded | Discriminating check |
|---|---|---|---|---|
| Native official validity | Official/FABRICS/PPO smoke succeeds, but no native task success has been measured locally. | The native stack may reproduce normally; alternatively an installation/config mismatch may appear only in full evaluation/training. | Smoke success does not prove policy success or convergence. | Evaluate supplied checkpoints on their exact KUKA-Allegro tasks and run one bounded native training reproduction with immutable config/provenance. |
| G2 pre-manipulation quality | Each reset is visually near the intended object, but neither hold-zero side learned a successful grasp. | The pose may be kinematically plausible yet outside the useful exploration basin for H2S2R; controller or contact geometry may make finger closure ineffective. | Reset pose is not the primary cause of the prior sustained penetration artifact. | In the unchanged hold-zero environment, inspect scripted/local finger closure and small wrist perturbations around each G2 state before another PPO run. This is a diagnostic, not a replacement grasp generator. |
| Reference/reset alignment | Left can progress to reference end without final success; right stalls near an early reference segment. Starts 32/14 are manually reviewed. | Permissive early target motion, incorrect anchoring/coordinates, lift-onset mismatch, or a genuinely difficult reward basin. | Shared frame 32 is no longer the explanation for Run09/11. | Plot per-frame SE(3)/anchor displacement and evaluate the reset object against every early target; audit transformations against raw estimated trajectories. |
| Missing official randomization | The paper specifies object XY/yaw perturbation, relative wrist relocation, observation/action/dynamics noise and force perturbations; no complete local equivalent is verified. | Missing exploration/robustness mechanisms may materially reduce learnability. | Adding task-specific grasp reward is not a parity-preserving substitute. | Trace every official randomization from config to runtime and run a native parity probe before deciding which parts can be transferred to Pour17. |
| Embodiment/controller parity | The adapted robot has 29 DOF per side and 171-D observations instead of KUKA-Allegro's 23 DOF/144-D actor observation. | PCA basis, action scaling, PD/FABRICS task maps or observation ordering may create a different optimization problem. | Merely expanding bounds or centering zero action was insufficient. | Produce a field-by-field observation/action/controller parity table and compare reset-time random-policy target/contact statistics in native and adapted environments. |
| Optimization budget | Local runs are far below the paper's approximate 0.6B-frame scale, but success is zero and some metrics plateau. | More frames may be required; or the current adapted problem may be mis-specified and extra training would reinforce a local optimum. | Neither “training is definitely sufficient” nor “more training definitely fixes it” is supported. | Revisit scaling only after native validity and parity checks; then use task-success windows and multiple seeds, not return alone. |
| Bimanual composition | Current clean experiments are independent single-side policies. | Separate policies may be synchronizable, or their actions/reference clocks may interfere when combined. | Independent single-side training does not itself establish a bimanual baseline. | First obtain successful side policies; then specify and test a frozen synchronization/composition rule against the common bimanual evaluator. |
| Convergence criterion | No project-wide criterion is frozen; reward/reference progress can rise without success. | A task-success plateau plus a minimum budget and stability window is likely necessary. | Mean return or reference end alone is invalid. | Agree the success metric, rolling window, minimum training exposure, seeds and stopping tolerance before the next formal run. |
| Residual collision integrity | Hold zero removed the observed sustained penetration; collision shapes/materials/contact were present. Signed penetration and full self-collision state remain unmeasured. | Residual visual interpenetration could arise from collision-hull/visual-mesh mismatch, solver tolerance or disabled self-collision. | Globally missing object collision is not the primary cause of the old artifact. | Export cooked collision geometry/runtime flags and measure signed separation/contact impulses in a bounded, nonlearning contact probe if the artifact recurs. |

## 7. Handoff boundary for a fresh official reproduction

The next effort should start from the untouched official checkout at the pinned
commit, not from the v9 hold-0 Pour17 adapter.

1. Revalidate the exact official environment, assets, pretrained checkpoint,
   recurrent player and task success metric in the native embodiment.
2. Reproduce a bounded native training run using a manifest that states whether
   paper or released-code hyperparameters are used.
3. Build a parity matrix before moving to Pour17: input trajectory, pre-pose,
   reset distribution, observation, action, controller, reward, clock,
   termination, domain randomization and evaluator.
4. Introduce only necessary external-condition adaptations one at a time and
   preserve a native control at every stage.
5. Carry over monitoring and diagnostics only when they do not affect policy
   observations, reward, reset, clock or optimization.

The following local decisions must not be inherited silently: G2 reset, manual
left/right starts, side-specific zero-action centering, fixed `0.5 kg / friction
1` physics, inactive-arm isolation, hold zero, and the local success diagnostic.
Each may be useful, but each must be named and justified as an adaptation.

## 8. Current scientific claims allowed

Allowed:

- The official software/controller stack and the Pour17 integration execute.
- The current adapted H2S2R-style system did not learn successful Pour17 cup or
  bottle manipulation within the tested runs and settings.
- The local 15-step post-physics object overwrite caused a visible penetration
  artifact and has been removed from Run10/11.
- Several necessary parity corrections were made, but none was sufficient to
  yield success.

Not allowed:

- “Official H2S2R fails on Pour17.”
- “Run10/11 are reproduced H2S2R baselines.”
- “More training definitely will” or “definitely will not” solve the failure.
- Any success-rate or sample-efficiency comparison against ours from Run00--11.
- Treating reference progress, return growth or a single replay as task success.

## 2026-09-08 implementation pointer

The bounded implementation session did not run native reproduction or PPO.
Its Stage A static manifest, adaptation ledger, runtime-gate decision, and
explicitly unrun validations are in
`reports/H2S2R_BOUNDED_IMPLEMENTATION_RESULTS.md` and
`reports/H2S2R_STAGE_A_STATIC_MANIFEST.json`. Historical conclusions above are
unchanged.

## 2026-09-08 bounded-session completion pointer

The remote access addendum enabled an audited, hold-zero single-side Stage A
and Stage B diagnostic in a separate implementation copy; it did not run
native reproduction. Gate A passed for that bounded diagnostic. Both Stage B
sides completed their capped probes but **did not pass Gate B**: force
calibration and the required repeat/video physical witnesses are absent, and
the right side produced no qualifying lift. Consequently no Stage C shaping,
PPO, or Stage D experiment was started. The complete evidence, zero training
cost, and explicitly unrun validations are in
`reports/H2S2R_BOUNDED_IMPLEMENTATION_RESULTS.md`; source/runtime hashes are
in `reports/H2S2R_STAGE_A_REMOTE_RUNTIME_MANIFEST.json`, and every change is
listed in `reports/H2S2R_ADAPTATION_CHANGELOG.md`. Historical conclusions
above remain unchanged.

## 2026-09-08 Stage B gate-status correction

The later Stage B review found that the original scripted lift replay replaced
the selected candidate wrist pose. Therefore its left/right controllability
conclusions are not final: left Gate B is **PENDING**, and right Gate B is
**INVALID/PENDING**, not a confirmed interface failure. The bounded repaired
validator is implemented, but the required replay/calibration cannot construct
the unchanged Isaac environment because IsaacLab cannot open its configured
`/tmp/isaaclab/logs/...` file (permission denied). No alternate runtime,
controller/action/physics/reset modification, PPO, or Stage C work was used to
work around that infrastructure blocker. See
`reports/H2S2R_BOUNDED_IMPLEMENTATION_RESULTS.md` for the exact remote status
path and validation boundary. Historical Run00--11 conclusions above remain
unchanged.
