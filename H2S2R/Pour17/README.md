# Pour17 H2S2R adapted baseline — 当前交接入口

交接截止：2026-09-13 01:45:00 Asia/Shanghai (UTC+08:00)。本轮只做文档与源码快照整理，没有实验、部署或GitHub上传。

## 当前结论与阅读顺序

本次交接状态：V13/256 + 独立周期traceback禁用overlay，已实现、已部署，且一次GDB初始化成功；尚未完成训练集成或任何PPO更新。最新运行于2026-09-13 00:32:20–00:34:42北京时间（2026-09-12 16:32:20–16:34:42 UTC），环境在观测135.312秒返回，进程141.872秒，护栏在策略前退出。控制/更新/epoch/transitions均0，无checkpoint/normalizer，自主评测有效分母0，成功率未定义。左杯穿模不能由本次初始化判定解决，独立左杯track结论不在此更改。

先读本页，再读 `docs/H2S2R_CURRENT_BASELINE_STATUS_20260912.md`、`docs/H2S2R_REVIEWER_HANDOFF_20260912.md`、`docs/H2S2R_TRACEBACK_INTERVENTION_20260913.md` 和 `evidence/traceback/RESULTS.md`。上传文件及逐文件服务器SHA对应见 `UPLOAD_MANIFEST.md`、`SOURCE_IDENTITY.json`；服务器访问需项目账号授权，不在仓库分发凭据。

本文所说仓库根为建议的 `H2S2R/Pour17/`，避免与Clean3、ViViDex及ours冲突。上级旧RL-Correction README描述ours，不是本baseline入口。当前是右瓶单侧pilot（左侧固定命令），不是完整双手Pour17比较结果，不承担5–7.5天完整原生复现。

## 同步基线与增量

可核实的旧本地任务README基线为2026-09-08；其“缺少single-side入口”描述针对旧checkout，现在被本页替代。一般status顶部此前停在V13 SIGSEGV；reviewer handoff已收到最新初始化结果但其他入口未同步。没有上次GitHub上传commit/清单，不能声称已核实远端仓库同步日期；本次提供完整选定集合而不是猜测Git差分。

9月8日至本截止的关键变化：右Sharpa手限位零内缩补丁隔离集成；控制节奏修为每control 1次set_targets、12次advance/submit；hold0、黑桌及左委托验证修正；16环境×256controls录像支持完成（不等于左杯穿模已解决）；1024训练入口超时/崩溃，256配置CPU/部署通过但原入口仍崩溃；GDB捕获CPython dump_frame，完整采集因脚本错误失败；随后修复GDB采集并仅关闭周期dump，一次256初始化返回。历史源命令抓举和FABRICS回放是特权诊断，不是已训练policy评测。

## 代码身份与依赖

- 旧本地checkout：`D:/UCBP/RL/Baselines/RL-Correction-H2S2R`，branch `baseline/h2s2r-dexmate-single-side`，HEAD `ead03335336857cfb0585e6e4ee1ebcb36e98cb1`。它缺少现行入口，不能当作V13；本次不覆盖其旧功能代码。
- 当前远端源码根：`/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v13_256_pilot_20260912`。本次只读确认没有.git；无可验证的V13 branch/commit，用 `SOURCE_IDENTITY.json` 的逐文件SHA标识。`source_v13`取回内容逐字节保存，不是代码修复。
- 历史V12根：`/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912`。禁止用V12标签冒充V13批量适配。
- 成功初始化额外加载 `/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/traceback_intervention_01/overlay/runtime/training_pilot_20260912/resume_observation.py`，SHA `23122aad8cdb8d7ebfd1e31de584f9ded43ac14b34e45cd55fdccef0381dda30`。原V13文件仍SHA `7fe7b1371a4c2aa493454ac74e27a84c8292f6f78e084673e674744f3030b9ab`。两份在仓库分别保存；设置开关但仍加载原文件不会生效。诊断guard使用sys.settrace，在line585前退出，不是训练启动器。
- Python `/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python`，运行日志3.11.15；Isaac Sim5.1.0、IsaacLab `/home/kailang/opt/IsaacLab-5.1`、Kit107.3.3、驱动580.173.02、报告CUDA13.0、GDB15.1。649扩展集合身份见 `evidence/runtime/RESULTS.md`。没有完整可重建conda/pip lock；这些版本来自既有运行证据，不是本次全依赖重装验证。
- 官方PPO包随快照保留 `third_party/h2s2r_official`，来源commit `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df`，保留SOURCE与MIT LICENSE。
- FABRICS外部源码 `/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src`，历史pin `b56c9b1d7c927f5f8b7ca9464a3b5607a3777dae`；本轮未重核整棵外部checkout，补丁检查特定基础文件SHA。保留 `third_party/FABRICS_LICENSE` 与THIRD_PARTY_NOTICES，不把其许可替换成MIT。机器人/数据资产沿用原许可和项目访问约束。

## 数据和外部文件准备（先检查，不自动生成新数据）

从获授权服务器复制或使用原路径，保持目录结构并核对已有manifest：

| 资源 | 绝对路径 | 用途/核验 |
|---|---|---|
| frozen Pour17 bundle | `/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17` | world/world_manifest.json、canonical_reset_v1.json、world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd、world/objects/object_1.usd、world/cache/object_0.usd、perception/pour17_perception.npz及reference匹配reset SHA的npz；必须整体保留USD依赖 |
| 两侧PCA | V13根下 `tasks/h2s2r_pour17/artifacts/synergies/right_synergy_pca5.npz` 与 `left_synergy_pca5.npz` | 现行5D参数；同目录synergy_manifest.json已收入仓库，列input与outputs SHA；不重新fit替换 |
| 固定躯干派生资产 | V13根下 `assets/vega_1p_sharpa_fixedtorso.usd` | 配置继承构造时依赖；复制时保留其USD引用资产，不能退回MagicSim旧默认路径 |
| FABRICS包 | `/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src` | 加入PYTHONPATH，使用已有部署，不能因缺依赖换库 |
| 训练/诊断证据 | `/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/` | right_training_result_01、right_training_256_01、native_debug_01、traceback_intervention_01；远端原始日志/转储，相关索引收入evidence |
| 本机已有真实录像 | `D:/UCBP/reports/H2S2R_TRAINING_PILOT_20260912/batch_delivery_01/FILES.md` | 16环境短验证录像入口，不是训练成果；本次初始化无视频 |

phase1_g2只取冻结参考的单个初态行，不把成功源整条命令作为policy。输入estimated、reference_start_index14。右action11维（腕6+手PCA5），手22关节，保留FABRICS。右zero-margin补丁文件在 `runtime/cadence_20260912_deploy`，训练构造前加载；左侧保持原实现，不代表左手缺陷已修。

## 已执行命令与使用限制

以下是已启动过的**历史256训练子命令**，结果为初始化SIGSEGV、零学习，不是可直接重启授权；完整环境与argv见 `evidence/v13/launch_manifest.json`：

```bash
cd /home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v13_256_pilot_20260912
/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python tasks/h2s2r_pour17/train_right_lstm.py --bundle_root /home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17 --name V13_RIGHT_256_PILOT --num_envs 256 --seed 42 --reference_start_index 14 --training_pilot --pilot_output_root /media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/right_training_256_01/train01 --device cuda:0 --headless
```

该命令由已存 `runtime/training_pilot_20260912/launch_with_root.py`监督，不能脱离超时/资源/额度配置手工运行。关键PYTHONPATH为V13根、FABRICS/src、V13/third_party/h2s2r_official。现GPU规则只以仓库 `../../GPU_POLICY.md`（本机D:/UCBP/GPU_POLICY.md）为准，允许GPU0/1及满足预算的共享；准入后在本任务子进程设RL_ISAAC_NO_GUARD=1并记录CUDA_VISIBLE_DEVICES映射，不改全局设置。旧manifest固定GPU只是历史记录。

成功初始化命令使用同一V13/256子入口，由远端 `traceback_intervention_01/support/launch_intervention.py`安排GDB、overlay和guard；实际脚本在 `diagnostics/traceback_intervention/`，已运行结果见 `evidence/traceback/launch_status.json`。脚本含防重跑和绝对路径，不能把它当可迁移的一键脚本或生产训练入口。只设置H2S2R_DISABLE_PERIODIC_TRACEBACK=1不足以让原V13训练自动获修。

终评入口 `runtime/training_pilot_20260912/endpoint_eval.py --bundle_root <bundle> --checkpoint <last> --out <new-output>`：仅源接口/CPU检查过，未实际终评，尚无可用last。现行方案16episode、seeds1000–1015、deterministic/frozen normalizer；不提供伪造checkpoint命令。可视化使用训练/终评共用camera_delivery；本次未渲染新视频，不引用历史录像为当前训练成功。

PPO配置256×horizon16、minibatch4096、mini_epochs4、seq_length16，1500epoch/6h上限，最多6,144,000 transitions只是预算，不是实际训练量。保留tracking reward，无新增grasp reward、无22D方案；较原1024缩小batch/样本预算属于明确适配，不是完全原论文训练协议。

## 截止状态、下一步与缺口

旧1024包、256训练包、原生捕获包及周期dump干预包均已按分支关闭；剩余名义次数不构成重启资格。下一方向已讨论为将最小诊断缓解用于256训练，但**新的训练集成/启动预算尚未发布，待reviewer决策**；本轮文档任务不放行它。接手先审阅最新INIT_RETURNED与overlay加载身份，再由负责Pour17的reviewer明确新包，不能把本页命令当授权。

2026-09-13 01:36:36北京时间只读进程快照：未见kailang用户命令行匹配V13/traceback_intervention的Python进程；仅此范围与时间，不代表全服务器永久空闲。此次未核对其他track运行状态。

仍缺：长期/非GDB训练稳定性，左右首轮物理有效性，任何自主训练checkpoint与终评；完整依赖lock、跨机器路径迁移验证；目标GitHub现有目录/上次上传commit未核实。所有历史论文/诊断结论按其原scope，不将失败初始化记0%。共享文档GPU_POLICY只上传仓库根一份；Clean3/左杯文档由对应负责人汇总，不在此覆盖。

---

## 历史本地README（截至2026-09-08，下面不是现行执行指令）


# H2S2R Pour17 baseline

## Current implementation status (2026-09-08)

This checkout has no locally validated runnable single-side H2S2R path. The
main tree is missing the single-side wrappers, recurrent PPO adapter, and
recurrent PPO YAML; the historical v6/v7/v9 directories are source candidates,
not a verified overlay. The authoritative Stage A inventory and bounded result
are:

```
D:/UCBP/reports/H2S2R_STAGE_A_STATIC_MANIFEST.json
D:/UCBP/reports/H2S2R_BOUNDED_IMPLEMENTATION_RESULTS.md
```

When a complete verified runtime is available, the eligible diagnostic path
uses hold zero from the first control step and executes a physical single-side
probe before PPO. `tracking_only` is the required baseline mode. The proposed
`tracking_plus_grasp_potential` mode is not implemented and must not be added
until the relevant side passes Gate B. Single-side diagnostics and evaluation
are not formal bimanual Pour17 results.

The historical material below is retained as evidence only. Where it conflicts
with this status or the 2026-09-08 handoff, it is not an active instruction.

This adapter retrains Human2Sim2Robot (H2S2R) for the synchronized bimanual
Pour17 task. It aligns the external benchmark conditions with ours while
preserving H2S2R's policy-facing method.

## Frozen experiment contract

| Item | Frozen value |
|---|---|
| Simulator / robot / assets | Pour17 bundle's Isaac world and 58-DoF Vega + two Sharpa hands |
| Reset | Ours `g2` pre-manipulation entry: both hands at the interaction-first grasp row, both objects at the same physical reset poses; no reset randomization |
| Warmup | Training: both methods use `POUR_HOLD_K=15`; formal evaluation: both use **`POUR_HOLD_K=0`** |
| Horizon | Evaluator-enforced 903 control steps at 20 Hz |
| Evaluation policy | Deterministic mean, 512 episodes, seeds `20260829+i` |
| Phase-one success | Frozen evaluator G3 (pour geometry held); G1/G2 are preset by the shared already-grasped reset |
| Main input | `Estimated`: the same model-estimated EgoDex hand/object inputs |
| Oracle | Sensor-measured GT only, with provenance manifest and SHA-256; fail closed |

The `STATUS` sentence in the 2026-08-29 bundle's
`world/canonical_reset_v1.json` still says warmup needs confirmation. That
sentence is stale: the project decision is now final and the adapter rejects any
canonical reset whose evaluation `POUR_HOLD_K` is not zero.

This is deliberately the short-term **pour-only** protocol. It skips approach,
grasp acquisition, placement, and return-to-stance. A later full-sequence run
will switch both methods back to `canonical_t0`; its G4 result must not be mixed
with the phase-one G3 result.

## What remains H2S2R

- One H2S2R FABRICS controller per side.
- Per-side action: 6-D palm pose plus 5-D hand PCA; joint action is 22-D in
  `[right 11, left 11]` order.
- Per-side observation keeps the upstream field order, expanded for Vega-Sharpa
  kinematics to 171 values; joint observation is 342-D in right/left order.
- Per-side reward is upstream object keypoint tracking gated by fingertip-object
  distance. The joint reward is the arithmetic mean of the two terms, which
  preserves the original one-side reward scale.
- The shared reference clock advances only when **both** cup and bottle
  keypoint errors are at most 0.2 m.
- Upstream PPO hyperparameters are copied from `CrossEmbodimentPPO.yaml`.
  The local PPO already applies the same reward scale of 0.01.

No ours hand-trajectory reward, confidence-weighted object reward, GraspPose,
affordance/contact supervision, residual action, curriculum, or privileged
observation is imported into H2S2R training.

## Necessary, disclosed adaptations

1. The original one-hand/one-object task is made synchronized bimanual with two
   independent FABRICS instances and one centralized policy.
2. Kuka-Allegro kinematics are replaced by the shared Vega-Sharpa embodiment.
   The public FABRICS implementation is not modified.
3. Each estimated object trajectory receives one constant SE(3) transform that
   aligns its first pose with the shared g2 simulator reset. Relative motion and
   perception noise are preserved. Validity masks are report-only and never gate
   the reward or reference clock.
4. The reset imports exactly one 58-DoF row from the frozen ours tape: the
   interaction-first row used by ours' `g2` entry. This is a shared external
   initial condition, not a tracking target; no other robot row, confidence, or
   ours reward term enters H2S2R. Object poses use the physically corrected
   values in `canonical_reset_v1.json`.
5. Ours' 15-step post-reset object clamp is mirrored during H2S2R training so
   both policies get the same finger-settling interval. Evaluation disables it
   for both methods. Because reset is now near the objects, upstream H2S2R's
   original 0.3 m far-fingertip termination is restored during training.
6. Phase-one evaluation presets G1/G2 and counts G3 as task success. The
   evaluator's G2 lift intervention and G4 placement/return requirements belong
   to the later full-sequence experiment, not this pour-only result.

## Input regimes

Calibration GT (camera intrinsics/extrinsics/timestamps), scene meshes, physical
parameters, simulator state, and evaluator state are shared. Task supervision
is not silently upgraded.

- `Estimated` is the main comparison. It reads
  `perception/pour17_perception.npz`, whose hand and object values are model
  estimates. Camera calibration may be dataset/device provided.
- `Oracle` is optional. It accepts only a separate archive declared as
  `sensor_measured_ground_truth` by an `h2s2r_input_provenance_v1` manifest whose
  archive SHA matches. Dataset annotations, pseudo-GT, ours IK/reference tapes,
  and the bundle's `reference/*.npz` are rejected as Oracle supervision.

The frozen `reference/` tape supplies one disclosed robot row for the shared g2
reset and is otherwise consumed only by the common evaluator. Training never
uses its robot trajectory, confidence weights, or certification IK as H2S2R
supervision.

## Preparation and verification

Run from the adaptation repository root in the A6000 Isaac environment.

```bash
python -m tasks.h2s2r_pour17.prepare_synergies \
  --bundle_root /path/to/pour17 \
  --output_dir tasks/h2s2r_pour17/artifacts/synergies

python -m unittest discover -s rl_rebuild/baselines/h2s2r/tests -v

python -m tasks.h2s2r_pour17.selftest_progress_batch \
  --bundle_root /path/to/pour17
```

The PCA files must be fitted from the declared input archive. Formal training
checks the archive SHA embedded in each synergy. The analytic per-finger basis
is only a bring-up fallback and is not permitted by the default configuration.

## Training

First run a small integration smoke after an A6000 is free:

```bash
SHARPA_WANDB=0 python -m tasks.h2s2r_pour17.train \
  --bundle_root /path/to/pour17 \
  --num_envs 16 --max_agent_steps 128 --headless \
  --name H2S2R_Pour17_IntegrationSmoke
```

Then choose the largest stable vectorization on one A6000 and keep it fixed
across seeds. The default is 512 environments; upstream H2S2R used 4096 for its
lighter single-hand scene. Formal runs use multiple fixed seeds and train until
the external success metric plateaus; the large `max_agent_steps` value is only
a safety ceiling.

```bash
SHARPA_WANDB=0 python -m tasks.h2s2r_pour17.train \
  --bundle_root /path/to/pour17 \
  --num_envs 512 --headless \
  --name H2S2R_Pour17_Estimated_seed42
```

Every run writes `run_manifest.json` with the input/checkpoint-relevant hashes,
PPO config, adaptations, git state, and explicit excluded priors. Training-time
milestones are labelled as H2S2R's internal metric; they are not paper success.

## Frozen evaluation

Do **not** use the bundle's bare `evaluator/run_eval.py` defaults for this
phase-one result: that generic entry still starts at canonical t0 and defines
success as G4. Both ours and H2S2R must initialize the progress machine with
G1/G2 preset and use G3 as the primary pour-only success gate. The adapter below
does this explicitly and records the choice in its result JSON.

```bash
python -m tasks.h2s2r_pour17.vector_eval \
  --bundle_root /path/to/pour17 \
  --checkpoint /path/to/checkpoint.pth \
  --episodes 512 --batch_size 512 --headless \
  --out /path/to/eval_result.json
```

The evaluator suppresses Isaac Lab's automatic reset so it can inspect and
attribute the terminal physical state itself. It uses the bundle's
`PourProgressBatch`, enforces 903 steps independently of environment
`truncated`, counts every matching deadline, records its denominator, and keeps
`episode i -> seed_base+i` independent of vector batch partitioning.

For phase one, the primary success rate is G3 reach rate under
`--objective pour_only` (the default). Report environment steps to convergence,
wall-clock time, GPU count, and all-hit deadline distribution. G4 is retained as
a diagnostic only and must not be presented as the phase-one success metric.
Training curves are retained as supplementary diagnostics.

## Upstream provenance

- Human2Sim2Robot reference: commit
  `894eae2ec3ae39a573b81bd1860d14cc6bdfa6df` (MIT).
- FABRICS Human2Sim2Robot branch: commit
  `b56c9b1d7c927f5f8b7ca9464a3b5607a3777dae` (NVIDIA License).
- Both reference checkouts remain untouched outside this adaptation repository.
