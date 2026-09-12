# M1-A 输入与几何映射核验

状态日期：2026-09-12。本文件只记录 M1-A 输入解码、左手 landmark、候选 Sharpa FK/限位、局部标定、
合成求解、限定六例真实 MANO 单帧候选拟合、帧 0..15 I/T、固定目标递推归因、完整 142 帧 T 候选
几何诊断、限位/残差归因、tip–middle 点对距离相容性、固定 q 后验刚体配准、固定 A 构造依据审计及
既有候选骨架及登记 visual mesh 表面可视化交付的当前可审查事实。没有正式机器人 reference、世界回放、
训练或正式 M1-A 验收。

## 1. 当前状态

| 项目 | 状态 | 边界 |
|---|---|---|
| 候选 NumPy MANO decoder | **VERIFIED** | 仅指 v4 指定模型、左 shapedirs 候选预处理、两种显式 mean 配置和冻结案例下，与指定官方 MuJoCo `ManoLayer` 数值一致；不认证历史 HaWoR 输入语义。 |
| v4 输入/输出证据绑定 | **PASS** | 复用已有数值输出，60/60 项身份、配置、案例和接口检查通过；没有重跑远端 forward。 |
| 当前可见代码中的 pose 表示 | **当前源码支持 axis-angle** | 平滑、导出和消费路径把 global/local 三元组当作 rotvec；这些源码未与历史 pour/17 run 绑定，历史生产语义仍为 **UNKNOWN**。 |
| 历史 HaWoR 模型、mean、左 shapedirs、完整 root/translation 语义 | **UNKNOWN** | 缺少历史 source/config/environment/model/run manifest 绑定。 |
| wrist-origin/zero-global hand-local 工程候选 | **PROPOSED；坐标不变性 PASS** | reviewer 指定定义在固定七案例 × 两个既有 variant 下 14/14 通过整体旋转/平移不变性检查；只证明候选坐标代数，不恢复历史语义、不选择 mean、不批准输入合同。 |
| 左手 landmark | 九项 **PROPOSED**；指定 thumb 候选已完成结构核验，仍待批准 | `proposed_thumb_internal_segment_midpoint_v1` 的 selector/父子链、URDF 端点和有限扰动 FK 通过；不等于解剖映射 VERIFIED 或十点合同冻结。 |
| 候选左 Sharpa 22-joint FK/限位接口 | **PROPOSED；内部一致性 PASS** | q=0、q_mid、22/22 单关节独立轴/支点关系和拒绝路径通过；未与权威 USD/Isaac articulation 对拍，十点仅获准用于本轮接口测试。 |
| 隔离单帧合成求解器 | **PROPOSED；目标/Jacobian PASS；S0/S1/S2 PASS** | 专用隔离 venv 中实际调用 NLopt LD_SLSQP；仅证明三个冻结 Sharpa-generated synthetic cases，不是实际 retargeting。 |
| 真实 MANO 六例单帧候选拟合 | **执行检查 6/6 PASS；几何质量待 reviewer 评估** | 仅帧 0/71/141 × 两个既有 mean 分支，全部独立从限位中点求解；不是 mean 选择、正式十点/输入合同批准或完整 M1-A。 |
| 中点锚定影响 probe | **执行检查 6/6 PASS；局部 tracking 影响已量化** | 从各自原 `q_final` 做一次 `alpha=0` 的 tracking-only 诊断；正式候选 `alpha=0.004` 不变，不构成参数选择或几何质量 PASS。 |
| 帧 0..15 I/T 短序列候选诊断 | **执行检查 64/64 PASS；误差与连续性已量化** | 两分支各 16 帧；I 每帧锚定中点，T 以上一帧解初始化并锚定，`alpha=0.004`；共 62 次实际求解。不是正式时间策略、mean 选择或完整 M1-A。 |
| 固定目标递推归因对照 | **执行检查 30/30 PASS；无目标运动时仍继续调整** | 每分支固定复用 frame-0 目标并从共享 frame-0 解递推 15 步；`k` 不是 source frame/time。只支持归因边界，不批准 warmup 或时间策略。 |
| 完整 142 帧 T 候选几何诊断 | **284/284 分支/帧执行检查 PASS；几何质量待评估** | 复用 32 条 frame 0..15 前缀且不重跑，从 frame 16 新执行 252 次；不等于正式 reference、runtime 等价或 M1-A 验收通过。 |
| 完整序列限位/残差归因审计 | **只读诊断完成；固定帧局部一阶证据** | 逐关节/逐点全序列与 `16..141` 表已生成；固定 12 个分支/帧中共 58 个贴边关节分量均有 tracking 与 total 向外梯度。只支持局部边界限制解释，不证明不可达、唯一原因或正式 M1-A 通过。 |
| tip–middle 距离相容性审计 | **只读解析诊断完成；发现特定点对距离不相容** | 五指实际相对链均满足单 revolute 前提；raw 的 thumb/pinky、+mean 的 index/pinky 存在明确超界帧。只证明当前候选点、尺度与限位组合的点对必要条件冲突，不选择 mean 或改动任何合同。 |
| 固定 q 后验刚体配准诊断 | **只读反事实诊断完成；不是标定** | 共享刚体将 raw/+mean 全序列 tracking 分别降低 `47.57%/25.19%`；只说明当前固定 q 残差含可由共同刚体变换减少的成分，不能直接判定正式 A 错误或授权逐帧变换。 |
| 固定 A 构造依据审计 | **静态复核完成；未发现明确实现矛盾** | 当前 A 精确复现四 MCP 均值及构造基对齐；`left_hand_C_MC↔MANO J0` 缺少解剖/作者对应依据，仍是用于纵向轴的工程假设。 |
| 完整候选几何可视化 | **骨架版保留；精确 visual mesh 表面版交付完成；reviewer 已查看固定五帧图，完整动画未逐帧评审** | 28 个登记引用对应的 13 个去重 LFS 对象已按 oid 精确恢复到隔离目录；两分支完整 142 帧及固定五帧表面图均保留原十点/残差。该展示不是碰撞、物理或正式质量验收。 |
| 旧 uncentered-canonical 固定 A | **PROPOSED；旧坐标证据保留** | 旧脚本实际用未减 wrist 的 canonical MANO 点；旧 A 在该输入坐标下不判为错误，但不能直接作用于 wrist-zero 点。 |
| wrist-zero hand-local 固定 A | **PROPOSED；坐标换算 PASS** | 同一 canonical 下按解析关系重建，B/R、det、原点/基方向及新旧 21 点映射全部通过；不是解剖或 runtime 批准。 |
| 正式 M1-A retargeting 验收 | **NOT PASSED** | 已运行完整 142 帧候选几何诊断及限位/残差、点对距离、固定 q 后验配准、固定 A 构造四项只读审计，但正式输入/十点/时间策略、权威 runtime、reference、姿态/物理验收均未批准或完成。 |
| M0 authoritative runtime | **BLOCKED** | 交接包尚未绑定为当前或历史 Ours 权威运行版本。 |

## 2. 当前有效证据

- 候选配置：[`m1a_left_landmark_calibration_candidate_v1.json`](README.md#archive-access)（归档：`configs/m1a_left_landmark_calibration_candidate_v1.json`）
- 完整静态审计：[`m1a_full_mano_decode_audit.json`](README.md#archive-access)（归档：`artifacts/m1a_full_mano_decode_audit_v4/m1a_full_mano_decode_audit.json`）
- v4 数值对拍：[`m1a_mano_crosscheck_v4.json`](README.md#archive-access)（归档：`artifacts/m1a_mano_crosscheck_v4/m1a_mano_crosscheck_v4.json`）
- trusted runtime：[`trusted_layer_runtime_evidence.json`](README.md#archive-access)（归档：`artifacts/m1a_trusted_layer_validation_v4/trusted_layer_runtime_evidence.json`）
- v4 证据绑定：[`m1a_v4_evidence_binding.json`](README.md#archive-access)（归档：`artifacts/m1a_mano_crosscheck_v4/m1a_v4_evidence_binding.json`）
- hand-local 候选诊断：[`m1a_hand_local_candidate_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_hand_local_candidate_v1/m1a_hand_local_candidate_v1.json`）
- hand-local diagnostic samples：[`m1a_hand_local_diagnostic_samples_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_hand_local_candidate_v1/m1a_hand_local_diagnostic_samples_v1.npz`）
- hand-local 诊断脚本：[`m1a_hand_local_candidate_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_hand_local_candidate_v1.py`）
- wrist-zero 标定候选配置：[`m1a_wrist_zero_hand_local_calibration_candidate_v1.json`](README.md#archive-access)（归档：`configs/m1a_wrist_zero_hand_local_calibration_candidate_v1.json`）
- 公式范围修正版配置：[`m1a_wrist_zero_hand_local_calibration_candidate_v2.json`](configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json)
- wrist-zero 标定诊断：[`m1a_wrist_zero_hand_local_calibration_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_wrist_zero_hand_local_calibration_candidate_v1/m1a_wrist_zero_hand_local_calibration_audit_v1.json`）
- wrist-zero 标定脚本：[`m1a_wrist_zero_hand_local_calibration_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_wrist_zero_hand_local_calibration_audit_v1.py`）
- thumb 候选配置：[`m1a_thumb_internal_segment_midpoint_candidate_v1.json`](README.md#archive-access)（归档：`configs/m1a_thumb_internal_segment_midpoint_candidate_v1.json`）
- thumb 结构诊断：[`m1a_thumb_internal_segment_midpoint_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_thumb_internal_segment_midpoint_candidate_v1/m1a_thumb_internal_segment_midpoint_audit_v1.json`）
- thumb 结构脚本：[`m1a_thumb_internal_segment_midpoint_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_thumb_internal_segment_midpoint_audit_v1.py`）
- thumb 静态图：[`m1a_thumb_internal_segment_midpoint_candidate_v1.png`](README.md#archive-access)（归档：`artifacts/m1a_thumb_internal_segment_midpoint_candidate_v1/m1a_thumb_internal_segment_midpoint_candidate_v1.png`）
- Sharpa FK 候选配置：[`m1a_sharpa_left_fk_candidate_v1.json`](configs/m1a_sharpa_left_fk_candidate_v1.json)
- Sharpa FK 接口：[`m1a_sharpa_left_fk_candidate_v1.py`](diagnostics/m1a_sharpa_left_fk_candidate_v1.py)
- Sharpa FK 诊断脚本：[`m1a_sharpa_left_fk_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_sharpa_left_fk_audit_v1.py`）
- Sharpa FK 诊断：[`m1a_sharpa_left_fk_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_left_fk_candidate_v1/m1a_sharpa_left_fk_audit_v1.json`）
- 旧依赖阻塞配置：[`m1a_sharpa_single_frame_solver_synthetic_candidate_v1.json`](README.md#archive-access)（归档：`configs/m1a_sharpa_single_frame_solver_synthetic_candidate_v1.json`）
- 单帧求解器：[`m1a_sharpa_single_frame_solver_candidate_v1.py`](diagnostics/m1a_sharpa_single_frame_solver_candidate_v1.py)
- 合成诊断脚本：[`m1a_sharpa_single_frame_solver_synthetic_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_sharpa_single_frame_solver_synthetic_audit_v1.py`）
- 旧依赖阻塞报告：[`m1a_sharpa_single_frame_solver_synthetic_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_single_frame_solver_synthetic_candidate_v1/m1a_sharpa_single_frame_solver_synthetic_audit_v1.json`）
- 当前 NLopt 执行配置：[`m1a_sharpa_single_frame_solver_synthetic_candidate_v2.json`](README.md#archive-access)（归档：`configs/m1a_sharpa_single_frame_solver_synthetic_candidate_v2.json`）
- 专用环境证据配置：[`m1a_nlopt_environment_v1.json`](README.md#archive-access)（归档：`configs/m1a_nlopt_environment_v1.json`）
- 哈希锁文件：[`m1a_nlopt_v1_requirements.lock.txt`](README.md#archive-access)（归档：`configs/m1a_nlopt_v1_requirements.lock.txt`）
- 当前三例原始报告：[`m1a_sharpa_single_frame_solver_synthetic_audit_v2.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_single_frame_solver_synthetic_candidate_v2/m1a_sharpa_single_frame_solver_synthetic_audit_v2.json`）
- 当前环境/执行绑定：[`m1a_nlopt_environment_and_execution_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_single_frame_solver_synthetic_candidate_v2/m1a_nlopt_environment_and_execution_audit_v1.json`）
- 真实六例候选配置：[`m1a_real_mano_single_frame_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`configs/m1a_real_mano_single_frame_candidate_diagnostic_v1.json`）
- 真实六例几何准备报告：[`real_mano_single_frame_candidate_geometry_preparation_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_geometry_preparation_v1.json`）
- 真实六例求解报告：[`real_mano_single_frame_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_v1.json`）
- 真实六例诊断样本：[`real_mano_single_frame_candidate_diagnostic_samples_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_samples_v1.npz`）
- 真实六例静态图：[`real_mano_single_frame_candidate_diagnostic_v1.png`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_v1.png`）
- 中点锚定 probe 配置：[`m1a_midpoint_anchor_effect_probe_v1.json`](README.md#archive-access)（归档：`configs/m1a_midpoint_anchor_effect_probe_v1.json`）
- 中点锚定 probe 脚本：[`m1a_midpoint_anchor_effect_probe_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_midpoint_anchor_effect_probe_v1.py`）
- 中点锚定 probe 报告：[`m1a_midpoint_anchor_effect_probe_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_midpoint_anchor_effect_probe_v1/m1a_midpoint_anchor_effect_probe_v1.json`）
- 短序列 I/T 配置：[`m1a_short_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`configs/m1a_short_sequence_candidate_diagnostic_v1.json`）
- 短序列几何准备证据：[`short_sequence_candidate_geometry_preparation_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_geometry_preparation_v1.json`）
- 短序列 I/T 报告：[`short_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_v1.json`）
- 短序列诊断数组：[`short_sequence_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_arrays_v1.npz`）
- 短序列曲线：[raw](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_raw_pose_v1.png`）、[+mean](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）
- 固定目标递推配置：[`m1a_fixed_target_recursion_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`configs/m1a_fixed_target_recursion_candidate_diagnostic_v1.json`）
- 固定目标递推报告：[`fixed_target_recursion_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_v1.json`）
- 固定目标递推数组：[`fixed_target_recursion_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_arrays_v1.npz`）
- 固定目标 F/T 曲线：[raw](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_raw_pose_v1.png`）、[+mean](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）
- 完整序列配置：[`m1a_full_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`configs/m1a_full_sequence_candidate_diagnostic_v1.json`）
- 完整序列几何准备证据：[`full_sequence_candidate_geometry_preparation_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_geometry_preparation_v1.json`）
- 完整序列诊断报告：[`full_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_v1.json`）
- 完整序列诊断数组：[`full_sequence_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_arrays_v1.npz`）
- 完整序列曲线：[raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_curves_v1.png`）、[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_curves_v1.png`）
- 固定四帧对照：[raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_static_v1.png`）、[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_static_v1.png`）
- 限位/残差审计配置：[`m1a_limit_residual_audit_v1.json`](README.md#archive-access)（归档：`configs/m1a_limit_residual_audit_v1.json`）
- 限位/残差审计脚本：[`m1a_limit_residual_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_limit_residual_audit_v1.py`）
- 限位/残差紧凑表：[`M1A_LIMIT_RESIDUAL_AUDIT_V1.md`](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/M1A_LIMIT_RESIDUAL_AUDIT_V1.md`）
- 限位/残差完整报告：[`m1a_limit_residual_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_v1.json`）
- 限位/残差数组：[`m1a_limit_residual_audit_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_arrays_v1.npz`）
- 限位/残差哈希清单：[`m1a_limit_residual_audit_hashes_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_hashes_v1.json`）
- tip–middle 距离审计配置：[`m1a_tip_middle_distance_compatibility_audit_v1.json`](README.md#archive-access)（归档：`configs/m1a_tip_middle_distance_compatibility_audit_v1.json`）
- tip–middle 距离审计脚本：[`m1a_tip_middle_distance_compatibility_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_tip_middle_distance_compatibility_audit_v1.py`）
- tip–middle 距离紧凑表：[`M1A_TIP_MIDDLE_DISTANCE_COMPATIBILITY_AUDIT_V1.md`](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/M1A_TIP_MIDDLE_DISTANCE_COMPATIBILITY_AUDIT_V1.md`）
- tip–middle 距离完整报告：[`m1a_tip_middle_distance_compatibility_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_v1.json`）
- tip–middle 距离数组：[`m1a_tip_middle_distance_compatibility_audit_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_arrays_v1.npz`）
- tip–middle 距离哈希清单：[`m1a_tip_middle_distance_compatibility_audit_hashes_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_hashes_v1.json`）
- 固定 q 后验配准配置：[`m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`](README.md#archive-access)（归档：`configs/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`）
- 固定 q 后验配准脚本：[`m1a_fixed_q_posthoc_rigid_alignment_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.py`）
- 固定 q 后验配准紧凑表：[`M1A_FIXED_Q_POSTHOC_RIGID_ALIGNMENT_AUDIT_V1.md`](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/M1A_FIXED_Q_POSTHOC_RIGID_ALIGNMENT_AUDIT_V1.md`）
- 固定 q 后验配准完整报告：[`m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`）
- 固定 q 后验配准数组/图/哈希：[`NPZ`](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_arrays_v1.npz`）、[`PNG`](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.png`）、[`hashes`](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_hashes_v1.json`）
- 固定 A 构造审计配置/脚本：[`config`](README.md#archive-access)（归档：`configs/m1a_fixed_A_construction_basis_audit_v1.json`）、[`script`](README.md#archive-access)（归档：`diagnostics/m1a_fixed_A_construction_basis_audit_v1.py`）
- 固定 A 构造审计报告/机读记录/图/哈希：[`table`](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/M1A_FIXED_A_CONSTRUCTION_BASIS_AUDIT_V1.md`）、[`JSON`](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.json`）、[`PNG`](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.png`）、[`hashes`](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_hashes_v1.json`）
- 当前静态图：[`m1a_full_mano_real_frames_candidates.png`](README.md#archive-access)（归档：`artifacts/m1a_full_mano_decode_audit_v4/m1a_full_mano_real_frames_candidates.png`）、
  [`m1a_proposed_local_calibration.png`](README.md#archive-access)（归档：`artifacts/m1a_full_mano_decode_audit_v4/m1a_proposed_local_calibration.png`）

旧 v0/v2/v3 artifacts 为 provenance，保留但不作为当前结论来源。v0 的自写 FK 已按原始 MANO
kinematic-tree 索引构造 transforms，却又套用了只适用于 breadth-level FK 输出的 16-joint reorder，随后再套
21-point selector，造成双重重排。受该错误影响的旧 21 点、thumb middle、palm anchors、`R,t`、尺寸统计和
静态图全部失效；不能从旧图或旧 JSON 恢复任何几何数值。

### 2.1 v4 decoder 对拍

候选模型 `MANO_LEFT.pkl` SHA-256 为
`c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30`；指定官方 MuJoCo
`common/mano/manolayer.py` SHA-256 为
`e9a38f5df9b54ec82e631dbab678ebb9e7276210a76ee88de6df771bf6d77927`；冻结输入 SHA-256 为
`91b10f1a9e32135b8e9b6b4d670c000af4df9787a7ca443ef4fbf602e8df41a4`。

两种显式配置分别对拍：

1. NumPy 不加 `hands_mean` ↔ 官方 `flat_hand_mean=True`；
2. NumPy 显式加 `hands_mean` ↔ 官方 `flat_hand_mean=False`。

两种配置均运行中立、index MCP 单关节旋转、非零 betas、非零 global rotation/translation 和固定实帧
0/71/141。778 vertices、从官方 joints21 恢复的 raw16 diagnostic view、final21 共 42 项全部通过预登记
阈值；全局最大 max-abs 为 `7.7393e-8 m`，最大 RMS 为 `2.3300e-8 m`。该结论只验证实现一致性：
**两种分支都通过不能用于选择历史 mean 分支。**

### 2.2 v4 证据绑定

绑定脚本 [`m1a_v4_evidence_binding_audit.py`](README.md#archive-access)（归档：`diagnostics/m1a_v4_evidence_binding_audit.py`） SHA-256 为
`3570d98c2e3ddf5e1cc8b427352035df398baae2df4e18548bf65bb0e5dc3b04`；绑定记录 SHA-256 为
`a1cae5301196f041ab46269cce71c15aae8c8d623b6c7a8e6dee8e59d25fc0a0`。原 v4 对拍报告保持不变，
SHA-256 为 `14aa7a7577f842ccf1592dde4b8619d5dc35e30726a4338d72aa3c5eb2fe22b9`。

60/60 项 PASS：input、custom output、trusted output 与各自 metadata 哈希一致；prepare metadata 的
input/config 哈希一致；案例顺序、variant、输入/输出 shape、有限值、trusted constructor/forward 参数、mean
选项和 raw16 恢复索引均匹配冻结配置。该 PASS 只加强 v4 证据身份，不扩大其语义结论。

### 2.3 wrist-origin/zero-global hand-local 工程候选

新候选 `proposed_wrist_zero_global_hand_local_v1` 标记为 **PROPOSED**。对每个样本及每个既有 variant：

`P0=D(beta,pose45,global_aa=0,trans=0;variant)`，`o0=P0.joints21[0]`；
`vertices_H=P0.vertices-o0`，`joints21_H=P0.joints21-o0`。

H 原点是同一样本、同一 variant 在候选模型中解码得到的 wrist joint，轴沿用 global rotation 为零时的
模型轴。`betas`、`pose45`、既有 variant mean 行为和米制手尺寸保持不变；不额外减 `hands_mean`，不做
尺寸归一化，不减掌心均值，不使用机器人/物体坐标、`hand_root_offset_mano` 或 Sharpa `A`。

执行前只读检查确认候选 decoder 把 `global_rot` 施加到 raw root joint 0，子关节继承根变换，并在 LBS
后对 vertices 与 joints 统一加 `trans`。因此列向量恢复式为
`R^T(Pglobal-oglobal)=PH`；诊断数组以行向量存点，代码等价实现为 `(Pglobal-oglobal)@R`。

固定范围为原 v4 的四个合成样本和真实帧 0/71/141 的 `betas/pose45`，两个既有 variant 都执行；统一
使用合成 `global_aa=[0.2,-0.1,0.15] rad` 与 `trans=[0.02,-0.03,0.04] m`。预登记阈值为
max-absolute `≤2e-6 m`、RMS `≤5e-7 m`。结果 **PASS：14/14**；shape、finite 和精确 wrist-zero
均通过，vertices/joints21 总体最大 max-abs 为 `8.326672684688674e-17 m`，最大 RMS 为
`1.3524489597802808e-17 m`。

报告 SHA-256 为 `2fab28bd99782763d28826281fd774ee233ea2ccd89b04ac9a39eb06575253b8`；diagnostic samples
SHA-256 为 `77d4bc3513b6868d9493ea9ca87695ceae7862d0b99e5d5b1e5be5545722ff7b`；脚本 SHA-256 为
`c64e3dd8863faf23d9803da56721b0c96fcb38cd76c7c8dd0ba6ca38fc335c94`。样本 NPZ 只含七案例输入和
两分支的直接/恢复 hand-local vertices/joints21，明确不是完整序列或机器人 reference。未重跑官方 layer，
本轮 14 项不计入原 v4 42 项。

该 PASS 只说明此候选 hand-local 定义在上述测试范围内具有要求的整体变换不变性；它不是历史 HaWoR/
论文作者坐标复原，不选择 mean，不批准正式输入合同。历史模型、mean、shapedirs、root_offset 等仍为
**UNKNOWN**；该 decoder 对拍本身不构成 M1-A 验收，当前完整序列候选状态见 3.8，M0 仍为 **BLOCKED**。

## 3. MANO 21 点与左手 landmark 候选

修复后的官方 21 点顺序和 tip selector 已由指定官方 source 独立核验。每条手指的三个 raw joints 来自同一
kinematic parent chain，并连接对应官方 tip vertex；五条结构检查均 PASS。

| final21 | raw joint / tip vertex | 可视化角色 |
|---:|---|---|
| 0 | raw 0 | wrist |
| 1–4 | raw 13,14,15；tip 745 | thumb mcp, pip, dip, tip |
| 5–8 | raw 1,2,3；tip 317 | index mcp, pip, dip, tip |
| 9–12 | raw 4,5,6；tip 445 | middle mcp, pip, dip, tip |
| 13–16 | raw 10,11,12；tip 556 | ring mcp, pip, dip, tip |
| 17–20 | raw 7,8,9；tip 673 | little mcp, pip, dip, tip |

当前九项 **PROPOSED** landmark 如下；Sharpa 点是候选 URDF/config 中的 link-local 几何点，均不是 COM。

| 点 | MANO 候选 | Sharpa 候选 |
|---|---|---|
| thumb tip | `J4` / vertex 745 | `left_thumb_fingertip:[0,0,0]` |
| index tip | `J8` / vertex 317 | `left_index_fingertip:[0,0,0]` |
| middle tip | `J12` / vertex 445 | `left_middle_fingertip:[0,0,0]` |
| ring tip | `J16` / vertex 556 | `left_ring_fingertip:[0,0,0]` |
| little tip | `J20` / vertex 673 | `left_pinky_fingertip:[0,0,0]` |
| index middle | `(J6+J7)/2` | `left_index_MP:[0.01575,0,0]` |
| middle middle | `(J10+J11)/2` | `left_middle_MP:[0.01575,0,0]` |
| ring middle | `(J14+J15)/2` | `left_ring_MP:[0.01575,0,0]` |
| little middle | `(J18+J19)/2` | `left_pinky_MP:[0.01575,0,0]` |

### 3.1 指定 thumb internal-segment midpoint 候选

reviewer 唯一指定 `proposed_thumb_internal_segment_midpoint_v1`：MANO `(J2+J3)/2`，Sharpa 为
`left_thumb_PP` 进入关节位置与通向 IP 的关节端点之中点。本轮没有评估其它 midpoint，也没有用图像观感
选择候选。

**VERIFIED 模型结构事实：**公开 selector 为 `J1→raw13`、`J2→raw14`、`J3→raw15`、
`J4→tip vertex 745`；MANO kinematic tree 为 `raw13(parent=0)→raw14(parent=13)→raw15(parent=14)`，
故 J2/raw14 与 J3/raw15 是直接父子 joint origins。候选是这两个**运动学关节原点**的线段中点，不能等同于
网格骨段或解剖中心；public `mcp/pip/dip` 可视化 labels 不提供 CMC/MCP/IP 解剖认证。

候选 URDF 中，进入 `left_thumb_PP` 的 `left_thumb_MCP_AA` 为
`left_thumb_MCP_VL→left_thumb_PP`，origin `[0,0,0]`、rpy `[1.5708,0,0]`、axis `[0,0,1]`；通向
IP 的 `left_thumb_IP` 为 `left_thumb_PP→left_thumb_DP`，origin `[0.039,0,0]`、rpy
`[-1.5708,0,0]`、axis `[0,0,1]`。在同一 `left_thumb_PP` 坐标中，两端为 `[0,0,0]` 与
`[0.039,0,0] m`，中点精确为 `[0.0195,0,0] m`。该点不同于 link origin `[0,0,0]`，也不同于
inertial COM `[0.021256,-0.00095,0.00092] m`；没有使用 COM 替代。

**结构/FK 检查：PASS。**MANO raw13/14/15 各自 x/y/z `+0.1 rad` 共九项扰动全部 finite；中点均值
残差为 `0`，J2–J3 刚性段最大长度变化 `3.469446951953614e-18 m`，预期不动的上游 joint origins
最大变化 `0`。Sharpa `left_thumb_MCP_AA` 与 `left_thumb_IP` 的 `+0.1 rad` 均在候选 URDF 限位内；
两项均 finite，中点均值和段长变化为 `0`，变换回 PP frame 后端点/中点最大误差
`3.469446951953614e-17 m`。全部低于 `1e-8 m` 内部容差。逐扰动位置见[诊断报告](README.md#archive-access)（归档：`artifacts/m1a_thumb_internal_segment_midpoint_candidate_v1/m1a_thumb_internal_segment_midpoint_audit_v1.json`）；图只展示各自局部结构，未用 A、物体或世界坐标。

候选配置 SHA-256 为 `537e2ffe25a34ec8727d99a91f3bd273062d6c437c6854da49ea2b6f45cde3c5`；
诊断报告为 `1a80970b0b57a25252507df95902f25d3b5aa8e902aee44b5bd18f0be9839ea4`；脚本为
`fb3b6edd050ed49529bbaf1e325c6786a9d3197aed9ea7a5ffe193eeeec0c0a2`；静态图为
`1869519d232dcd18e6eaa98f26cd17c9b4dfb305c3d4c8f47e7a7956ac23a478`。MANO 扰动直接调用已验证
decoder，没有编写第二个 MANO FK。

没有发现足以否定该指定候选的具体结构矛盾。当前状态只能写作：**“指定候选已完成结构核验，仍待批准”。**
仍不能证明 J2/J3 的解剖命名、MANO 与 Sharpa 的解剖对应、论文作者原始 middle-phalanx 索引/映射或
十点合同已批准；thumb landmark 未正式冻结。

### 3.2 候选左 Sharpa 22-joint FK/限位接口

候选 `proposed_sharpa_left_22joint_fk_v1` 严格绑定已登记归档
`pour17_baseline_bundle_20260829.tar.gz`（SHA-256 `77edc03c…10f6`）中的
`pour17/world/world_manifest.json`（`2339f7a9…dcd5`）和候选 URDF
`vega_1p_sharpa_fix.urdf`（`968b41f8…76db`）。同名 `_inspection` manifest 的现场哈希是
`44519e35…ec3`，与登记成员不同，故没有用作本轮输入。实际 manifest 字段为
`robot.controlled_joint_names_in_order[36:58]` 和同索引的 `robot.controlled_joint_limits_rad`；22 名称
唯一且与 `BASELINE_PLAN.md` 顺序完全一致。

**结构/限位事实：PASS。**22 个名称全部存在且可由 `left_hand_C_MC` 到达，均为 `revolute`，无
`mimic`/coupling；每项 type、axis、origin、父子 link 与 URDF/manifest lower/upper 对照均保存在诊断
JSON。所有 lower/upper finite 且 lower<upper；manifest 软限位相对 URDF 的最大逐端差为
`1.609802247060088e-7 rad`，低于既定 `1e-6 rad` 比较容差，没有取交集、替换来源或修正输入。

接口输入为严格 manifest 顺序的 finite `q_rad (22,)`，超出软限位时明确报错且不 clip。输出为
`left_hand_C_MC` 根下 33 个 link transforms（root 加 32 个可达 movable/fixed 子 link）和固定顺序
`(10,3)` 点：`thumb_tip, thumb_middle, index_tip, index_middle, middle_tip, middle_middle, ring_tip,
ring_middle, pinky_tip, pinky_middle`。tip 为各 fingertip frame origin；thumb middle 是
`left_thumb_PP:[0.0195,0,0] m`，四个长指 middle 是各 MP frame `[0.01575,0,0] m`。该十点集合仅获准
用于本轮 FK 接口测试，未升级为正式方法合同，也不涉及 MANO 对应或 A。

**固定验证：PASS。**q=0 在候选限位内，新接口与既有 q=0-only zero-frame 的 33 个 transform
max-absolute 差为 `0`。q_mid 的 `(22,)` 输入、`(33,4,4)` transforms 和 `(10,3)` points 均 shape
正确且 finite。22/22 单关节扰动全部通过；每次从 q_mid 增加该关节 manifest 限位跨度的 10%，预期值
由基准姿态世界轴/支点与诊断侧独立 Rodrigues 公式计算，而非再次调用 FK。非后代 link 最大变化 `0`；
后代位置最大误差 `2.7755575615628914e-17 m`、旋转最大误差 `3.3306690738754696e-16`；link-local
landmark 投影误差 `0`、运动关系最大误差 `2.7755575615628914e-17 m`。全部扰动的最大正交残差
`1.1102230246251565e-15`、最大 `|det-1|=9.992007221626409e-16`，低于预登记 `1e-10` 旋转容差；
位置关系低于 `1e-8 m`。错误 shape、NaN、明确超限均被拒绝。首个失败项为 `null`。

配置 SHA-256 为 `c6ff31d3961fe28714d4556f5fc503f29d745805c6666dcd11af386d0800312e`，接口为
`1824dd4eaeef280c6d19008e6707f25935889faa9f219960778617ebcbc82c64`，诊断脚本为
`6618a661638b0b30318620313f2cc5849685dcf2f2b6b82b9be11eaaade6763d`，诊断 JSON 为
`fa1cfba3fb3ea5f307791c0a60b3648ec918f5e5790dc981db28e4e44f676aff`；JSON 内含完整 22 项对照、测试
明细和可重复执行命令。

该 PASS 只说明候选 manifest/URDF 组合下的 FK 实现满足本轮内部结构、限位和运动代数检查。候选
URDF/manifest 是否等价于权威 Sharpa runtime 仍 **NOT VERIFIED**；十点正式合同仍待批准；该 FK 核验轮
没有读取真实 MANO 序列、选择 mean、运行优化器、生成 robot q/reference、回放或训练；该 FK 核验本身
不构成 M1-A 验收。当前完整序列候选状态见 3.8，M0 保持 **BLOCKED**。

### 3.3 隔离单帧合成求解器候选

候选目标严格为
`sum_j ||FK_j(q)-y_j||² + 4e-3 ||q-q_prev||²`，对固定顺序十点和 xyz 求和、不求平均；位置米、
关节 rad，约束使用已登记 manifest lower/upper。求解器文件只调用已验证候选 FK，不修改 FK，也不读取
MANO、A 或 perception。NLopt 合同固定为 `LD_SLSQP`、`maxeval=500`、`ftol_abs=1e-12`、
`xtol_abs=1e-8 rad`；不含其它损失。求解函数只接收 y、q_prev、q_init 和冻结合同，不接收测试驱动的
q_star。

manifest 只读比较结论：归档成员 `2339f7a9…dcd5` 与 `_inspection` 副本 `44519e35…ec3` 的左手 22
名称、顺序、限位完全一致；其余差异 JSON path 仅为 `robot.self_collision_note`、
`robot.self_collision_source`。本轮仍以归档成员为唯一来源，未合并或修改文件。既有 FK v1 诊断实际
绑定归档成员；q=0 zero-frame、wrist-zero A 与 thumb 结构诊断只绑定候选 URDF，未使用 manifest
顺序/限位，故不因这两个无关字段差异而失效。

**解析梯度检查：PASS。**在 m 对 S1 目标以固定 `h=1e-6 rad` 做中心差分，全部正负扰动在限位内。
landmark Jacobian 最大误差为 `2.285326788920017e-11 m/rad`（阈值 `1e-7`）；完整目标梯度最大误差为
`1.1095837790242587e-12`（阈值 `1e-7`）。解析列使用运动前 root-frame axis/pivot 的
`axis×(point-pivot)`；数值差分未作为求解梯度。

三个固定测试均已构造：S0 `q_star=m`，S1 `q_star=m+0.1d`，S2
`q_star_i=m_i+0.1d_i(-1)^i`；三者 q_prev=q_star、q_init=m、y=FK(q_star)。q_star oracle total 均为
`0`，q_init 的初始 total 为 `0`、`0.006143438831401291`、`0.005113563450043941`。

专用环境位于 `D:/UCBP/RL/Baselines/ViViDex/.venvs/m1a_nlopt_v1`，由已知 UCBP Python 3.10.19
64-bit 创建，`include-system-site-packages=false`。仅安装官方 PyPI binary wheels：
`numpy-2.2.6-cp310-cp310-win_amd64.whl`（SHA-256 `f0fd6321…03f3`）和
`nlopt-2.10.0-cp310-cp310-win_amd64.whl`（`49aa0439…8e1`）；哈希锁文件为 `09f05ea3…3b9f`。
实际导入均来自该 venv 的 `lib/site-packages`，版本分别为 NumPy 2.2.6、NLopt 2.10.0；`pip check`
返回 `No broken requirements found.`。没有修改 UCBP/base/远端/training 环境，没有源码编译、换版本或
SciPy 替代。

环境变化后使用相同配置复跑梯度检查，结果与旧数值一致：landmark Jacobian 最大误差
`2.285326788920017e-11 m/rad`，目标梯度最大误差 `1.1095837790242587e-12`，继续 **PASS**。

**NLopt 三例实际求解：PASS。**三例均由 `LD_SLSQP` 返回 code `3 / FTOL_REACHED`，无异常、负码或
maxeval/maxtime 耗尽；S0/S1/S2 objective call count 分别为 `1/47/50`，证明 S0 也实际进入 NLopt。

| case | initial tracking / smoothness / total | final tracking / smoothness / total | landmark max-abs | limit violation | reported↔recomputed | 质量 |
|---|---|---|---:|---:|---:|---|
| S0 | `0 / 0 / 0` | `0 / 0 / 0` | `0 m` | `0 rad` | `0` | PASS |
| S1 | `0.004292251496770735 / 0.0018511873346305557 / 0.006143438831401291` | `6.53406504563831e-12 / 4.59955553854043e-12 / 1.11336205841787e-11` | `1.62973684625367e-6 m` | `0 rad` | `0` | PASS |
| S2 | `0.0032623761154133854 / 0.0018511873346305561 / 0.005113563450043941` | `5.65508233141535e-12 / 4.65730341254054e-12 / 1.03123857439559e-11` | `9.07009166245532e-7 m` | `0 rad` | `0` | PASS |

全部 final q/landmarks finite；未 clipping；最终总目标、landmark max-abs 和限位均通过原阈值。NLopt
返回码、质量条件及整体状态在报告中分开记录。当前 v2 配置 SHA-256 为 `a7d118a3…2c20`，原始三例
报告为 `02dd747e…001f`，环境/执行绑定脚本为 `48a90b57…91cd`，绑定报告为 `918bcd09…4aae`。
旧依赖阻塞报告 `96204f79…0951` 保留为历史证据，不再作为当前状态来源。

结论只限于 q_prev=q_star、正则项指向已知解的三个冻结 synthetic cases；不证明十点唯一决定 22 关节、
真实 MANO retargeting 质量、实际时间平滑、mean 选择或 M0。当前分层状态是：候选 FK 内部一致性
**PASS**；目标函数与解析梯度 **PASS**；三例 NLopt 合成求解 **PASS**。该合成阶段未执行真实
retargeting；当前完整序列候选状态见 3.8。十点正式适配合同未批准，权威 runtime 未验证，M0 仍 **BLOCKED**。

### 3.4 真实 MANO 六例单帧候选集成诊断

输入严格限定为登记 perception NPZ 的帧 0/71/141，以及 `proposed_raw_pose`、
`proposed_plus_hands_mean` 两个既有候选分支。三帧 `left_mano_valid=true`，pose45/betas finite；只读取
`frame_ids/left_mano_pose45/left_mano_betas/left_mano_valid`，没有读取 world、root offset、object、
Ours q/reference/confidence 或 GraspPose。每例按同次 J0 做 wrist-zero，应用同一 calibration v2 固定 A，
再按获准诊断用的十点固定顺序构造目标。两个分支没有额外加减 mean，也没有逐帧/逐 beta/逐分支重标定。

**输入复核：PASS。**六例相对既有 hand-local diagnostic samples 的 pose45、betas、vertices_H、
joints21_H max-absolute 差全部为 `0 m`（阈值 `1e-10 m`）。纯数值准备 artifact SHA-256 为
`d20ede12…cb75`，准备报告为 `6c6aacc2…0a7a`；它们只把已验证 decoder 的六例输出传给有意仅含
NumPy/NLopt 的专用求解环境，不构成第二个 decoder。

**求解执行检查：6/6 PASS。**每例独立使用 `q_init=q_prev=m`，不跨帧或跨分支传递初值；实际调用
NLopt `LD_SLSQP` 后均返回 code `3 / FTOL_REACHED`。无异常、负码、maxeval/maxtime 耗尽、clipping
或超过 `1e-6 rad` 比较容差的限位违例；全部 q/landmarks finite。独立重算 total 与 NLopt reported
value 的绝对差均为 `0`，final total 均未高于 initial total。返回码、执行质量与几何评估在报告中分字段保存。

| frame / variant | init→final total | init→final all-10 RMS 欧氏距离 (mm) | final tip / middle RMS (mm) | final 最大点误差 | calls |
|---|---:|---:|---:|---:|---:|
| 0 / raw | `0.0439714→0.0177137` | `66.311→29.373` | `33.216 / 24.945` | pinky_tip `43.576 mm` | 57 |
| 71 / raw | `0.0328537→0.0141696` | `57.318→27.542` | `29.724 / 25.171` | pinky_tip `38.487 mm` | 50 |
| 141 / raw | `0.0442279→0.0178378` | `66.504→29.281` | `32.775 / 25.310` | pinky_tip `41.137 mm` | 58 |
| 0 / +mean | `0.0140272→0.00764455` | `37.453→23.069` | `19.543 / 26.123` | thumb_middle `36.432 mm` | 60 |
| 71 / +mean | `0.0168299→0.00915213` | `41.024→24.555` | `22.549 / 26.410` | thumb_middle `34.297 mm` | 54 |
| 141 / +mean | `0.0159309→0.00856648` | `39.914→24.014` | `21.341 / 26.418` | thumb_middle `35.424 mm` | 55 |

表中的 RMS 是“十点各自欧氏距离”的 RMS，不是全部 xyz 分量 RMS。完整逐点毫米误差、initial/final
tracking/smoothness/total、最终 q、逐关节距限位距离和复现入口见当前求解报告；q 仅存在于命名为
`real_mano_single_frame_candidate_diagnostic` 的六例样本中，不是正式 reference 或轨迹。静态图使用
共同视角/坐标范围展示目标、初始/最终点、FK 链和残差线，未叠加物体或世界坐标。

当前配置 SHA-256 为 `4a140ca8…6a4`；几何准备脚本为 `1e865de7…f1f`，求解诊断脚本为
`8a9ec219…b96`，绘图脚本为 `79be4403…dffb`；求解报告为 `82e6def2…be3`，诊断样本为
`1aac57b8…3192`，静态图为 `b573893b…c3c5`。求解报告另完整绑定 perception、MANO 模型、固定 A、
十点定义、FK、solver、URDF、归档 manifest、配置和代码哈希，并保存专用解释器与 NumPy/NLopt 版本。

所有几何误差均如实记录为 **PENDING REVIEWER ASSESSMENT**，没有套用合成 `objective≤1e-8` 或
`landmark≤1e-4 m` 阈值，也不按较小误差选择或推荐 mean 分支。该结果不证明十点唯一解、时间平滑、
碰撞/接触、真实世界对齐、权威 runtime 等价或完整 142 帧质量。正式 mean 与十点合同仍未批准；该六例
本身不构成 M1-A 验收。当前完整序列候选状态见 3.8，M0 仍 **BLOCKED**。

### 3.5 中点锚定影响 probe

本轮唯一干预是在独立诊断中把原目标
`tracking(q)+0.004||q-m||²` 临时改为 `tracking(q)`，且每例从同例上一轮 `q_final` 启动。目标 y、
十点顺序、固定 A、mean 分支、MANO 模型、FK、manifest 顺序/限位和 NLopt 参数均保持不变。probe
直接读取已保存六例数值；prior config/report/samples 哈希和点序均通过检查，没有重新解码 MANO、重算 A、
修改既有 solver/FK、扫描参数、restart 或跨案例传递结果。正式候选 `alpha=0.004` 未改变。

在原解处记录 `g_tracking`、`g_anchor=2*0.004(q-m)`、`g_total` 及各自 L2/max-absolute 范数和逐关节
分量。projected gradient 明确采用：距下界 `≤1e-6 rad` 时可消去正梯度分量；距上界 `≤1e-6 rad`
时可消去负梯度分量；其余保留。六例原解的梯度范数如下：

| frame / variant | `||g_tracking||₂` | `||g_anchor||₂` | `||g_total||₂` | projected `||g_total||₂` | 原解 active bound |
|---|---:|---:|---:|---:|---|
| 0 / raw | `1.20572e-2` | `1.20572e-2` | `2.45898e-7` | `2.45898e-7` | none |
| 71 / raw | `1.02639e-2` | `1.02639e-2` | `1.11244e-7` | `1.11244e-7` | none |
| 141 / raw | `1.21976e-2` | `1.21746e-2` | `2.40131e-4` | `1.95720e-7` | `left_pinky_CMC` lower |
| 0 / +mean | `6.09629e-3` | `6.09629e-3` | `2.57139e-7` | `2.57139e-7` | none |
| 71 / +mean | `7.06824e-3` | `7.06824e-3` | `2.49455e-7` | `2.49455e-7` | none |
| 141 / +mean | `6.69287e-3` | `6.69287e-3` | `5.19730e-7` | `5.19730e-7` | none |

这些值显示原目标下 tracking 梯度与中点 anchor 梯度的数值平衡；没有为旧结果新增梯度阈值或重新定级。
原 q 到每个上下限的距离在报告中逐关节保留，不能仅凭未投影梯度判断边界约束收敛。

六次 probe 均返回 code `3 / FTOL_REACHED`，calls 为 `139/147/148/161/160/162`；finite、无 clipping、
无负码或预算耗尽，限位违例为 `0 rad`，独立重算 tracking 与 NLopt reported 值差均为 `0`，并满足
probe tracking 不高于同定义初始 tracking 超过 `1e-10`。逐例比较：

| frame / variant | tracking 原→probe / 降幅 | all-10 RMS 原→probe (mm) | tip RMS 原→probe (mm) | middle RMS 原→probe (mm) | max error 原→probe (mm) |
|---|---:|---:|---:|---:|---|
| 0 / raw | `0.00862765→0.00380806 / 55.86%` | `29.373→19.514` | `33.216→14.991` | `24.945→23.171` | pinky_tip `43.576` → pinky_middle `29.594` |
| 71 / raw | `0.00758541→0.00296106 / 60.96%` | `27.542→17.208` | `29.724→13.103` | `25.171→20.507` | pinky_tip `38.487` → pinky_middle `28.126` |
| 141 / raw | `0.00857393→0.00360521 / 57.95%` | `29.281→18.987` | `32.775→14.865` | `25.310→22.363` | pinky_tip `41.137` → pinky_middle `28.479` |
| 0 / +mean | `0.00532175→0.00178478 / 66.46%` | `23.069→13.360` | `19.543→7.717` | `26.123→17.245` | thumb_middle `36.432→25.381` |
| 71 / +mean | `0.00602963→0.00181923 / 69.83%` | `24.555→13.488` | `22.549→7.772` | `26.410→17.420` | thumb_middle `34.297→24.776` |
| 141 / +mean | `0.00576683→0.00175921 / 69.49%` | `24.014→13.264` | `21.341→7.769` | `26.418→17.073` | thumb_middle `35.424→24.759` |

聚合 tracking 与 RMS 的下降不等于所有点单调改善：0/raw 的 index_middle、middle_middle、ring_middle
分别增加 `0.200/1.747/1.977 mm`，141/raw 的 ring_middle 增加 `1.183 mm`；其余案例十点均未增加。
完整报告并列保存十点各自欧氏误差及变化、tracking 绝对/百分比下降、每关节相对旧解变化、相对 m
距离、上下限距离、梯度分解、返回码和复算检查。`||q_probe-q_old||₂` 为 `1.905–2.364 rad`，说明本次
从旧解出发的局部优化调用并不等于“小关节扰动”；它仍不能证明全局最小值或机构不可达下界。

probe 配置 SHA-256 为 `a4fb88d8ab6174581437af0597595fa9c04cafdcbbe13fefdca5e54f7dbac65e`，
脚本为 `5050533e87599cbc6f94d31c000cef5022508e647643084db27b0110f3586c11`，报告为
`15c5db6feaf0f5c0bfafc6c81e878c84700505e229440240b5cccec9c3c7b08a`；既有 solver 哈希仍为
`fd8c753a03664cbc426a91796b2a8e5befd295a007edea163df7af4f23acd89d`。

允许结论只限于：去除中点锚定并从原解继续优化后，当前六个局部解的 tracking 降低了
`55.86%–69.83%`。不能据此断言剩余误差来自错误 mapping/机构差异、批准 `alpha=0`、选择 mean 分支，
或升级十点/输入合同。未新增静态图，因为同一干预的数值表和逐点报告已足够；该 probe 不构成
M1-A 验收。当前完整序列候选状态见 3.8，M0 仍 **BLOCKED**。

### 3.6 帧 0..15 I/T 短序列候选诊断

本诊断从登记 perception NPZ 读取原始 source frame `0..15` 的 pose45/betas/valid；16 帧均
`left_mano_valid=true` 且数值 finite。`proposed_raw_pose` 和 `proposed_plus_hands_mean` 独立解码，
以同次 J0 归零并复用 calibration v2 的同一固定 A，再严格按候选 FK 的十点顺序生成目标。没有读取
world/root offset、Ours reference、认证 q、confidence 或 GraspPose；没有重采样、补帧或假 timestamps。

I（`independent_midpoint_control`）每帧 `q_init=q_prev=m`。T（`previous_solution_sequence`）的 frame 0
与 I 共用同一次中点求解，frame `t>0` 只使用同分支上一帧成功解作为 `q_init=q_prev`。两个条件都使用
`alpha=0.004`、NLopt `LD_SLSQP`、解析梯度和既有固定预算/阈值。两分支 frame 0 各只求解一次，故完整
覆盖为 62 次实际调用、64 条条件/帧结果，而不是 64 次调用。

**执行检查：64/64 PASS。**62 次实际调用全为 code `3 / FTOL_REACHED`，calls 范围 `18–60`，无异常、
负码、预算耗尽、clipping 或限位违例。全部 q/landmarks/objective finite；reported 与独立重算 objective
最大绝对差 `0`；final total 相对本次 initial total 的最大变化仍为下降 `-1.4279e-5`。无 T 递推中断或
未执行帧。详细报告为
[`short_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_v1.json`），
逐帧 q/误差/连续性数组为
[`short_sequence_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_arrays_v1.npz`）。

| variant | mean tracking I / T | mean all-10 RMS I / T (mm) | mean tip RMS I / T (mm) | mean middle RMS I / T (mm) | mean `||Δq||₂` I / T (rad) | max `||Δq||₂` I / T (rad) |
|---|---:|---:|---:|---:|---:|---:|
| raw | `0.0087303 / 0.00516863` | `29.547 / 22.657` | `33.299 / 20.743` | `25.242 / 24.295` | `0.01306 / 0.12209` | `0.03542 / 0.40511` |
| +mean | `0.00544726 / 0.00277692` | `23.339 / 16.547` | `19.647 / 9.246` | `26.521 / 21.406` | `0.01024 / 0.11454` | `0.03105 / 0.37674` |

16/16 帧在 I/T 间可配对；共享 frame 0 数值相同，T 在每个分支余下 15/15 帧的 tracking 与 all-10
RMS 都低于 I。但 T 的平均及最大帧间关节增量也都高于 I，而目标本身相邻十点 RMS 位移均值仅为 raw
`0.4569 mm`、+mean `0.3921 mm`。曲线见 [raw](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_raw_pose_v1.png`）
和 [+mean](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）。

配置、几何准备脚本、求解脚本、绘图脚本、准备数组、准备报告、诊断数组、诊断报告的 SHA-256 依次为
`5c7c0aa2…3ac4`、`025f0282…3d79`、`596cf3f9…f81d`、`380578fb…881f`、`414eb431…3ba5`、
`d8cce50e…d0ef`、`b8bc35d1…e2b3`、`9b9c683d…a3ec`；完整哈希亦保存在报告和可视化证据中。
T 相对 I 同时改变初始化与正则参照，不能把差异仅归因于其中一项。该结果没有真实几何质量阈值，
不批准正式时间策略、mean 或十点/输入合同，也不构成 M1-A PASS；几何质量仍待 reviewer 评估。
当前完整序列候选状态见 3.8，M0 保持 **BLOCKED**。

### 3.7 固定目标递推归因对照

本对照不重新解码 MANO 或重算 A，而是校验并直接加载 16 帧 I/T 诊断的配置、报告、目标/q 数组。
每个分支的 `q_F[0]` 与上一轮 I/T 共享 frame-0 解完全一致，max-absolute 差均为 `0 rad`；固定目标
明确保存为同分支 `target_landmarks_S_m[frame 0]` 在 `k=0..15` 的重复。`k` 仅为诊断迭代索引，不能
解释为 source frame、时间戳或正式 warmup。

对 `k=1..15`，每步使用同一 `y0` 和 `q_init=q_prev=q_F[k-1]`，保持 `alpha=0.004`、候选 FK/限位、
解析梯度及冻结 NLopt 设置。两个分支共新增 30 次调用，执行检查 **30/30 PASS**：全为 code
`3 / FTOL_REACHED`，calls `14–54`；无异常、负码、预算耗尽、clipping 或限位违例；reported 与独立
重算 objective 最大差 `0`。每步初始正则为零，final total 低于 initial total，final tracking 也未高于
上一迭代 tracking；最大 tracking 变化仍为下降 `-2.7015e-5`。

| variant | F tracking `k0→k15` | F all-10 RMS `k0→k15` (mm) | F tip / middle RMS `k0→k15` (mm) | F mean/max step (rad) | F/T 相对 q[0] 净位移 at k15 (rad) | T–F q / FK at k15 |
|---|---:|---:|---:|---:|---:|---:|
| raw | `0.00862765→0.00429513` (`-50.22%`) | `29.373→20.725` | `33.216→17.080 / 24.945→23.818` | `0.12021 / 0.40516` | `1.41378 / 1.37473` | `0.22209 rad / 4.333 mm` |
| +mean | `0.00532175→0.00218649` (`-58.91%`) | `23.069→14.787` | `19.543→7.611 / 26.123→19.477` | `0.11163 / 0.37688` | `1.42921 / 1.46698` | `0.14006 rad / 3.042 mm` |

F 在目标不变时仍产生 `1.41378/1.42921 rad` 的相对 q[0] 净位移并持续降低 tracking，数值上接近既有
变化目标 T 的 `1.37473/1.46698 rad` 净位移。允许结论仅为：当前递推机制在没有目标运动时也继续调整，因此 T 的
运动不能全部归因于视频输入。F/T 路径差不是可加的输入运动贡献；非线性优化与路径依赖不支持严格
分解。F tracking 针对固定 y0，T tracking 针对各自 y[k]，不能把二者 total 直接当作同一目标比较。

完整逐迭代 tracking/regularization/total、十点逐点误差、关节步长/最大关节、相对 q[0] 净位移、近限位项、
返回码和 F/T 差见[报告](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_v1.json`）；
[数组](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_arrays_v1.npz`）
明确标注为归因诊断，不是 source-frame 序列、warmup、reference 或 simulator 轨迹。曲线见
[raw](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_raw_pose_v1.png`）
与 [+mean](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）。
旧 F 报告字段 `cumulative_q_displacement_rad_k15` 与数组字段 `cumulative_q_F_from_0_l2_rad` 的名称
保持不变以保全既有哈希，但其计算值是 `||q_F[k]-q_F[0]||₂`（净位移），不是累计路径长度；旧 artifact
不改写。
配置/诊断脚本/绘图脚本/报告/数组 SHA-256 为 `0b6658a6…0b3a`、`5479c2f2…7a12`、
`feee6ff5…c6c0`、`803554f8…369d`、`e0cbb400…0d0f`。本结果不证明全局最优、不批准固定目标
warmup、正式初始化/时间策略或 mean/十点合同；该归因对照不构成 M1-A 验收。当前状态见 3.8，
M0 仍 **BLOCKED**。

### 3.8 完整 142 帧 T 候选几何诊断

完整诊断严格复用已登记 T 流程。前缀核验确认：0..15 source frame 连续；两个分支 32 条 T 记录全部
执行通过；配置、perception、MANO 模型、固定 A、十点、FK、solver、manifest 限位及专用环境路径/哈希
与本轮一致；frame 0 为中点初始化，frame 1..15 的 `q_init/q_prev` 与本分支前一帧 `q_final` 的
max-absolute 差均为 `0 rad`。前缀目标与完整准备数组的差为 `0 m`。因此前缀逐值复用，solver 重跑数为
0；本轮只从两个 `q_T[15]` 分别求解 frame 16..141。

完整 perception `frame_ids` 正好为 0..141，142 帧 `left_mano_valid` 全为 true，pose45/betas 全部 finite；
没有填补、插值、删帧、重采样、假 timestamps、warmup、重复首帧或 alpha=0/F 初始化。后缀新增
252/252 次求解均返回 `3 / FTOL_REACHED`，calls 范围 `14–46`。无异常、负码、预算耗尽、clipping 或
限位违例；reported/独立重算 objective 最大差 `0`，final total 相对当次 initial total 最大变化仍为下降
`-4.1780e-7`。完整报告合计 284/284 条通过记录，其中 32 条为复用前缀、252 条为本轮新执行。

| variant / 描述分段 | mean tracking | mean all-10 RMS (mm) | mean tip / middle RMS (mm) | mean/max `||Δq||₂` (rad) | mean target step (mm) |
|---|---:|---:|---:|---:|---:|
| raw / 0..15 | `0.00516863` | `22.657` | `20.743 / 24.295` | `0.12209 / 0.40511` | `0.4569` |
| raw / 16..141 | `0.00335876` | `18.289` | `13.100 / 22.235` | `0.04727 / 0.15244` | `1.0655` |
| +mean / 0..15 | `0.00277692` | `16.547` | `9.246 / 21.406` | `0.11454 / 0.37674` | `0.3921` |
| +mean / 16..141 | `0.00188959` | `13.742` | `7.959 / 17.726` | `0.03610 / 0.11573` | `0.8105` |

0..15 与 16..141 只是预先指定的描述性分段，不能命名为已证明的过渡期/稳态。初始化之后关节变化仍
持续存在：两个分支后缀最大 `||Δq||₂` 均在 frame 27，raw 为 `0.15244 rad`、最大分量
`left_index_MCP_AA=0.09123 rad`；+mean 为 `0.11573 rad`、最大分量
`left_index_MCP_FE=0.07085 rad`。后缀最大单点误差分别为 frame 21 `pinky_middle=30.358 mm` 和
frame 16 `thumb_middle=28.163 mm`。完整序列最大误差与步长仍在 frame 0/1，首段未被删除。

位移字段严格区分：`||q[t]-q[0]||₂` 是净位移，逐步 `||q[k]-q[k-1]||₂` 求和才是累计路径长度。
frame 141 的 raw/+mean 净位移为 `2.14479/2.18557 rad`，累计路径长度为 `7.78712/6.26707 rad`。
近限位频次较高：raw 的 `left_pinky_CMC`、`left_pinky_MCP_FE` 各 `141/142`，
`left_index_MCP_FE=139/142`；+mean 的 `left_thumb_MCP_AA=141/142`、
`left_thumb_MCP_FE=123/142`。这些只描述候选 q 与候选软限位，不证明权威 runtime 或物理可执行性。
后段 `16..141` 的 middle RMS 仍为 raw `22.235 mm`、+mean `17.726 mm`；完整候选序列已经运行，
但 residual 的来源仍未由现有证据区分，不能自动归因于 landmark、mean、候选限位或机构差异。

完整逐帧/分段结果见[报告](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_v1.json`），
[数组](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_arrays_v1.npz`）
明确标记为 `full_sequence_candidate_diagnostic`，不是 robot reference 或 simulator 执行格式。完整曲线见
[raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_curves_v1.png`）、
[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_curves_v1.png`）；
固定 frame 0/15/71/141 对照见 [raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_static_v1.png`）、
[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_static_v1.png`）。
配置/几何准备脚本/求解脚本/绘图脚本/报告/数组 SHA-256 为 `a61c47e9…e974`、`82ef836c…5250`、
`8f8fef60…10ea`、`cdfac8f0…249f`、`1f75cb81…60f3`、`5d9f5947…852c`。几何质量不自动标 PASS，
两个 mean 不在此选择。正式输入/十点/初始化时间策略、权威 runtime、reference 及姿态/物理验收仍未
批准或完成，因此正式 M1-A 验收 **NOT PASSED**，M0 仍 **BLOCKED**。

### 3.9 已完成：完整序列限位与残差归因审计

本轮只读取 `full_sequence_candidate_diagnostic_arrays_v1.npz` 的既有 target、`q_final`、实际 `q_prev`
及限位，并从注册归档 `pour17/world/world_manifest.json` 再次读取左手 22 项名称/软限位。未调用优化器、
未重新解码 MANO、未重算 A、未修改任何旧代码/配置/资产/结果，也未生成轨迹。284 组候选 FK 全量
重算与保存值 max-abs 差为 `0 m`；固定帧分解的 `g_tracking+g_smooth` 与冻结 objective gradient
max-abs 差为 `0`。残差方向统一定义为 `FK(q)-target`，坐标为候选 Sharpa root frame。

**长期贴边。**贴边定义为距相应候选 manifest 边界 `≤1e-6 rad`。raw 全序列中
`left_pinky_CMC`、`left_pinky_MCP_FE` 均在下界 `141/142`（最长连续 frame `1..141`），
`left_index_MCP_FE` 在下界 `139/142`，`left_pinky_DIP` 在上界 `130/142`，
`left_index_DIP` 在上界 `126/142`；在描述性 `16..141` 中，`left_pinky_CMC`、
`left_pinky_MCP_FE` 与 `left_index_DIP` 均为 `126/126`。+mean 全序列中
`left_thumb_MCP_AA` 在下界 `141/142`（frame `1..141`），`left_thumb_MCP_FE` 在上界
`123/142`（frame `19..141`），随后是 pinky/ring/index PIP 上界 `117/106/103` 帧；对应
`16..141` 计数分别为 `126/123/117/106/103`。这是候选限位占用事实，不把贴边标作错误或不可执行；
所有 22 关节的上下界次数/占比、最长区间和最小边距见紧凑表。

**残差承担。**按累计 tracking 平方和，raw 全序列 middle/tip 分别承担 `70.96%/29.04%`；主要点为
`pinky_middle=23.17%`、`thumb_middle=15.61%`、`index_middle=14.30%`，按手指则 pinky/index
为 `35.05%/23.77%`。+mean 的 middle/tip 为 `83.17%/16.83%`；主要点为
`thumb_middle=34.22%`、`pinky_middle=23.98%`、`ring_middle=9.99%`，按手指则 thumb/pinky
为 `41.51%/27.41%`。在 `16..141` 这些排序不变。主要 middle 点的
`||mean(residual)||/mean(||residual||)` 为 `0.9854..0.9943`（raw）和 `0.9893..0.9922`
(+mean)，显示其 root-frame 残差方向在序列内高度一致；这只是方向相关性，不认作标定错误。逐点
mean/median/P95/max@frame、mean xyz、xyz 标准差和方向一致度均在报告中保留。

**固定帧梯度。**预先固定 frame `0/15/16/27/71/141`。两分支 frame 0 均无贴边关节；其余 raw
五帧分别有 `7/8/7/9/11` 个、+mean 分别有 `1/1/3/6/5` 个贴边分量。共 58 个贴边分量中，
tracking 单独与 total 的负梯度下降方向均按同一符号约定指向边界外；因此本次抽样不是“只有贴边而
无梯度证据”，而是支持“当前候选边界限制这些关节在这些帧的局部一阶改善”。投影后 total L2 范数
为 `1.329e-7..7.887e-7`，而未投影为 `7.151e-4..4.164e-3`。这不外推为全局最优、不可达、
全部残差的唯一原因或限位应被放宽。

完整逐关节/逐点/贴边梯度表见[紧凑报告](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/M1A_LIMIT_RESIDUAL_AUDIT_V1.md`），
完整 22 维原始/投影梯度见[JSON](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_v1.json`）与
[NPZ](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_arrays_v1.npz`）。报告/数组/配置/
FK/Jacobian/注册 manifest member 的实际 SHA-256 分别为 `990d8916…907c`、`4a02dc8b…8a78`、
`9bf06f7a…2ced`、`1824dd4e…2c64`、`fd8c753a…d89d`、`2339f7a9…dcd5`；完整值见[哈希清单](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_hashes_v1.json`）。

仍不能区分机构尺寸/运动学 embodiment、十点的解剖/分段定义、固定 A/尺度、历史 mean/decoder 语义、
局部极值/求解路径或权威 runtime FK/物理限位等价性。不能因任一分支残差较低选择 mean，也不授权
放宽限位、删除 middle、修改 `alpha`、继续优化、replay 或训练。正式 M1-A 仍 **NOT PASSED**，M0 仍
**BLOCKED**。该任务当时停止；reviewer 随后接受该审计并单独授权下述 3.10 的只读检查。

### 3.10 已完成：五指 tip–middle 目标距离与候选 Sharpa 可达区间相容性

本轮严格按名称配对五组 tip/middle，读取完整序列已绑定 target、`q_final`、候选 FK 配置、注册 manifest
限位及实际 URDF；没有依赖点数组位置猜配对。scale 保持 `1`。未调用优化器，未重新解码 MANO 或重算
A，也未修改限位、`alpha`、mean、点定义、旧资产或旧结果。

**实际相对运动链。**从每个 middle link 到 fingertip link 的唯一 URDF 有向链均为“一个无 mimic 的
revolute + 两个 fixed”：thumb 为 `left_thumb_IP`，index/middle/ring/pinky 分别为各自 DIP。其余共同
上游关节不在 middle→tip 相对链中，因此不改变该点对距离。五指都满足预登记解析前提，没有跳过项。

在该 revolute 的运动前坐标系按 `d(q)^2=A+B cos(q)+C sin(q)` 计算，并只检查 manifest 闭区间端点与
区间内 `atan2(C,B)+kπ` 驻点。结果为：thumb 使用 `q∈[0,1.7452999353] rad`，距离区间
`[30.913684,47.110935] mm`；四个长指各使用 `q∈[0,1.3962998390] rad`，距离区间均为
`[32.661645,41.759257] mm`（ring 因 URDF rpy 文本精度产生小于 `5e-5 mm` 的末位差，完整值保留在报告）。
没有用稠密采样或无约束角度替代解析极值。

| variant | finger | target min/median/max (mm) | 明确超界帧 | 方向 | 最大 `delta` @ frame |
|---|---|---:|---:|---|---:|
| raw | thumb | `46.9390 / 47.6811 / 48.1292` | `117/142` | target 过长 | `1.0183 mm @ 10` |
| raw | pinky | `29.2513 / 29.6834 / 30.4386` | `142/142` | target 过短 | `3.4103 mm @ 93` |
| +mean | index | `32.4774 / 32.6789 / 33.1785` | `64/142` | target 过短 | `0.1843 mm @ 38` |
| +mean | pinky | `29.7782 / 30.3272 / 31.2629` | `142/142` | target 过短 | `2.8835 mm @ 93` |

其余 raw index/middle/ring 与 +mean thumb/middle/ring 均无 `delta>1e-8 m` 的明确超界；这只表示该项
必要条件未排除它们的点对距离可达性，不证明完整十点目标可达。

**独立核对与必要下界。**五指在下界/中点/上界及全部区间内解析驻点的解析距离与现有 FK 最大差为
`2.776e-17 m`；完整 284 个既有 q 的解析距离与 FK 最大差同为 `2.776e-17 m`，机器人实际距离超出
解析区间的最大数值量为 `2.776e-17 m`。按每对 `delta²/2` 后对五对求和，raw 的逐帧 tracking 必要
下界 min/median/max 为 `2.98887e-6 / 4.51199e-6 / 5.98527e-6 m²`，+mean 为
`9.78280e-7 / 2.72492e-6 / 4.15720e-6 m²`；实际 tracking 分别为
`2.78679e-3 / 3.32239e-3 / 8.62765e-3 m²` 与
`1.76248e-3 / 1.87847e-3 / 5.32175e-3 m²`，低于必要下界的帧数均为 `0`。该下界通常不紧，实际值减
下界不能归因为某一种单独原因。

完整运动链、`a/b/n/A/B/C`、驻点、解析/FK 核对、逐分支统计及下界表见[紧凑报告](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/M1A_TIP_MIDDLE_DISTANCE_COMPATIBILITY_AUDIT_V1.md`）；逐帧
`d_target`、机器人距离、`delta`、每对及总 tracking 下界见[NPZ](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_arrays_v1.npz`），完整记录见[JSON](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_v1.json`）。报告/数组/配置/FK/点定义配置/URDF/注册 manifest member 的实际哈希见[哈希清单](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_hashes_v1.json`）。

任一点对目标距离超出其候选区间，已足以排除该帧十点同时达到零误差。两分支 pinky 均为 `142/142`
超界，因此当前 `2×142=284` 个分支/帧样本全部不可能精确零误差匹配。这不表示近似 retargeting
不可用。上述必要下界通常不紧；`lower_bound/actual_tracking` 不能解释为任何原因已解释的因果比例，
也不能把实际 tracking 减去下界归给单一因素。

明确超界只证明当前候选点定义、模型尺度与关节范围组合存在相应点对距离不相容；固定 A 的旋转或平移
不能消除该矛盾。不能由此决定应修改模型、尺度、点定义或 mean，也不按超界量选择 mean，不提出放宽
限位、删除 middle 或缩放目标。正式 M1-A 仍 **NOT PASSED**，M0 仍 **BLOCKED**。reviewer 随后接受该
解析审计并单独授权下述 3.11 的固定 q 后验诊断。

### 3.11 已完成：固定 q 的后验共同平移/刚体配准诊断

本轮复用完整序列保存的 `q_final`、target、候选 FK 与点名顺序，以 FK(q) 重建 Sharpa 十点 `X`，并以
同帧同分支目标为 `Y`；没有重解 q、重新解码 MANO、重算或写回正式 A，也没有改动 target、模型、
限位、`alpha`、mean、点定义或轨迹。所有输出均标记
`POSTHOC_DIAGNOSTIC_ONLY_NOT_CALIBRATION`。

诊断按列向量定义 `X_aligned=R X+t`；数组以行存点，代码等价计算 `X R^T+t`。平移解取
`R=I, t=mean(Y)-mean(X)`；刚体解用等权十点的标准 SVD 最小二乘，强制正交且 `det(R)=+1`，禁止反射、
缩放及逐点偏移。每个分支分别拟合一个 142 帧共享变换和 142 个逐帧变换。

| variant | original tracking / RMS | shared translation | shared rigid | per-frame translation | per-frame rigid |
|---|---:|---:|---:|---:|---:|
| raw | `0.5059016 m² / 18.8751 mm` | `0.3257862 / 15.1468` (`-35.60%`) | `0.2652390 / 13.6670` (`-47.57%`) | `0.3140612 / 14.8718` (`-37.92%`) | `0.2553757 / 13.4105` (`-49.52%`) |
| +mean | `0.2825186 m² / 14.1052 mm` | `0.2176387 / 12.3801` (`-22.96%`) | `0.2113559 / 12.2001` (`-25.19%`) | `0.2132440 / 12.2545` (`-24.52%`) | `0.2039478 / 11.9844` (`-27.81%`) |

这里的 RMS 是全部 `142×10` 点平方误差均值开方，不是逐帧 RMS 的算术均值。共享刚体的 raw/+mean
平移分别为 `(19.0034,15.7784,-11.5482) / (-5.3950,2.9513,-4.6897) mm`，范数
`27.2662/7.7337 mm`，旋转角 `11.0606°/5.1868°`。对应 SVD 奇异值分别为
`(1.963603,0.864550,0.382761)` 和 `(1.021204,0.341170,0.108408) m²`，秩均为 3；逐帧拟合也全部秩 3，
没有标记几何退化或变换可能不唯一。

逐帧刚体平移范数 min/median/max 为 raw `20.7667/26.5357/32.0482 mm`、+mean
`6.6249/8.0453/26.8990 mm`；旋转角分别为 `7.6729/10.8764/14.0991°` 和
`3.8309/4.9122/12.1611°`。共享平移和共享刚体在 `1e-10 m²` 判据下各自使误差变大的帧数均为 0；
但该结果不被外推为任意共享拟合都逐帧改善。

原始 tracking 与旧数组逐帧最大差为 `0`；同一拟合范围内 SSE 次序全部通过。所有刚体的最大正交性和
`det(R)-1` 绝对误差均为 `2.22045e-15`，全部 tip–middle 距离变换前后最大差
`6.24500e-17 m`。逐点及 tip/middle 分组表、每帧统计、共享/逐帧 SVD 和核验见[紧凑报告](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/M1A_FIXED_Q_POSTHOC_RIGID_ALIGNMENT_AUDIT_V1.md`）、[完整 JSON](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`）与[数组](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_arrays_v1.npz`）；输入、模型、manifest、FK、点定义、配置、脚本、报告与图的实际哈希见[清单](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_hashes_v1.json`）。FK 实际 SHA-256 为
`1824dd4eaeef280c6d19008e6707f25935889faa9f219960778617ebcbc82c64`（短写 `1824dd4e…2c64`）。

共享刚体的降低只说明当前固定 q 残差包含可被共同刚体变换减少的成分。q 已在原 A 下求得，后验配准
也可能补偿关节姿态、机构差异或其他误差，不能直接推断正式 A 错误；逐帧结果不批准逐帧标定或腕部
动作。刚体不改变 tip–middle 距离，不能消除 3.10 已证明的点对冲突。不得据配准后残差选择 mean。
正式 M1-A 仍 **NOT PASSED**，M0 仍 **BLOCKED**；本轮停止，不进入新优化、replay 或训练，是否进一步
检查 A 的构造依据交由 reviewer 决定。

### 3.12 已完成：固定 A 的构造依据审计

本轮只读取 wrist-zero 标定配置 v2 登记的 MANO model/decoder/selector、既有 neutral/raw canonical
数组、候选 URDF、landmark 配置及注册 manifest member。所有路径与登记值一致；没有运行 MANO layer、
NLopt、retargeting 或拟合新变换。

MANO 构造锚点 `J0/J5/J9/J13/J17` 已由官方 21 点 selector 回溯到 raw joints `0/1/4/10/7`；本轮数值
直接来自既有 wrist-zero canonical 数组。Sharpa 局部根是 `left_hand_C_MC`；index/middle/ring MCP_VL
由各自 MCP_FE 从根直接定义，pinky 则经 `left_pinky_CMC→left_pinky_MCP_FE`。四个长指的 MCP_FE 与
MCP_AA 轴原点在 q=0 canonical 中均共点，最大距离 `0 m`；root-frame 轴范数最大误差 `2.22e-16`。

静态复算确认当前 A 对齐的是四 MCP 均值和由 `middle-root`、`index-pinky` 构成的正交基，不是直接
对齐 wrist/root。复算 R/t 与 v2 配置最大差均为 `0`；canonical 下
`A(J0)-left_hand_C_MC=(-2.008180,-1.578328,9.231794) mm`，范数 `9.578618 mm`，该非零本身不是 bug。

在限定的登记材料中没有可引用证据把 `left_hand_C_MC` 认证为 MANO J0 的解剖对应点；其结论是：
**用于构造纵向轴的工程对应假设，解剖对应未验证**。也没有发现坐标系混用、错误 frame、索引错误、
变换方向错误或配置/实际构造不一致等明确实现矛盾。完整逐项分类见[紧凑报告](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/M1A_FIXED_A_CONSTRUCTION_BASIS_AUDIT_V1.md`），机读证据见[JSON](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.json`），等比例示意图与实际输入/脚本哈希见[图](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.png`）和[清单](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_hashes_v1.json`）。当前 A 可复现，但四 MCP 均值、两条构轴参考、单一 canonical 与 scale=1 仍是未认证的工程约定；不升级为正式合同。M1-A 保持 **NOT PASSED**，M0 保持 **BLOCKED**。

### 3.13 已完成：既有完整序列候选几何可视化交付

只读复用完整序列既有 q/Y 并重算 FK 作图；284 组十点与保存值 max-abs 差为 `0 m`，tracking 差为
`0 m²`，同步 GIF 实际含 142 帧且无插值或首帧复制。两分支共用固定根、相机、等比例范围及标记规则；
固定 frame `0/15/27/71/141` 均含掌面与侧向视角，并标出 `thumb_middle`、`pinky_middle`。登记 URDF
左手子树的 28 个 visual mesh 条目本地均为 Git LFS 指针，故明确采用骨架/landmark fallback，未以
collision 或惯性几何替代；不能据图评估表面穿透。直接入口见[动画](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/m1a_full_sequence_candidate_synchronized_skeleton_landmarks_v1.gif`）、[固定帧图](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/m1a_full_sequence_candidate_fixed_frames_skeleton_landmarks_v1.png`）、[说明](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/M1A_FULL_SEQUENCE_CANDIDATE_GEOMETRY_VISUALIZATION_DELIVERY_V1.md`）和[哈希清单](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/m1a_full_sequence_candidate_geometry_visualization_delivery_hashes_v1.json`）。这是展示交付，不选择 mean、不作碰撞/物理或正式质量验收；M1-A 保持 **NOT PASSED**，M0 保持 **BLOCKED**。

### 3.14 已完成：登记 Sharpa visual mesh 精确恢复及既有候选表面展示

登记 URDF 左手子树的 28 个 visual 引用按 LFS oid 去重为 13 个文件、`5,954,492` 字节；全部对象从
已登记仓库的现有 LFS 负载逐项恢复到 ViViDex 隔离目录，远端与本地的字节数及 SHA-256 均与 pointer
一致，未覆盖原 pointer/URDF，也未执行全仓库 LFS pull 或使用替代表面。逐项表见[资产恢复报告](README.md#archive-access)（归档：`artifacts/m1a_sharpa_visual_mesh_exact_recovery_v1/M1A_SHARPA_VISUAL_MESH_EXACT_RECOVERY_V1.md`）。

表面渲染按 `T_root_link · T_link_visual_origin · Scale_mesh · p_mesh` 接线，复用原 q/Y、FK、相机与两分支；
FK 十点、tracking、RMS 与旧数组差均为 `0`，同步 GIF 含 142 个互异 source frame。入口见[表面动画](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_synchronized_surface_v1.gif`）、[固定五帧掌面/侧向图](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_fixed_frames_surface_v1.png`）、[说明](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/M1A_FULL_SEQUENCE_CANDIDATE_SURFACE_VISUALIZATION_DELIVERY_V1.md`）及[实际哈希清单](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_surface_visualization_delivery_hashes_v1.json`）。骨架版保留；表面图只供手形视觉检查，不证明穿透/碰撞安全、物理可执行或任务成功，不选择 mean。M1-A 保持 **NOT PASSED**，M0 保持 **BLOCKED**。

## 4. 固定局部标定候选

### 4.1 旧 uncentered-canonical A：保留的旧坐标证据

实际旧代码以 `mano_palm_frame(neutral["joints21_m"])` 直接使用未减 wrist 的 canonical MANO 点，旧产物也
记录了非零 canonical wrist；此前标作 `p_H` 的输入在新的严格命名下应记为 `p_old`。旧候选保持
`A_old(p_old)=R_old(1.0·p_old)+t_old`，米制且不逐帧拟合。固定姿态为 MANO
`beta=0, global_aa=0, pose45=0`（raw-pose calibration canonical）和 Sharpa `q=0`。

- 两侧原点：四个长指 MCP 的均值；
- `x=normalize(middle_MCP-wrist/root)`；
- `y_raw=index_MCP-pinky_MCP`；
- `z=normalize(x×y_raw)`，再令 `y=normalize(z×x)`；
- `R=B_S B_H^T`，`t=O_S-R O_H`；任一构轴向量小于 `1e-8` 时失败。

当前 v4 代数检查：`||R^TR-I||_F=3.33e-16`、`det(R)=1.0000000000000002`、round-trip
`1.55e-17 m`、reflection=`false`。这些只证明固定候选是右手正交刚体变换；**它仍为 PROPOSED，未获批准，
也不证明 reconstruction-world→Isaac、MANO root→robot wrist 或 wrist→EE 对齐。**没有按手形、物体交互或
任务分数调整尺度和偏移。旧 A 在 `p_old` 坐标内继续作为证据，不宣称其计算错误；但不能直接与新的
wrist-zero `p_H` 混用。

### 4.2 wrist-zero hand-local A：新坐标配套候选

新配置当前版本为 [`m1a_wrist_zero_hand_local_calibration_candidate_v2.json`](configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json)。v2 仅修正公式适用范围：一般样本用
`D(beta,pose45,0,0;variant)` 减同次解码 `J0`，canonical 专用式才是
`Pcanonical_H=Pcanonical-w_ref`；运行逻辑和全部数值不变，v1 与既有代数证据原样保留。该配置继续采用完全相同的 canonical、构轴规则、
候选 URDF/FK、Sharpa `q=0`、root link `left_hand_C_MC` 与 `scale=1`，只先将 MANO canonical 21 点减去
同次解码 wrist `w_ref`。该 raw-pose canonical 是对旧标定定义的延续，不选择真实输入 mean 语义。

full-precision 值为：

- `w_ref=[-0.09566993092407175, 0.006383428857461437, 0.006186305280135195] m`；
- `R_new=R_old=[[-0.014021578466266971,-0.9989024597392235,0.044690840943569454],`
  `[-0.136442201351016,-0.042365752788848576,-0.9897417181674839],`
  `[0.9905487979033658,-0.019975457881192,-0.13569841581475298]]`；
- `t_old=[0.0027503280391410413,-0.008238460426711069,0.10496451298191413] m`；
- `t_new=[-0.0020081801195304307,-0.001578327634321078,0.009231794169036217] m`。

解析关系 `p_old=p_H+w_ref`、`R_new=R_old`、`t_new=t_old+R_old w_ref` 已通过：R 差 `0`，t 解析差
`4.336808689942018e-19 m`，canonical 全 21 点映射差 `2.7755575615628914e-17 m`。最大 B/R
正交残差 `3.3342503417246054e-16`，`|det(R_new)-1|=2.220446049250313e-16`，掌原点映射误差 `0`，
掌基方向误差 `2.220446049250313e-16`；均通过预登记 `1e-10` 代数容差。数值来源 v1 配置 SHA-256 为
`6506ed41389f8d56771118bd73c5f3e3c0c984675ea9cd26dc9f0a522c711149`，公式范围修正版 v2 SHA-256 为
`7c37127838a94589af3d8b64cb860ea7efb74b77d2034710c27007364eb81c03`；诊断报告 SHA-256 为
`b102a28d683d453785118d50e62f43d4afa01ac17b16cd6366612c3ddb405cfd`，脚本 SHA-256 为
`5fa06830dfdbb4f82c55d512766268b4ee196cec20d704d5faf1af1667f9eb80`。

同一个固定 `A_new` 应用于已有七样本 × 两个 mean 分支 diagnostic arrays，14/14 shape/finite PASS；没有
比较分支。非零 beta 仍按每个样本自身 wrist 局部化，不重标定 A。该结果只验证同一 canonical 几何的坐标
换算；新 A 仍为 **PROPOSED**，不批准 landmark 解剖、thumb middle、权威 Sharpa runtime、正式输入合同或
任何机器人目标，也不构成正式 M1-A 验收；当前完整序列候选状态见 3.8，M0 仍为 **BLOCKED**。

## 5. Pour17 HaWoR 输入语义溯源

已在限定范围内检查交接包、其 manifest/HASHES/导出脚本、当前重建包装/平滑/消费源码及这些文件明确指向的
HaWoR 路径。没有新的外部材料时，不再重复扩大搜索。

| 阶段 | 证据 | 等级与结论 |
|---|---|---|
| 交接包参数 | `world_fused.npz` `e4816486…43eb8` | 只验证交接包字节：含 trans/rot/pose45/betas/valid 和两套 global 字段；无 layer/model/run 配置，且 M0 未绑定其为权威 runtime。 |
| 当前生产包装 | `recon_pipeline/_legacy/hawor/common.py:509-564` `27d69cc6…ba09` | **当前源码支持，未关联历史 run**：把 `init_hand_pose` flatten 为 45 维；实际预测与 `cam2world_convert` 源码缺失。 |
| 当前启动入口 | `recon_pipeline/hawor/run_sequence.py:70-78` `6e51d360…d744` | **当前源码支持，未关联历史 run**：当前为 `use_infiller=False`、默认 `smooth_poses=False`；不能据此断言历史值。 |
| 可选平滑 | `pose_smoothing.py:68-168,202-207` `be445881…9ca` | **当前源码支持，未关联历史 run**：global/local 三元组按 rotvec 平滑；不加减 mean，betas 不变。 |
| z-up 导出 | `fuse/run_sequence.py:424-446,616-620` `95758901…c9ca` | **当前源码支持，未关联历史 run**：只变换 global translation/root rotation，local pose 不变。 |
| baseline 导出 | `make_pour17_perception.py:54,77-80` `e435dc7d…f93c` | 随包源码与字节关系一致：142 帧时索引为 identity，左右 trans/rot/pose45/betas 八项与 `world_fused` cast 后逐元素相等。无执行 manifest。 |
| 当前消费 | `recon_pipeline/_common/hawor_mano.py:209-229` `6eadeb51…9472` | **当前源码支持，未关联历史 run**：调用缺失的 `run_mano_left(trans,rot,pose,betas)`。 |

历史 HaWoR 的模型身份、mean、左 shapedirs 预处理及完整 root/translation 语义均为 **UNKNOWN**。当前可见
包装、平滑和导出代码没有 hands-mean 加减，但真正的生产与解码实现在缺失的 HaWoR 源码中，不能据此选择
v4 的任一 mean 分支。交接 `world_fused.npz` 含有当前检查的导出代码无法解释的
`hand_root_offset_mano`，这是 producer 版本/路径未绑定的具体证据，不能静默忽略。

精确缺口：历史 pour/17 的 `world_space_res.pth`、`run_meta.json`、启动命令/日志；实际
`hawor_video.py`、`hawor/utils/process.py`、`custom_utils.py`；历史 environment lock、model config、
`MANO_LEFT.pkl` 与哈希；producer commit/diff。交接记录的 source take 在本机和已知 ViViDex 远端均不存在；
当前重建 checkout 的 `third_party/hawor` 为空且 git worktree 元数据失效。

`world_fused.npz` 与 `pour17_perception.npz` 没有同阶段、点序/单位/坐标有证据的 MANO joints 或 vertices，
故不具备独立上游几何数值检查条件。

## 6. 不变量与待项目决定事项

必须防止重犯：

- v4 只证明指定冻结条件下的 decoder 实现一致性，不能认证历史 HaWoR 或选择 mean 分支；
- 没有新外部材料时，不重复扩大 HaWoR 搜索；
- 不得用 Ours robot reference、认证 `q`、confidence、GraspPose、CuRobo guide 或其它 Ours 派生量补齐
  ViViDex 人侧输入；
- 不得按视觉观感、物体交互效果、retargeting residual 或后续任务分数选择解码语义、尺度或偏移；
- 旧双重重排产物不得重新进入当前证据链。

仍需项目决定：

1. 若历史材料无法恢复，是否冻结一份新的可复现 MANO 输入合同；
2. 新合同采用哪个显式 mean 分支；
3. 是否批准九项 landmark 候选，以及已完成结构核验的 `proposed_thumb_internal_segment_midpoint_v1`；
4. 是否批准固定掌坐标标定候选；
5. M0 采用历史 runtime 复原，还是把交接包冻结为新共同 benchmark 并让双方重新评测。

在这些决定前，候选配置和审计证据都不是正式运行授权：**M1-A formal acceptance = NOT PASSED，
M0 authoritative runtime = BLOCKED。**

