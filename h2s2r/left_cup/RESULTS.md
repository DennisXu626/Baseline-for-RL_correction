> 团队同步：本包已关闭。当前权威状态及外部资源获取见[REVIEWER_HANDOFF.md](REVIEWER_HANDOFF.md)。本文件实验事实保留；app.close失败不使已完成D数据无效，也不放行新启动。

# 左杯 A 环境差异定位、真实记录链修复与固定输入 D 结果

## 结论先行

本包完成了 A 来源定向核对、真实记录支持修复和固定异常输入 D 的一次完整 100-control 数据采集；没有修改共享/V12、右瓶、plate、算法、物理、初态或控制参数，也没有运行 policy、PPO、FABRICS advance 或 A 移植。

- **记录链实现通过。** CPU 测试调用运行入口实际使用的同一函数，覆盖 torch 嵌套序列化、四元数、路由、传感器注册、退出裁决、录像关闭和首 12 子步原子保存。第 1 次 App 启动在 control 前暴露 `cup_to_table` 注册缩进错误；修复通过真实 helper 回归后，第 2 次完成 100 controls/1200 子步。
- **第一次实际接触过程已定位。** 第 1 control 的第 1 个物理子步，也就是命令开始约 4.167 ms 后，最先出现非零力的是 `left_pinky_MP` 对杯：7.057 N，ContactView separation 为 +0.0566 mm。接触点位于记录杯心径向约 36.888 mm、下方约 39.152 mm，即杯的下侧/底缘附近。杯在该子步已相对 reset 移动 0.587 mm；第 12 子步累计移动 6.610 mm。随后拇指近节从子步1、食指近节从子步2产生非零力，不能把杯运动全部归因于小指。
- **没有后续状态回写解释这次运动。** 记录器在 explicit reset 前安装；reset 的 Robot/Aux/Object 写入均成对保存，control 阶段状态写入为 0。constructor 内的活动按计划明确标为未观测，未冒充全覆盖。
- **真实记录没有证明 cooked collider 穿透。** 261,636 条实际 ContactView separation 全部为正，范围 +0.0566 至 +6.0000 mm；其中14,376条力大于0。小指中节实际 collision prim 启用、`convexHull`、contact/rest offset 4/1 mm；杯为 `convexDecomposition`、2/0 mm。结合约6 mm的总接触边界，这支持“初态立即触发接触响应”，但不支持把离线 visual/input-mesh 相交直接改称 PhysX 实际烘焙碰撞体穿透。此前 CPU 重建的输入网格相交仍成立，两类证据必须并列保留。
- **A 单项移植不具备资格。** PMIN_P17s51 的 A1 证据能确认时钟、质量/摩擦、关节顺序、机器人 MD5 与运行日志，但缺 executed commit/source tree、杯 cache bytes/hash、逐 prim collision/装配/filter/cooked identity。找到的 `f44e77b…` 只是 A2 同名候选。没有唯一 A1 碰撞资产/装配/filter 差异，故未选择候选、未运行第二条轨迹。
- **D 进程最终裁决为失败，但物理数据和媒体已完整。** 主循环及媒体在 203.358 秒完成，随后 `SimulationApp.close()` 卡在 `omni.replicator.orchestrator.stop/status → SimulationContext.render`；900 秒监督截止触发，最终 `FAILED(timeout)`。这不是 PPO 慢，也不是 control/录像未完成。按 control 后支持故障停止规则，本包不重跑、不延长、不消费剩余额度。

## D 数据定义与可信度

分析粒度是一环境每个 1/240 秒物理子步一行；100 controls×12子步=1200行。接触记录以 `global_substep` 和 34 个 ContactView 索引关联。冻结输入是 v7 `warmup0_retry3` 的 trace1..100 目标，trace0只作为实测初态，不用 measured q 替代目标。

| 检查 | 结果 | 证据 |
|---|---:|---|
| reset 左29关节、腕、杯位姿、hold=0 | PASS | `reset_validation.json` 全6项通过 |
| 100×29请求与冻结输入逐值相等 | PASS | 1200子步按control重复后 exact equality |
| joint target / processed / direct drive读回 | PASS | 三层均与冻结目标 exact equality；逐名映射 |
| control计数与显式子步 | PASS | 100/100、1200/1200 |
| policy / PPO / FABRICS调用 | PASS | 均为0；五个 guard 计数均0 |
| 接触完整性 | PASS | 34 views；count总和=261,636原始记录；无非有限值/overflow |
| reset后control状态回写 | PASS | 0条；14条reset CALL/RETURNED记录完整 |
| 首12子步原子证据 | PASS | 12个独立NPZ，索引0..11，均可读 |
| 三视角视频 | PASS | front/top/side各101帧，640×480、20fps、非黑；远端/本机SHA一致 |
| 子进程正常关闭 | **FAIL** | completion完整后卡在App close，900.913秒超时 |

原始数组复算见 [D_ANALYSIS.json](analysis/D_ANALYSIS.json)、[首12子步表](analysis/FIRST_12_SUBSTEPS.csv) 和 [完整接触时间线](REVIEWER_HANDOFF.md#external-resources)。数据质量检查本身为 PASS；进程生命周期单独为 FAIL，二者没有互相覆盖。

## 首个异常接触前后

reset 帧已在三视角中保存。物理接触从第一个被记录子步开始：

1. 子步0：`left_pinky_MP` 唯一产生非零力，7.057 N；接触点 `[-0.106401, 0.178346, 0.898578] m`，separation +0.0566 mm。杯线速度0.0941 m/s，相对reset位移0.587 mm。
2. 子步1：小指中节仍有近接触记录但力为0；`left_thumb_PP` 出现6.444 N。杯累计位移1.042 mm。
3. 子步2–5：拇指近节和食指近节共同产生力；食指近节峰值到39.531 N。杯累计位移由1.840增至3.401 mm。
4. 子步6–11：小指中节重新产生13.858→2.360 N，且间歇伴随拇指近节；杯在首个control末累计位移6.610 mm。

杯到桌面的 ContactView 全程为0条；这排除了本次记录中“杯-桌面接触先于手杯运动”的解释，但没有证明杯运动只由小指导致。100 controls结束时杯相对reset净位移117.876 mm，最终z高43.694 mm；历史固定命令确实移动并抬起了杯，但这不是 policy 成功评测或 LEFT_PASS。

## A 与 D：哪些已证实，哪些缺失

完整表见 [A_D_DIFFERENCES.md](A_D_DIFFERENCES.md)。可直接复核的要点：

- 已证实一致：机器人内容MD5、58关节顺序、dt/decimation/control时钟、已存驱动项、物体质量/摩擦和pad摩擦。
- 已证实不同但禁止本包自动移植：A的env0 XY扰动、前15 control物体钳位/回写、初始手杯状态。
- 仍缺失：A1杯 cache/cooked collision内容、左手逐prim装配与collision identity、具体手杯pair/filter运行清单、executed commit/source SHA。
- 因此 A2 同名候选不能作为实验变量，`pre_transfer_decision.json` 固定为 `DO_NOT_RUN_A_TRANSFER`。

## 实际执行差异

所有改动只在本包隔离副本：新增共享 `runtime_support.py`，让运行入口和CPU测试共用相同的序列化、四元数、路由、传感器注册、状态写入记录、原子子步保存、关闭及监督裁决函数；录像移除运行中等待SCP批准文件，保留冻结三相机、真实preview、非黑检查、完整PNG/MP4；ContactView严格按六返回值与真实 `(N,1)` 力/间距形状保存。

第1次启动发现的类体缩进错误只影响观察传感器注册；修复为实际 `register_observation_sensor` helper 后用CPU反例验收。核心 `env.py`、`left_env.py`、`cfg.py`、`ours_physics.py` 保持只读，详见 [实际统一diff](actual_execution.diff) 与 [执行版本](execution_versions.json)。没有处理 `app.close()` 新卡点，因为它出现在control后且本包停止规则禁止再启动验证。

## 资源、成本与停止决定

依据最新共享 [GPU_POLICY](../../GPU_POLICY.md) 选择物理GPU0。第二次启动前10秒：平均/最大利用率0%，最小空闲47,391 MiB，准入要求13,927 MiB；运行90个10秒样本峰值总占用6,628 MiB、最小空闲41,912 MiB、最大利用率55%，没有触发资源停止。

新包消耗2/4次App启动、1/2条有control轨迹、100/200 controls、1200/2400显式子步；App进程合计914.393秒，小于45分钟总上限；A移植、旧E2、PPO和训练均为0。旧contact-cause包的3/3历史失败原样保留，不重置、不覆盖。详见 [costs.json](costs.json)。

最终状态：`D_DIAGNOSTIC_COMPLETE_PROCESS_TIMEOUT_A_TRANSFER_INELIGIBLE`。已回答异常发生位置、开始时间、前后状态回写以及实际接触记录；仍未证明唯一根因、actual cooked collider 穿透或可部署修复。提交 reviewer 决定是否另立初始化适配/碰撞表示研究，以及是否单独处理 App close 卡点；本包不自行继续。

## 本机三视角真实视频

- [front：完整101帧](REVIEWER_HANDOFF.md#external-resources)
- [top：完整101帧](REVIEWER_HANDOFF.md#external-resources)
- [side：完整101帧](REVIEWER_HANDOFF.md#external-resources)

远端完整原始数据仍在 `/media/msc-auto/HDD/users/kailang/h2s2r_left_cup_a_env_transfer_20260912/D_replay02`；本机同字节副本、大小、SHA与可读性见 [raw_evidence_index.json](raw_evidence_index.json) 和 [delivery_validation.json](delivery_validation.json)。
