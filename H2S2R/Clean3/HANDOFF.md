# Clean3 GitHub同步截止：2026-09-13 03:02:17 Asia/Shanghai

上次维护基线2026-09-13 01:01:27的C3-P2放行记录，非GitHub提交。此后本地补了机器人路径、预检表达式、监督器failure-marker和启动包装，但远端v3仍旧版。阅读：任务README→本交接→C3-P2包→当前结果和SOURCE_IDENTITY。

R2历史2/2关闭且0controls；C3-P2另有最多2次条件启动，目前0次，禁止称“恢复attempt02剩余”。完整USD/依赖/碰撞CPU前置未通过。C3-P2累计CPU成本不完整，先核账，不能以局部0.5分钟推算全部余额。无PPO、外评、视频、checkpoint，成功率N/A。5D路线未改，22关节仅备选讨论。

03:02:17只读服务器可达，无Clean3进程；Python直接find_spec(pxr)=None，是时点事实。远端不含两个新preflight文件，cfg/guard/commands/manifest与本地不同。runtime_decision未来UTC字段未核实，不用于排序。证据GITHUB_REMOTE_SNAPSHOT_20260913.json。

本次仅同步文档，不部署、不实验、不改方法预算、不GitHub上传。GPU_POLICY.md是唯一运营权威；共享文件由team leader汇总，不覆盖别线结论。下方均为历史记录，“最新”只指当时；以本节及C3-P2为当前入口。

# Clean3 / plate — 独立 reviewer 交接

## 最新reviewer裁定：C3-P2资产恢复包（2026-09-13）

执行权威为`reports/H2S2R_CLEAN3_C3P1R_20260912/REVIEWER_C3P2_ASSET_RECOVERY_20260913.md`，R2方法合同继续有效，预算以新包为准。C3-P1R2历史2/2启动关闭不改写、0物理样本/0外评分母N/A。

已定位cfg把V12代码根误作资产根。reviewer只读远端确认正确robot_usd为`/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17/world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`，SHA256 `c2bba25aeb7b04b270df51d0c0efbe8eec95b499d8a7aef46834cefcfb71e347`，与既有冻结记录一致；不重造机器人，不修改共享源。尚须CPU stage/依赖/58关节/碰撞身份检查，路径存在不是物理通过。

新包CPU60分钟补齐实际消费路径/依赖检查及监控parser（attempt02监控曾失败），均须在SimulationApp前验证。显式新增最多2次4-env物理启动/45分钟，仅支持故障允许第二次；真实方法/物理失败不盲重试。通过后继续原未执行的256env/512轮或120分钟PPO及45分钟外评交付，总270分钟。GPU按GPU_POLICY，持续上传授权有效。reviewer本轮未启动或派发实验，仅只读定位与报告。

不要再以源码/文件清单部分通过替代真实cfg全部资源预检。常规支持修复同包自主完成；真正需要资产/控制/物理方法变更或预算到限才交付。保持5D路线，尚无物理证据支持换22关节。

## 最新 imple3 最终状态：C3-P1R2 CLOSED_ATTEMPT02_MISSING_ROBOT_USD（2026-09-13）

reviewer 上传的 v3 已解包至远端独立 `deployment_v3`，30/30 清单及五项冻结依赖身份 PASS。10 秒 GPU_POLICY 准入后选物理 GPU1（48,063 MiB free，util 2–3%，无既有 compute），设置 `RL_ISAAC_NO_GUARD=1`。最后 attempt02 运行 15.002870s，修复后的证据链保存 `FileNotFoundError`：cfg 冻结路径 `V12/world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd` 不存在；V12 配置声明的 `rl_rebuild/assets/vega_1p_sharpa_fixedtorso.usd` 及 MagicSim 回退路径同样不存在。

环境构造 begin/complete 均未到达，0 controls/0 substeps/0 frames，无视频/轨迹。启动额度 2/2 已用完；按用户停止条件没有 attempt03、路径替换、资产生成、PPO 或外评。物理有效性、零动作稳定、双手真实推进、持物和穿模均 NOT_TESTED；外评分母 0，成功率 N/A。下一步需 reviewer 在新包冻结并提供有碰撞身份的 58-DOF fixed-torso USD，同时另行授权新启动额度。机器可读结果见 `reports/H2S2R_CLEAN3_C3P1R_20260912/c3p1r2_decision.json`。

## 最新：上传已获持续授权，v3已传输并核验（2026-09-13）

用户在知悉内部源码/任务资产传输风险及具体服务器目的地后明确回复：“我确认授权此次上传，以及未来类似上传。”对Clean3内部部署源码/任务资产向已确认服务器`kailang@128.32.164.89`的独立任务目录进行同类上传，此授权持续有效，不再重复询问；不能扩张为向任意新第三方目的地传输的授权。

本reviewer已实际上传`c3p1r2_deployment_upload_v3.tar.gz`至`/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/incoming/`，scp返回0；随后远端stat与sha256sum确认26,448,738字节，SHA256 `cc424a2c22e55393022aa9586e5de21e50bea0ba64f16bf4a788e1b268f1cf8c`。本次上传审批阻塞已解除，历史拒绝/RESULTS保留原样。无需imple3重复上传同一包。

只完成传输与远端身份核验，未解包或启动attempt02、Isaac、PPO。attempt01仍计1次启动且0物理样本，剩余attempt02及原预算不重置。imple3按R2既有条件继续独立部署和后续执行；GPU准入统一按`D:/UCBP/GPU_POLICY.md`，允许满足负载及峰值余量的同卡共享，旧idle-only条款失效。当前外评分母0、成功率N/A。

## 最新 imple3 执行状态：C3-P1R2 DEPLOYMENT_APPROVAL_BLOCKED（2026-09-13，待 reviewer 审阅）

R2 的 Clean3-only 左掌 `z_min=-0.03m`、大写 `ZYX`、双掌完整 set_targets/FK/七点/端点/reset/变换/12子步 CPU 契约均 PASS；手 PCA 和 R 版范围侧车 SHA 未变。真实 cfg/env、官方 PPO、外评及录像入口已在独立本地部署包接线。

旧支持下物理 attempt01 启动 15.003s 后在 `build_cfg()` 内、环境构造前以 0 返回；0 controls、0 substeps、0 frames、无轨迹。已定位记录层根因：入口 `finally: app.close()` 会覆盖异常退出，旧监督器又未要求终态成功标志。imple3 已修为关闭前原子存异常、ready+terminal-success 双标志，并用正常/异常/直接退出 CPU 替身 3/3 PASS；attempt01 保留为 `ZERO_PHYSICS_SUPPORT_FAILURE`，不能解释穿模或方法表现。

修复后的 v3 本地包为 `D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_c3p1_20260912/c3p1r2_deployment_upload_v3.tar.gz`，26,448,738 bytes，SHA256 `cc424a2c22e55393022aa9586e5de21e50bea0ba64f16bf4a788e1b268f1cf8c`。向 `kailang@128.32.164.89:/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/incoming/` 上传在进程创建前再次被安全审批拒绝，0 bytes；未绕过。按 R2 停止条款，attempt02 尚未启动，PPO/外评 0，有效分母 0，成功率 N/A。完整执行结果和拒绝原文见 `reports/H2S2R_CLEAN3_C3P1R_20260912/C3P1R2_RESULTS.md` 与 `deployment_approval_rejection_c3p1r2_v3.json`。

> **GPU运营规则已于2026-09-12由用户替换：** [GPU_POLICY.md](../../GPU_POLICY.md) 为双A6000的当前准入规则。GPU0/1均可选择，允许资源预算与持续监测支持的同卡/跨卡并行；已有compute进程不再单独阻止启动。通过准入后在本次子进程设置 `RL_ISAAC_NO_GUARD=1` 关闭旧机器级锁，记录实际GPU映射，不删除他人锁或中断他人任务。本文下方历史“独占/空闲卡/固定GPU”运营限制不再生效；历史运行事实、科学参数、track所有权及启动额度不变。

## 最新裁定：C3-P1R2 掌边界与角度编码（2026-09-12）

执行权威更新为`reports/H2S2R_CLEAN3_C3P1R_20260912/REVIEWER_DECISION_C3P1R2.md`，覆盖R版冲突条款，其余沿用。接受imple3按旧条件停止，当前物理0/2、训练0、外评分母0，N/A。

独立XML FK复现左掌arm_center z=-0.0149358039m。V12 z>=0为固定动作裁剪常量，不是桌面或已证明可达域；reviewer此前称可达范围不准确。裁定只在Clean3侧车把左掌z下界设-0.03m，保留其余边界、初态、桌高、坐标和控制方法。固定名义FK为零点，不在线扩界。

另发现并裁定同包修复明确bug：prepare_c3p1r.py的SciPy小写zyx与实际FABRICS Rz·Ry·Rx不符，须改为大写ZYX/等价公式。旧角度若用于实际解码，会有右73.40°、左138.80°旋转偏差（非实测物理事件）。证据`reviewer_palm_audit.py/.json`；必须一次性检查左右完整掌位置/旋转往返、实际set_targets、reset、根系与世界变换及子步计数后继续物理、PPO、评测，不能只通过z比较就放行。

剩余预算CPU110分钟、物理45分钟、训练120分钟、评测45分钟，总320分钟，不重置配额。R版另报告源码上传曾被安全审查拒绝但未附具体原因：执行者先完成本地允许清单和可审阅部署包，查原拒绝原因，正常审批内解决；不能绕过或混称资源不足，仍阻塞则单独交DEPLOYMENT_APPROVAL_BLOCKED。本轮reviewer未尝试上传或启动GPU。

以下为历史交付/裁定记录，冲突以上述R2为准。

## Imple3 C3-P1R execution update (2026-09-12 22:39 CST; not reviewer verdict)

C3-P1R hand calibration is implemented in the existing independent Clean3
directory without changing the original PCA/runtime artifacts. The actual
`H2S2RFabricController.set_targets` and `reset` CPU harness passes 4/4 tests:
right/left zero-action errors are 2.63e-8/5.19e-8, fixed range ratios are
2.081357/4.332928, and full q/default_config remain intact. The next explicit
CPU gate fails: using the actual generated left FABRICS URDF and approved reset,
the left palm z in `arm_center` is -0.0149358 m, outside the unchanged V12
minimum 0 m. Imple3 stopped before Isaac/physics/PPO/evaluation without changing
palm bounds or task placement. Review package:
`D:\UCBP\reports\H2S2R_CLEAN3_C3P1R_20260912`; current success rate remains N/A
with denominator 0.

## 最新裁定：C3-P1 PCA阻塞复审与C3-P1R（2026-09-12）

接受imple3按旧门槛提前停止的交付，历史decision不改写。**当前执行权威为`reports/H2S2R_CLEAN3_C3P1_20260912/REVIEWER_DECISION_C3P1R.md`，与原C3-P1其余条款合用；冲突以R版为准。**

reviewer纠正自己制定的PCA重建门槛：实际FABRICS采用Bq任务映射及完整静态default_config，不下发mu+B.T@B@(q-mu)重建关节目标。右/左0.6367/0.6201rad重建残差投回B仅约8e-9，不能当实际跳变。不过原范围真实排除右2维/左4维初态投影，仍需动作范围适配。

裁定不改PCA语料、5维、B/mu、初态或奖励。独立动作侧车以x0=Bq0固定零点，将原绝对范围[L,U]扩成[min(L,x0-0.05w),max(U,x0+0.05w)]，w=U-L，复用既有分段中心解码；保留示范原范围和初态两向小余量，明确披露为方法适配，不再称纯分位运行范围。不是逐帧GraspPose前馈或在线重锚。静态cspace姿态条件也必须披露。最大范围宽度倍数右2.0814、左4.3329，禁止继续调范围。

CPU实际接口零动作目标误差≤1e-6后进入原物理配额，验证真实FABRICS首步和前1秒目标/实测关节相对reset均≤0.25rad、无超限/深穿，并通过原保持门，再直接继续PPO和外评。当前不是physics PASS。仍需完成实际运行接线，fake cadence不能替代。

不重置6小时预算：扣除已报25分钟，余CPU125分钟、物理45分钟、训练120分钟、评测45分钟，共最多335分钟；启动次数与原包相同。资源重新检查，大产物用远端HDD，根盘空间风险不得靠删共享数据解决。本次reviewer仅做只读远端SHA核验、CPU代数检查与报告，未启动GPU、Isaac或训练，未派发实施任务。

详细公式、真实物理门、来源SHA及复现脚本在上述R版与同目录`reviewer_pca_taskmap_check.py/.json`。此前C3-P1门槛中PCA重建误差>0.25rad的停止条款作废，保留为历史诊断；其他参考、初态、奖励、外评和视频要求继续有效。下面为imple3交付及原审阅历史。

## Imple3 C3-P1 execution update (2026-09-12 22:16 CST; not reviewer verdict)

The reviewer-frozen CPU adapter work was executed in the independent directory
`D:\UCBP\RL\Baselines\task_plate\h2s2r_clean3_c3p1_20260912`. RTS reference,
18 cm no-pressure support geometry, the permitted 2 mm one-time right-arm IK,
58-joint limits, separate 300-row PCA identities, bilateral cadence support,
and external-evaluator negative cases pass 10 CPU tests. The mandatory PCA
initial-state gate fails: clipped max joint errors are right 0.695525 rad and
left 0.768242 rad; even unbounded 5D subspace errors are 0.636731/0.620079 rad,
above 0.25 rad. Imple3 therefore stopped before Isaac, physical validation,
PPO, or evaluation and did not change PCA/corpus/initialization. Review package:
`D:\UCBP\reports\H2S2R_CLEAN3_C3P1_20260912`. Current method success rate remains
N/A with denominator 0.

## 最新接续裁定（2026-09-12，覆盖下方旧收尾状态）

用户已在新任务明确指定负责人为Clean3独立reviewer，与imple3（Sol-high）配合。当前接续审阅已完成，不再处于“等待新reviewer首次读代码”的状态。Pour17右瓶、左杯线路均未改动。未派发实施任务，未启动Isaac/GPU/训练，未修改生产功能代码。

下一包的唯一Clean3执行入口是[独立审阅与C3-P1工作包](decisions/C3P1.md)。收到用户转交prompt后按其条件放行推进；不要重复已完成CPU接入包，也不要用Pour17训练计划替代它。

- 查新：指定本地imple3结果仍为21:19左右交付，未发现晚于旧交接的新结果。
- 独立检查：实际prepare_clean3.py、V12输入/参考/cfg与远端env/right_env/train_right_lstm、原HDF5/NPZ/OBJ、原任务geometry/build_reference和DECISIONS。证据在`H2S2R_CLEAN3_PACKAGE_REVIEW_20260912/reviewer_independent/`。
- 接受30Hz、混合来源、18cm资产、22关节PCA示范输入；原候选不能直接开训。原HDF5腕本已近静止；左腕固定偏移约2.71cm，腕字段只留证。raw海绵中心盘面法向间距中位24.9cm，RTS仍12.6cm，无法作为擦拭目标原样使用。
- 已裁定必要且公开的适配：RTS相对面内位移、固定盘姿和海绵yaw、当前网格支撑高度；不加载ours后续轨迹、不额外滤波/裁帧/展宽。独立CPU几何探针覆盖0.541667、接触行程0.868447m；不是物理或训练PASS。
- 明确允许共享一次性、GraspPose-informed静态初态（基础v1第0行arm/object及独立f_grasp数组），必须披露，禁止逐帧GraspPose前馈、squeeze、pin/release、settled_oh、checkpoint。细节和约2mm支撑高度修正见工作包。这个有限初态许可不能扩张成对ours控制/参考/奖励的许可。
- 双手采用342obs/22action集中式PPO，左右都学，现有两侧H2S2R跟踪奖励取均值。V12父env左侧每控制步step一次、右侧每物理子步advance的不对称必须在Clean3副本修复；右瓶171/11入口不能冒充双手装配。
- 预注册外评v1关闭负gap无限接触、单端接触行程和末段失稳闩锁问题；保留旧历史成绩原定义，禁止直接当同口径比较。
- C3-P1总6小时：CPU150分钟、物理45分钟、PPO120分钟、评测交付45分钟。最多2次物理启动、训练启动1次加仅未更新前的支持失败重试1次、外评1次。256env/seed42/rollout16/minibatch4096，最多512轮=2,097,152转移；从零训练，前后各16episode外评，具体门槛及停止表见工作包。
- 当前H2S2R训练0、外评分母0、成功率N/A。新物理需本机完整三视角视频和总成本；原始轨迹/checkpoint可留远端HDD，附路径、索引和SHA。

以下是此前交接的归档上下文；其中“新reviewer尚未审阅”“收尾结束项目工作”“下一步等待首次裁定”等状态均已被以上接续裁定替代，历史实施者结果不改写。

更新时间：2026-09-12。本文件是 Clean3 新 reviewer 的上下文入口，不是物理/训练启动授权。

收尾状态：用户已明确要求本会话结束项目工作。已核对本文件、共享AGENTS职责条款及imple3的decision.json；后者仍为CPU_DATA_PREP_PASS_TRAINING_BLOCKED_PENDING_REVIEWER，training_ready=false，输入来源未获训练放行。此处核对是文档/状态记录核对，不是重新运行或独立签核CPU及物理结果。没有追加implementation任务、启动实验或改动其他线状态；后续由用户另开Clean3 reviewer会话接手。

## 职责纠正与优先级

用户明确纠正：产生本交接的会话负责 Clean3，不是左手杯穿模。此前助手声称本会话仅负责 LEFT/CUP 的回复作废。原因可定位到共享 AGENTS 中使用了跨会话不成立的“this reviewer conversation”；不能据此推断模型记忆内部故障。共享文件已改成按任务命名的分工。

- Clean3 reviewer（reviewer2）+ imple3：擦盘子 baseline 接入、方法裁定、训练计划及审阅。
- reviewer1 + 主 implementation：Pour17 右手瓶训练。
- 左杯 reviewer + imple2：Pour17 左手杯穿模。
- 不改、不重启、不重新规划其他线。共享修复只在独立副本提出，跨线集成需对应负责人接受。

## 用户目标与已定路线

用户知道原包属于 ours；明确要复用其中视频及训练该任务需要的数据，通过我们搭好的 H2S2R adapted baseline 训练并评测 Clean3。原包是 ours 不是阻塞，也不要再问是否重跑 ours。

路线：示范视频/兼容的重建任务数据 → H2S2R 输入及任务装配 → FABRICS + PPO → 外部任务评测。保持方法身份，允许必要且明示的适配；不承担完整原论文长周期复现。不得静默移植 ours residual policy、checkpoint、GraspPose前馈、奖励或 oracle 命令。示范得到的手关节表征与训练成功命令不是同一来源，必须逐字段区分。

用户希望尽快拿到训练结果，按影响训练结果的优先级处理问题。一个工作包包含多个相互依赖的步骤，完成后统一审阅；常规支持修复自主执行，明确阻塞/方法变更/时间上限则提前交付。不每修一个小函数就回问，也不把全项目无限打包。路线和放行由 reviewer 决定，Sol-high 负责实现和证据。Prompt 直接给可复制文字，不要求用户去打开 prompt 文件。

## 原始输入、实际位置

- 原包：`D:/UCBP/RL/Baselines/task_plate/clean3_base_package_20260909(1).tar.gz`
- ours 历史结果：`D:/UCBP/RL/Baselines/task_plate/clean3_results_20260909.tar.gz`
- 独立 staging：`D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_staging_20260912`
- 使用说明：上述 staging 的 `README.md`
- 实际转换/测试：`prepare_clean3.py`、`test_prepare_clean3.py`
- 解包来源：`extracted/base/`、`extracted/historical_results/`；精确文件及 SHA 查 `unpack_manifest.json`，不要重复解包。
- 原包内视频：`inputs/source_video/3.mp4` 和 `3.hdf5`；重建数据：`inputs/clean_tableware_3/`。
- 候选输入：`prepared/clean3_perception_candidate.npz`；逐字段来源：`prepared/provenance.json`；校准/ARKit留证：`prepared/source_calibration_and_arkit.npz`。
- 实际V12只读接口快照：staging 下 `interface_snapshot/`。以该快照和身份记录定位执行版本，不默认本地旧代码等同远端。
- 已知 V12 远端参考根：`/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912`，仅只读参考，不作为 Clean3 部署目录。Fabrics参考源码根 `/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src`。新会话必要时重新核验实际身份，不继承旧 SHA 即宣称现状。

## 当前已交付状态（不是 reviewer 全面签核）

独立证据边界：早期本reviewer直接查看过原包实际控制源码、归档成员及历史评测JSON，确认原控制是58维RL-Correction残差、存在原视频/HDF5，并核对历史三组1/16、9/16、16/16。最新接入包的854文件复核、22项测试、300帧加载/时间对应和数据质量计数来自实施者交付，尚未由reviewer逐项复验。下列细节按该来源理解，不应从“已保存交接”推断“训练已放行”。

实施者交付 `CPU_DATA_PREP_PASS / TRAINING_BLOCKED_PENDING_REVIEWER`，约20分钟，854文件复核、22项CPU测试通过。报告称原版V12 loader及参考构建器加载成功。本次收尾实际读取了 RESULTS、INTERFACE_GAPS 和 staging README；没有独立重跑测试、审计全部转换代码或验证所有原始数值。新 reviewer 必须追到这些代码和对应数据再作裁定。

300帧，视频1920×1080、30fps、10秒。四组位姿300×7、左右手300×22，角色为 object_0=左盘，object_1=右海绵。原 ours env.object/env.aux 名称与 Pour17相反，按语义映射，不能按变量名照搬。

候选保存30Hz；原replay/ref_qpos标15Hz。实施者以视频/HDF5/相机逐帧对齐为依据，未修改轨迹数值或平滑、填补、裁帧/展宽。时间口径尚需reviewer独立核实。

本线 GPU/Isaac/训练/评测均未启动，成功率未定义，不是0%。其他线是否运行不能据本线状态推断。

## 证据索引（先读这几项，再按问题深入）

目录：`D:/UCBP/reports/H2S2R_CLEAN3_PACKAGE_REVIEW_20260912/imple3_results`

- `RESULTS.md`：完成项、限制、可运行入口。
- `INTERFACE_GAPS.md`：八类具体缺口、函数/字段及最小后续动作。
- `validation.json`、`cpu_tests.json`、`decision.json`、`costs.json`、`delivery_manifest.json`：实际状态和证据身份。
- `source_quality.json`、`hdf5_schema.json`、staging `prepared/provenance.json`：数据质量和来源。
- `assets.json`：OBJ/USD/材质和几何。
- `evaluation_definition.json`、`historical_results_audit.json`：历史任务语义和成绩。
- 早期初审：`D:/UCBP/reports/H2S2R_CLEAN3_PACKAGE_REVIEW_20260912/REVIEW.md`。其90分钟CPU包已完成，不得作为下一次重复准备任务。

## 新 reviewer 要统一裁定的核心问题

1. 输入质量与时间：来源混合（设备相机/ARKit手、模型物体、盘类别尺度先验）；loader笼统MODEL_ESTIMATE不代表纯模型来源。两腕几乎静止，盘 rotation_usable=false；trustworthy187/126与旧summary192/153不同。判断是否真正阻止训练，不能只凭schema通过或警告数量裁定。
2. 几何：当前盘约18cm，旧replay采样点约24cm；不拿旧点作当前碰撞/评测网格，不猜乘0.75就修好姿态。检查实际使用路径后决定必要修复。
3. 初始化：Pour17 reset要求特定阶段及瓶杯字段；Clean3原来持物起步并用pin/release/GraspPose/认证。决定允许共享哪种静态初态、准备期及信息边界，不能伪造Pour17 reset过接口。
4. 双手装配：当前已用训练入口171obs/11action为右手、左固定；公共22维声明不等于双手训练已实现。reviewer明确左盘与右海绵如何控制，不将选择推给executor。
5. PCA：既有环境要求synergy.source包含当前输入SHA。按已接受来源和既有PCA流程决定拟合集合；不伪造SHA，不直接把Pour17 PCA称Clean3 PCA，不擅自扩大维数。
6. 奖励/参考/评测：原clean3_reference_v*有低通、常量盘姿、海绵重贴、IK和展宽，不是原视频估计。当前H2S2R为跟踪奖励，是否需要任务适配由reviewer明确决定；用户对有效抓持学习很重视，但未批准任意搬入ours奖励。

历史 ours 三组为1/16、9/16、16/16（base/gate/soft-relative版本），不能算H2S2R结果。原评测覆盖≥0.45、接触行程≥0.70m、盘倾角<15°、偏移<3cm、双物未掉，还有认证/死亡语义；signed gap<5mm没有负向下界，不能证明无深穿。统一外部评测与训练奖励分开裁定，不静默改历史成绩。

## 下一步与授权边界

新 reviewer 的第一项工作是独立审阅当前接入包并作有限工程/方法裁定，形成下一份可执行、多步骤工作包与内联prompt，目标指向独立Clean3训练和评测，不是重复90分钟输入准备。先查是否有本文件后更新的 imple3 结果。只补充影响决策的源码/数据检查；必须明确预算耗尽后的选择，不要求唯一根因或无限历史审计。

本交接允许读取本地/必要远端源码数据、轻量CPU验证和维护Clean3审阅文档；不自动授权GPU、Isaac、PPO、生产核心修改或替其他线决策。后续物理包需要明确启动数、步数、墙钟、资源条件和结果决策表；共享GPU不得干扰已有任务。

新物理实验交付本机可观看的完整三视角视频、简短结果与成本。用户已允许完整数值轨迹、PNG和checkpoint保留远端HDD，提供路径、索引和SHA便于reviewer访问；不要重新要求全部原始帧下载D盘。不能用回报增长、截断录像或无有效评测分母代替成功率。

本次交接没有启动新会话、关闭/归档会话或派发implementation；只保存上下文和提供用户复制用prompt。
