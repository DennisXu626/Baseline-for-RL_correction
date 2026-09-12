> **2026-09-13交接：历史已关闭工作包，不可执行。** 其1024恢复与后续256训练均已关闭；最新周期dump干预只验证初始化，未重开训练额度。新训练包待reviewer发布，见最新handoff。GPU_POLICY覆盖下方旧GPU限制。

# 右手训练结果工作包：左侧风险随首轮检查，PPO → 自主终评

> **GPU运营规则已于2026-09-12由用户替换：** [GPU_POLICY.md](../../../GPU_POLICY.md) 为双A6000的当前准入规则。GPU0/1均可选择，允许资源预算与持续监测支持的同卡/跨卡并行；已有compute进程不再单独阻止启动。通过准入后在本次子进程设置 `RL_ISAAC_NO_GUARD=1` 关闭旧机器级锁，记录实际GPU映射，不删除他人锁或中断他人任务。本文下方历史“独占/空闲卡/固定GPU”运营限制不再生效；历史运行事实、科学参数、track所有权及启动额度不变。

**FINAL：原1024工作包已关闭。** resume02最后一次启动122.233336s原生SIGSEGV，非初始化超时；两次额度2/2，累计609.315635s，0采样/更新/模型。禁止原样第3次启动，下方恢复授权均为历史。用户已批准并放行新的256env/4096 minibatch隔离适配，见 `D:/UCBP/reports/H2S2R_NATIVE_CRASH_REVIEW_20260912.md`；按该文件执行新工作包；下文1024恢复授权不可复用。

当前最终状态（2026-09-12 13:21:27UTC，imple1右手）：resume02 STOP_NATIVE_CRASH。原最后第2次启动已使用（2/2）；GPU1、1024env、seed42，进程122.233336s以SIGSEGV/-11退出，非超时。环境未返回，0采样/更新，无checkpoint/视频；左杯未观察，终评未运行，不能记成功率0%。累计训练进程609.315635s。21项CPU与32项部署身份PASS不等于物理PASS。停止本恢复，无第3次；完整证据见H2S2R_TRAINING_PILOT_20260912/right_training_result_01/resume02/RESULTS.md。以下历史放行/运行中记录已被本状态覆盖，不影响其他并行任务。

**当前恢复裁定：** H2S2R_1024_INITIALIZATION_REVIEW_20260912.md（外部/历史定位：`D:/UCBP/reports/H2S2R_1024_INITIALIZATION_REVIEW_20260912.md`）。resume01第1次训练启动在1024env的PhysX USD加载阶段被480s门终止，0采样/更新，旧FAIL保留。reviewer允许原最后第2次启动：仅train初始化1800s（eval仍480s），累计6h需扣旧487.082299s，增加20min支持准备/资源记录，无第3次或独立探针。不改1024/方法/物理；成功初始化直接同进程首轮→PPO→终评，失败则结束仅延时路线。用户最新交付仅要求本机视频与简报，完整raw/checkpoint/PNG可远端完整保留并提供路径/SHA，不再强制本地复制。细则优先于下文旧8min、恢复资格和本地全raw条款。reviewer未启动新运行。

最终执行状态（12:45:08UTC结束，2026-09-12）：right_training_result_01/resume01 / STOP_INITIALIZATION_TIMEOUT，覆盖下方INITIALIZING。新容量/26项CPU/32项身份门通过后实际启动1024env seed42 PPO入口，训练启动1/2；环境构造未返回，480秒硬截止、7.082秒额外清理，进程487.082秒。主栈initialize_physics，经SimulationContext.reset调用；不宣称唯一原因或OOM。0control/0优化/0epoch，无checkpoint、视频、raw或终评；左杯问题未观察，成功率未定义。无已定位并修好的支持故障，第二次不具资格，停止不重试。完整现有证据/成本/实际diff本机已交付：H2S2R_TRAINING_PILOT_20260912/right_training_result_01/resume01/RESULTS.md。旧存储阻塞与batch2/2保持。

当前执行（2026-09-12 12:37UTC）：right_training_result_01/resume01 / TRAIN01_INITIALIZING，覆盖旧存储阻塞。按新分层交付核算，本机78.89GB余量通过66.264GB总预留（含20%与10GB独立空闲），远端2.807TB通过500GB。新增26项CPU/部署32项身份通过，未重跑短验证/探针；已于12:37:01UTC在GPU1启动wrapper2463468，1024env/seed42真实PPO入口。累计6小时截止18:37:01UTC，初始化≤480秒；首轮左杯/右瓶签核在优化前，实际更新待证据。训练启动1/2已用，不能重复派发或重置预算。源码支持差异和本机结果在上述resume01；旧失败和旧batch2/2保留。

**用户最新交付偏好覆盖旧本机全数据要求：** 按 `D:/UCBP/reports/H2S2R_STORAGE_REVIEW_20260912/REVIEW.md` 顶部最新裁定，完整轨迹、逐帧PNG、checkpoint/normalizer和日志可留远端指定HDD，reviewer远端审阅；本机只需完整约定三视角视频、结果/成本与远端证据清单，必要原图按需取回。不得把下文旧本机完整raw/权重义务继续当容量门。远端完整证据保留不变。另：用户要求当前任务先执行完，再补监控接入，不改正在执行的科学任务。

**当前存储裁定（覆盖旧本机全PNG预留）：** `D:/UCBP/reports/H2S2R_STORAGE_REVIEW_20260912/REVIEW.md`。前次因D盘74.64GB不足以满足终评全PNG107.86GB单项预留而未启动。训练在远端，不要求本机保存全量PNG和完整归档重复副本；它们保留在远端指定HDD。本机仍必须交付完整约定数值轨迹、三视角视频、checkpoint、metadata与成本。按新传输布局核算全包容量并留余量，不能再用本地全PNG107.86GB门拒绝；也不能跳过数值轨迹/视频/临时峰值预算。只修支持层打包/传输与容量检查，不删历史、不改采样/录像内容、不增实验额度。reviewer仅做本地盘点与文档裁定，未启动或清理数据。

执行状态（2026-09-12，覆盖下方初始RELEASED状态）：right_training_result_01 / BLOCKED_INFRA_LOCAL_STORAGE，启动前提前交付。单次快照GPU1可用，但D盘余量74.64GB；仅既有16×903×3视角960×720原始帧按不假设压缩+20%需107.86GB，尚未计训练raw/左右视频/checkpoint，容量门失败。未修改/部署支持，未重跑CPU或任何探针，未启动物理/PPO/终评。训练启动0/2、更新型运行0/1、终评0/1，旧短2/2及全部成本保持。左杯问题本轮未观察，右手未学习，无新视频/checkpoint。证据：H2S2R_TRAINING_PILOT_20260912/right_training_result_01/RESULTS.md。停止，不轮询；恢复需足够本机容量、完整schema容量预留与新资源/身份检查，不把单项33.22GB缺口当全包需求。

状态：REVIEWER_RELEASED_FOR_USER_DISPATCH。用户转发本计划的执行prompt即执行。只由implementation运行；本reviewer尚未启动。

## Goal / 决策用途

取得修正版H2S2R右手抓瓶的真实训练量、last checkpoint及自主评测，判断在既定方法与预算下是否出现抓举学习进展，供reviewer决定继续该方法、明确奖励适配或转task。顺带明确用户报告的左手抓杯问题在当前右手共享场景中是否构成物理阻塞，不能再答非所问。

不新增独立短验证/初始化/录像探针，不再要求成功源回放或随机策略先抓起来。前包已完成，成本2/2不重置。

## 方法决定 / Non-goals

本包保留当前reward.py、PPO/LSTM、5D任务/FABRICS、11维动作、phase1_g2/start14/estimated、seed42、hold0、右zero-margin与1/12/12、左固定命令、原物理/资产/黑桌。不开reward adaptation、PCA扩维、oracle、人工抬腕、左侧policy或完整双手pour训练。

当前奖励有瓶轨迹跟踪和手指距离门，没有额外抓取shaping；reviewer选择先得到此方法一个有效训练结果。后续奖励适配由reviewer决定，executor不得自行挑算法/权重。左侧动态抓杯没有本包认证，不把固定左侧检查称为全面修复。

## Root cause / evidence confidence

读取 `D:/UCBP/reports/H2S2R_TRAINING_RELEASE_REVIEW_20260912.md`。右侧实际限位/调度/hold/reset及可见接触已有通过证据；无需重做独立诊断。相机随瓶掌中点移动已被源码+256帧元数据确认，是画面晃动的直接机制，非训练阻塞。旧轨迹没有杯状态/左接触，左侧历史穿模仍未知。瓶杯在同一物理场景，不能从右policy不观测杯推出物理隔离。

## Files involved / 实际版本

ROOT `/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912`；Python `/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python`；bundle `/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17`。

当前32项身份清单：`D:/UCBP/reports/H2S2R_TRAINING_PILOT_20260912/batch_delivery_01/expected_identity_after.json`。支持源码以同包source_after为准；核心源码沿当前已核验V12，不能用本地原始仓库。

允许支持修改（ROOT相对）：
- `tasks/h2s2r_pour17/training_pilot_audit.py`：左侧/杯只读记录、训练首轮检查、奖励分项/训练进展与保存；不改变物理调用。
- `runtime/training_pilot_20260912/camera_delivery.py`：增加左杯/双手视野的观测支持。当前跟随镜头可保留；若≤10分钟可完成每episode固定机位并CPU验证则可做，否则明确搁置，不单独开实验。
- `tasks/h2s2r_pour17/train_right_lstm.py`：首rollout签核及停止/记录支持，禁止改PPO更新实现。
- `runtime/training_pilot_20260912/endpoint_eval.py`、`resume_observation.py`、`launch_with_root.py`：对应记录/数据schema、加载/保存和异常处理支持。

right_env/env、controller、补丁、限位验证、reward.py/cfg、物理/PCA/资产核心冻结；改支持更新SHA与diff。新本机根 `D:/UCBP/reports/H2S2R_TRAINING_PILOT_20260912/right_training_result_01`，远端根 `/media/msc-auto/HDD/users/kailang/h2s2r_training_pilot_20260912/right_training_result_01`。

## Required changes / 包内依赖步骤

### 1. 合并准备，最多45分钟，目标是训练

复用已通过的录像/启动/验收，不重跑整套历史测试。补充左杯数据：使用已存在 `_object_poses()[0]`、`_wrist(ControlledSide.LEFT)`、`aux.data.root_lin_vel_w/root_ang_vel_w`（先核实实际API；若速度读接口不同，记录pose差分及时间，勿猜字段）、现有q/qd/targets的左侧索引。接触只读既有左侧传感器，必须核对其filter对象，缺失/不覆盖就写明，不能用右力冒充左力；不增加新的物理传感器或碰撞配置来过门。

给env0左手/杯至少一组三视角近景，首轮与每100epoch留下同步左杯证据；右侧既有完整三视角不减少。明确env/episode/reset前后对应。记录必要时机器人根位姿用于区分相机运动，不做额外动力学试验。

原始schema增加cup pose/left palm与可取得的左接触、速度；遇reset时一致保留终止前状态或明确另列post-reset，不能混合。记录原奖励tracking值/close门、瓶误差、reference进度、右tip距离、动作分布和边界饱和、瓶/掌高度、左实际q与固定目标差。它们都是只读诊断，不改reward返回值。

CPU验证新增字段、左右索引、reset语义、读取不新增step/reset/RNG、首rollout门在optimizer前、失败保存与checkpoint可定位；不逐小函数请求批准。源码明确问题能在支持范围内修就一次处理；明确核心/方法变化才提前交付。

### 2. 启动唯一学习运行，左侧风险检查嵌入同一进程

使用已有真实训练入口和launcher、1024env/seed42从零，不先开16env诊断。部署/输入身份与资源存储核验通过后启动。应用、env、policy/reset阶段延用记录及初始化8分钟上限。

初始化预览和训练首个horizon16采样时，执行agent直接看左右画面、cup运动和已有碰撞几何：主要查用户所指的左手/杯实质深穿、初态非法覆盖、杯异常弹射/抖动、与瓶发生非物理支撑/碰撞。不能只报告右侧筛查。必要几何核对限于当前对象/当前帧，不做全资产历史审计。

首个rollout尚未返回PPO优化前，复用已有control16等待点签核；核实原finish_epoch检查在super.train_epoch后，不能把事后检查当作optimizer前阻止。只移动/调用支持检查到现有门，保持rollout/actions/reward/RNG不变。首次签核针对当前启动参数、媒体和左杯记录，不等待用户在线。

**准入分支**：
- 未见当前场景明显深穿/非法状态，右链通过，画面数据可审阅：继续同一进程PPO，不再回问。记录左侧仅有限初态/固定保持筛查，不能称历史动态抓杯问题修复。
- 已确认左侧实质深穿，或对共享场景物理有效性有未解决的重大疑问：停止更新并交付左右视频与异常数据。不能因为右policy没读cup就忽略；不移杯/改碰撞/冻结物品来绕过。
- 只存在跟随镜头晃动、轻微渲染壳差或缺少全环境深度证明，当前接触可判断且无实质矛盾：记录限制，允许训练。不可证明所有未来帧无穿透不是阻塞。

首轮额外人工执行agent判断≤15分钟，计入6小时；明确阻塞尽早交付，不能为用完时间而调查。训练期间首轮与每100epoch同样看左右实际片段，出现实质物理错误停止。

### 3. 持续学习并交付checkpoint

保持官方LSTM/PPO、horizon16、minibatch16384、mini_epochs4及网络/优化配置；1500epoch=24,576,000训练transitions或累计6小时先到。保留原2,000,000 max_epochs学习率日程，原独立停止钩子，留5分钟保存last。每100epoch checkpoint+normalizer+真实视频+原始块；不能用best替代last。

首个实际optimizer更新、首checkpoint、预算结束分别报告重要进度，不要求用户批准。不能只报告init/预览PASS。记录实际更新数、参数确实变化、训练量、奖励分项/接触/瓶高进度，确认没有隐性一直暂停或只采样未优化。

没有抓举进展、回报平稳或动作饱和是本包要观察的学习结果，不据此提前调奖励。NaN、非法写入、PPO接口错误、严重穿透停止；OOM不自行缩批或换算法。

### 4. 同工作包终点评测，形成reviewer决策依据

完成预算内有效训练后自动加载last有效checkpoint，固定normalizer、deterministic、16env逐episode seeds1000–1015、每episode≤903controls。左固定，无oracle。沿原辅助短时抓举定义（瓶/掌升≥10mm、连续5帧相对位移漂移≤5mm、首次窗前倾角≤0.4rad且瓶未下落超5mm，视频排除抛起/支撑/穿模）。不是完整双手Pour17成功率。

保持每episode三视角与全轨迹，报告有效N/成功数/无效原因，不合并失败与有效0%。训练中断/无完整checkpoint则不把它当完成训练的终评，交付已有结果。有效0/16照实交付；不补seed、重训或临时增加抓取reward。

## Tests / 资源 / 次数 / 预算

- CPU合并支持准备与修复累计≤45分钟；只测新增/受影响检查，不重新做完整初始化探针。
- 训练最多2次进程启动，但只允许1次进入optimizer更新；第2次仅用于第1次更新前已定位且修好的支持失败（路径、保存、媒体、只读schema），非OOM/物理/算法故障；累计墙钟≤6小时，不重置预算。更新后不重启/续训。
- 独立短验证0次。训练首轮1024×16数据属于真实rollout，预更新失败的采样单列，不隐藏。
- 终评1次、≤30分钟，最多16×903控制步；无有效性问题不得补跑。训练/终评初始化各≤8分钟，包含各阶段上限。
- 最终交付≤30分钟额外活跃时间，总≤7小时45分钟（45分钟+6小时+30分钟+30分钟）。清理额外成本单列。
- GPU0/1按D:/UCBP/GPU_POLICY.md共享准入；容量/负载不足时保留准备提前交付，不无限轮询。先按新增raw schema最大transitions估算本机与HDD容量+20%，持续分块回传；不往home写大文件，不以压缩假设或缺媒体换吞吐。

## 结果 → reviewer下一决策 / 停止条件

| 结果 | 本包必须交付 | reviewer后续用途 |
|---|---|---|
| 左cup当前实质深穿/共享场景污染 | 首异常左右三视角、cup/left/bottle状态、实际碰撞资产与原因范围，训练未启动或无效的准确计数 | 优先修确证物理问题；不再把右侧泛化验证当答案 |
| 首轮通过并开始更新 | 真实训练量、即时短视频，继续预算内训练与终评 | 不再等待逐项放行 |
| 有效训练及0/16，无/弱抓取进展 | 全部视频+checkpoint+reward/action证据，报告本预算失败 | reviewer决定显式grasp-reward-adapted变体或转task，不宣称方法不可能 |
| 有效训练及>0/16 | 实际自主抓举证据和完整分母/限制 | 决定训练扩展/左侧任务包，不自动双手训练 |
| 支持预算耗尽、更新后运行异常、训练/评测到时限 | 已有产物+失败证据+成本，提前交付 | 明确恢复或暂停决定，不自行新建第三次恢复campaign |

## Deliverables

RESULTS先分别回答：左手杯穿模是否观察到、此次有限检查范围；右手是否真正训练、更新/epoch/transition/时长；是否自主抓举及有效分母；晃动是跟随镜头并是否影响判读。不得只列S/P编号和PASS。

本机真实左右三视角、原始轨迹、source/input SHA、diff、CPU测试、first_rollout检查、checkpoint/normalizer、reward/action进度、逐episode评测、costs/attempt_registry/FILES齐全；最终回复直接内嵌视频而非只贴RESULTS。画面失败照样展示，离线替身不冒充物理。

完成后统一交reviewer，不自行追加训练或奖励适配。本计划放行有限右手训练，不宣称当前baseline已经解决左侧历史穿模或具备完整双手能力。

