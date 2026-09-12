# ViViDex 双手 Pour17：截止日期驱动的执行计划

> **GPU运营规则已于2026-09-12由用户替换：** [GPU_POLICY.md](../../../GPU_POLICY.md) 为双A6000的当前准入规则。GPU0/1均可选择，允许资源预算与持续监测支持的同卡/跨卡并行；已有compute进程不再单独阻止启动。通过准入后在本次子进程设置 `RL_ISAAC_NO_GUARD=1` 关闭旧机器级锁，记录实际GPU映射，不删除他人锁或中断他人任务。本文下方历史“独占/空闲卡/固定GPU”运营限制不再生效；历史运行事实、科学参数、track所有权及启动额度不变。


交接更新：2026-09-13 01:38:28 Asia/Shanghai。**唯一当前执行状态为H顶部：DEV_TO_FIRST_CURRICULUM_V1已批准且运行中。请先读README与REPRODUCTION_STATUS当前快照；接手agent须向Dennis索取训练结束的最终结果。H以下旧入口及本页前部WP1/WP2计划均为历史，不得据其重新启动。F保留已批准W1/S1及训练机制，A–D仅保留仍适用的输入/公平性约束。**
WP1 已在隔离目录执行：完整双手 reference、真实 Isaac 1/32-env 控制、短 PPO 更新、保存重载、登记 evaluator 两回合与长时域吞吐测量。原版 MuJoCo 源码和实验合同未修改。
最新用户决定：**D1、D2非world部分、D3联合训练，以及本次F节的W1共同放置和S1双手Pour合成均已批准。比较目标是human-video → RL correction最终能否学会任务，视觉policy阶段明确排除。** 轨迹增强/curriculum等影响状态RL表现的机制必须尽力保留，不得视为无影响或因截止时间自动关闭后替代主结果。W1固定新工程约定，不认证历史world或当前腕轨迹可达；不得重复请求已批准事项。

**历史WP1评审（后续已执行，非当前入口）：接收开发闭环，不接收“训练机制已忠实移植/正式训练就绪”。下一任务是F中的WP2训练就绪修复包；不是再补一轮局部审计。** WP1 的固定 stage-0 evaluator 为 0/2（均 903-step timeout），只说明开发策略未完成任务；不按该结果选择 mean、seed 或参数。两个 evaluator 回合未录视频；另一路 open-loop MP4 及启用相机后的原样重试均为全黑，不能作为视觉证据。显式camera-sensor/render接线已纳入下一包的既有录像交付范围，不能因此继续停在“需要新接口”的文字报告。用户已明确回复“批准W1和S1，按文中规则实现”；方法待批项已解除，正式长跑仍待WP2实际验收。

用户已确认：截止时间为 **2026-09-15 14:00 America/Los_Angeles（PDT）= 2026-09-15 21:00 UTC = 2026-09-16 05:00 北京时间**。
用户最新收敛范围：**当前只做 Pour17；sweep、take off bottle cup、clean plate 暂缓**。Pour17是整个baseline pipeline的首个训练验证任务；参考生成、双手控制、PPO、保存/重载和评测接线必须可复用，后续任务通过输入/场景/判据配置接入。最终四任务各自训练、success rate 和 policy 的项目目标保留，但本次不排另外三项的工期，不开发它们的资产或判据。
资源可随时排队，前提是不干扰他人；这不是共享占用中的 GPU、终止其他作业或本轮启动训练的授权。

## A. 独立判断与路线取舍

**最短可信路线是复用已有 Isaac 场景基础设施，在隔离目录接入 ViViDex 的 MANO reference、双手状态策略与轨迹奖励；先做真实闭环，再决定正式预算。**
目前不能承诺在两三天内完成完整 ViViDex，更不能承诺四任务稳定训练。已有单手 MuJoCo policy 不能迁移权重后直接控制双 Sharpa。
把双臂、双手、物体与接触重新搬到 MuJoCo，或安装另一套 SAPIEN，都比复用已存在的 Isaac 世界更不确定，均不作为截止日前主线。

保留：原版训练/eval 证据；已验证 decoder、左 FK、NLopt、固定 A、完整 142 帧候选和 mesh；原始 MANO→机器人十点目标；公平输入隔离；最终 checkpoint 评测与全部失败记录。
停止：无新矛盾时重复 decoder/FK/梯度/A 代数验证、零误差追求、逐点残差归因、两种 mean 的无限历史追溯、原版第二 seed 重启、SAPIEN/Place 修复和另外三任务开发。用户已明确排除用于视觉蒸馏的成功轨迹数据集收集、点云policy、BC/diffusion；仍须保存policy、完整任务评测记录与验证视频。此范围选择不授权删减状态RL的轨迹奖励、增强或curriculum。
改变：不再要求旧正式 M1-A 获得抽象“质量 PASS”后才接物理系统。获批的新工程约定用于独立适配版本，旧 M1-A 仍保持 NOT PASSED，不能倒填通过。

### 决策所需证据与可复用边界

| 只读抽查 | 结论及其用途 |
|---|---|
| `verification/seed_0_fixed_eval.json`、远端 campaign manifest/checkpoint | 60,002,304 transitions、300 episodes、0.8469444444 平均颗粒比例；manifest 仍为 `stopped_after_seed0`，checkpoint 仍存在。只是原版单手证据。 |
| `artifacts/m1a_full_sequence_candidate_diagnostic_v1/*arrays_v1.npz` | 两分支、142 帧、22 joints/10 landmarks 确实存在；复用左手已算结果，不重新从零诊断。 |
| H2S2R 的 `tasks/h2s2r_pour17/env.py`、`cfg.py` | 有双臂双手场景、接触传感器、名字映射、canonical reset 路径，可有选择地移植。默认却是 `phase1_g2`、Ours reference-derived reset、15 步 clamp，绝不能整类继承当 ViViDex。 |
| H2S2R `pour17/trajectory.py`、`vector_eval.py` | 前者对两物体分别 SE(3) 对齐；后者预置 G1/G2，且批量 seed/reset 不能从标签推断独立性。不能照搬其 reference 或正式分数。 |
| 官方 `rewards.py`、`control.py`、`tools/train.py`、`algos/policies.py` | 可按 pinned source 移植 reward、MLP/PPO、action bias 的结构；不能直接执行在 Isaac articulation 上。源码 reward 的具体范数/条件是本适配来源，不擅自按论文公式“修正”。 |
| 登记 tar 中 evaluator/reset 与本地 `_inspection` 比较 | 登记 `run_eval.py` SHA `40e6b157…` 包含 G2 callback、903 步 cap、warmup=0；inspection 的同名文件较旧，且缺 canonical reset。**所有 world/reset/evaluator 必须从登记归档读取或核对同哈希远端文件**，不能使用 inspection 文件夹整体替代归档。 |
| canonical reset 内容 | 无 reset 随机化；机器人 t0 是中立返航姿态且等于 G4 返航目标；并非 G2 抓握状态。512 次固定初态重放不能宣称为 512 个不同场景或泛化试验。 |
| 原始两物体首帧与canonical reset的直接6D配准 | 两物体分别要求的世界旋转相差 **47.8998°**；这是完整SO(3)数值冲突，没有区分绕物体对称轴旋转、长轴倾斜与物体局部系差异。**该数值本身不能作为任务/集成阻塞**，不要求把两物体完整四元数都配到reset。只保留对中心、口部/长轴方向和手物几何有实际影响的检查。 |

没有重新审查全部 artifacts，未修复根目录 Git。隔离适配用逐文件 SHA-256、来源和 patch 记录身份；Git 不可用时明确 unavailable。
公开方法边界参考 [ViViDex 论文 §III](https://arxiv.org/html/2404.15709v3#S3)；本地 pinned code 优先定义所移植的数值行为。

### 实时资源快照（只代表检查时点）

2026-09-11 17:29–17:33 UTC / 北京时间 09-12 01:29–01:33：

- 同一主机 `kailang@128.32.164.89` 可连接。A6000 GPU0：5520/49140 MiB、87%；GPU1：4838/49140 MiB、40%。两卡都有其他用户 Isaac 进程；**当前不能记为两张空闲训练卡**。未终止任何进程。
- `/home` 所在系统盘约余 **6.3 GB**，数据盘 `/media/msc-auto/HDD` 约余 **2.8 TB**；不是账户独占配额。新源码副本、venv、日志、TMP/cache/output 全放数据盘，不在根分区安装大依赖。
- 已安装 Isaac Python 可执行并读取包元数据：Python 3.11 环境 `/home/kailang/.local/miniconda3/envs/rl-correction-pour`；Torch 2.7.0+cu128、Isaac Sim 5.1.0.0、IsaacLab 0.54.2、Gymnasium 1.2.1、NumPy 1.26.0、NLopt 2.7.1；**没有 SB3**。这不是本轮 Kit 启动/吞吐量验证。
- 远端 `/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter`、`pour17_baseline_bundle_20260829/pour17`、`/home/kailang/opt/IsaacLab-5.1` 均存在。另一个 `rl_correction_step4_lfs/tasks` 本次没有 Pour 任务，不能按名称当成正式 Ours runtime。
- 本机 RTX 5080 **Laptop**，16303 MiB；检查时占用2456 MiB、15%；D盘约余70.2 GiB。可先做 reference、CPU 检查和审阅；本轮未证实本机能运行 Isaac，不把它计入物理训练卡数。

本轮 SSH 受限通道失败后，经获准的只读执行通道成功连接；不是凭旧 inventory 推断资源。

### 未知是否阻塞

| 项目 | 处理 |
|---|---|
| 历史 HaWoR mean/model 语义 | 不再追溯；D2 冻结为标注清楚的新输入约定，不冒充恢复历史。 |
| 十点近似、固定 A 的跨机构对应、不可为零的 residual | 复用，保留误差；不是运行前必须消除的缺陷。 |
| 右手来源/名字、世界变换、腕根组合、双臂 IK、运行时接触与动作接口 | 真实阻塞，WP1 一次集成并测量，不拆成新一串审计。 |
| Ours 历史 runtime 不明 | 阻塞历史公平比较；D1 可建立新的比较版本。无需因此阻止已标注开发运行，但正式对比必须双方重评。 |
| 当前 GPU 排期、Isaac+PPO 接口、吞吐量 | WP1 必测；没有卡就报告排队等待，不把 CPU/运动学输出称作物理闭环。 |
| evaluator 的 G2 控制、固定初态、多次重放含义 | D1 明示处理；不能用高 reward 或预置 G1/G2 替代完整任务成功。 |

## B. 投稿目标、备用目标与可支持结论

**推荐目标：Pour17 上可复现的双手 ViViDex 状态 RL 适配实验。** 交付最终可加载policy、逐episode外部评测、训练预算/耗时和失败分布；WP1容量评审后本次只安排正式seed0，不支持多seed稳定性结论，预算见F。
工程可训练与任务学会分开验收：WP1的PPO更新只证明前者。只有冻结最终policy在完整外部任务评测中产生成功证据，才可写“Pour17训练成功/该pipeline的首任务已验证”；有checkpoint但SR=0不满足这一表述。成功比例照实报告，不事后发明“足够成功”的阈值。
与 Ours 比较只在同一新世界、reset、evaluator 上重评后成立。若 Ours policy 的训练世界不匹配，必须重训或将该比较标成另一个实验；仅重评不能掩盖训练设置不同。
主表名称须写明 `ViViDex state-RL, bimanual adaptation`，并列出 simulator、PPO 版本和 reference 工程约定。
主结果应尽力保留ViViDex状态RL训练机制。仅固定场景的开发版本不能自动升级为正式baseline，不能用它独立推出完整方法或广泛泛化能力差于Ours。

**F1：固定场景开发诊断/可另行批准的消融，不是自动投稿备用。** 可用于首包验证物理闭环与测速度；若缺少trajectory augmentation/curriculum，不计主baseline结果。若截止前确实无法补齐，reviewer须给出缺失机制、实现障碍、预计影响与剩余成本，再决定是否作为明确标注的有限消融报告；不得自动省略、将其当无影响、或只挑它的较好分数替代主线。

**备用 F0：原版复现证据 + 双手适配完成度/失败分析。** 若不能取得有效 Pour17 policy/eval，只能报告原版 0.846944 平均颗粒比例，以及适配尚未完成的具体阻塞。Pour17 success rate 填 `not evaluated`，不是 0；不能用单手结果填双手 baseline 行。
只有同协议真实跑完而失败的 episode 才记失败；程序崩溃、缺模型、未完成评测分别标 execution error / incomplete。

用户已明确比较层级：不同human-video → RL correction方法最后是否学会任务。当前交付为每任务状态policy及完整任务success rate，先验证Pour17；视觉student及其训练数据集收集从本实验范围排除，不再作为待讨论或后续必做阶段。两方法的状态观测权限与外部任务合同仍需明确列出。

## C. 工时、关键路径与硬截止

以下是工程估计，不是完成承诺。规划锚点 **T0=北京时间 2026-09-12 02:00**；实际转达或批准晚于此时，消耗余量，**不顺延投稿截止**。
从资源检查时约余99小时；前48–72小时用于集成、训练和早期评测，最后24–30小时保护结果与写作。

| 工作包 | 目标、输入与文件范围 | 产物 / 可直接验证的完成条件 | 估计与依赖、停止条件 |
|---|---|---|---|
| WP1：双手闭环 + 首次测量 | D1–D3 后；原始MANO、登记bundle、现有左候选、Isaac场景基础设施；仅 `ViViDex/adaptation/pour17_v1/`、新配置/产物 | **开发里程碑已交付**：双手 reference、58维物理动作、真实 PPO 更新、checkpoint 保存重载、外部 evaluator smoke、3个吞吐窗口和长时域测量。证据见 `artifacts/pour17_v1/wp1_20260912_02/`。 | 8,192-step 必需 checkpoint PASS；额外测量累计至57,344步。任务结果0/2，不是正式训练或任务成功。 |
| WP2：训练就绪修复与版本冻结 | WP1真实代码/产物；同一隔离目录；详见F | 可见的实际执行、接触正反例、参考/训练目标修复、源机制差异处理、同步增强与自动curriculum、批量评测测量 | **8–10工程小时修订估计**；原4–6h只覆盖已自报机制缺口，没有包含本次发现的训练语义差异和评测成本。2h/4h检查有具体目的；详见F的硬截止。 |
| WP3：持续训练与唯一维护通道 | 通过WP2验收的冻结版本、已安排的GPU时段；`outputs/pour17_v1/<run_id>/` | 本次研究预算选择为一个正式seed0、3,002,368 transitions、最后完整更新的policy；见F，不恢复开发checkpoint | 按WP1保守速度仅训练约26.82h，另加增强、内部验证与恢复开销；尚非新版本实测时长。目标09-13 02:00开始，最迟09-13 14:00。启动资格仍须WP2通过，不自动切F1。 |
| WP4：独立评测与PD对照 | 批量评测接线前移WP2；与正式训练并行整理；开发结果标smoke | 保持512 episodes和登记逐步判据；独立slot状态/G2控制、实际reset、逐episode结果；同reference PD对照另列 | 当前scalar已经复用Kit，不能再把“复用Kit”当主要提速方案。512回合串行约40.20h，必须测批量执行并校验同trace判定等价，或重新裁决共同评测预算，不能私自减少回合。 |
| WP5：论文结果封存 | 所有run manifest、模型、逐episode评测；只整理结果与文档 | 每seed原始结果、限制、版本表、未筛选示例视频、policy加载说明；不丢失败seed/低分分支 | **6–8工程小时**，可提前写方法/限制。最后至少12小时只用于结果核对、图表和稿件收尾。 |

### 时间界线（均北京时间）

- **09-12 04:00（T+2）**：世界输入/资产/GPU通道的实质阻塞检查；找不到历史 Ours 不继续翻旧档，执行已批准的新合同路径。世界变换或资产真缺失则给具体失败证据，不能以小审计续命。
- **09-12 12:00–14:00（T+10–12）**：WP1应有真实 PPO 更新或明确的闭环阻塞。到14:00仍无物理闭环，只允许最后4小时集中解决一个已定位的接口问题；禁止扩方法。
- **09-12 20:00（T+18）**：主线机制里程碑。通过者冻结并持续训练；尚有缺口者提交具体机制/障碍/最短修复时间，集中决定修复或降低投稿结论。撤销“到时自动关闭augmentation/curriculum并切F1”的安排；开发短测仍受WP1上限约束。
- **09-13 02:00（T+24）**：目标已有一个冻结版本的持续训练run。**09-13 14:00（T+36）为最迟训练启动线**；此后仍无有效主线run，降为F0或另行明确批准的有限消融，不编造完整baseline结果。
- **09-14 02:00（T+48）**：按已预登记中间开发检查判断学习、完整任务覆盖与剩余算力；正常低分不换seed/mean。只有出现动作失效、reward断线等实现证据才修复，不能用低SR单独认定bug。
- **09-14 23:00（D−30h）**：停止方法/输入/任务定义改动，冻结结果候选版本。能提前完成的seed立即正式评测。
- **09-15 05:00（D−24h）**：所有拟用于投稿的训练结束或按预先墙钟cap保存最后完整PPO更新；最终测试开始/继续。未达到原步数目标明确记 partial-budget，不改写成完成原预算。
- **09-15 17:00（D−12h）**：数值、模型和结果表封存；剩余12小时留给稿件核对与提交。测得评测更慢时，相应提前训练停止，不侵占这个保留窗。

主串行依赖：可用场景+允许输入 → 双手reference/实际control → PPO闭环与训练机制验收 → 冻结训练 → 最终policy评测。增强/curriculum移植在WP1同步推进，不能等闭环完成后才开始研究。
CPU reference/文档/评测接口可以在GPU排队时推进；已冻结seed0训练期间，另一空闲卡可训练seed1或做独立对照。**不申请新agent；这是工作依赖的并行安排，不假定另有工程人员。** 单implementation情况下优先关键路径；并行主要来自无人值守训练。

### 吞吐量与预算决策规则

原版MuJoCo历史约600 transitions/s、60M约28h，只适用于旧场景，**不外推到双手Isaac**。WP1 三个完整 32-env PPO 窗口实测 79.78、38.87、79.88 global transitions/s；按既定 `0.8 × min` 得保守 `r=31.10` transitions/s。对应 1M/5M/10M/60M 仅训练外推约 8.93/44.66/89.32/535.95 小时，尚未加入正式在线验证、恢复与评测预算。两回合 scalar evaluator 共565.3秒，串行512回合点估计约40.20小时且不含已观察到的结束清理不确定性。完整数据见[throughput.json](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/throughput.json`）。
WP1分别记录 Kit启动/首reset、reference/IK耗时、纯sim steps/s、含PPO updates的**global environment transitions/s**、eval episodes/s（含reset/加载）、峰值显存/系统内存和输出增长。
32 env起测，4096 global transitions/rollout；先至少两次真实更新并保存重载，再用3个约3分钟稳态窗口测端到端速度。
额外只测128 env这一档（可用资源允许时），仍4096全局rollout；它改变每环境horizon，必须记入正式配置，不能当纯速度优化隐瞒。
至少一次接近完整horizon的执行计时，防止只测早期秒退episode。测试环境时间和训练时间分列。

正式预算在看测试分数前冻结：令保守速度 `r = 0.8 × min(稳态窗口速度)`，先按完整episode/reset额外开销修正；单run计划用时 `B/r + 已测启动及在线验证开销`。
以可用GPU时段和 D−24h 为上界，预留一次恢复空间；本次WP1评审已提前选择**只计划一个正式seed0、B=3,002,368**，见F。不是运行后挑最好seed；不授权第二seed或预算搜索。
不机械照搬60M；已提交外推表。B属于本次reviewer的容量决策，WP2必须补新版本实际开销才能放行，预算不足时提前报告或按既定墙钟截止记partial-budget。
预算/中途保存按完整4096步rollout计，列实际步数；固定每约1M transitions保存（取完整更新边界），独立测试只对最终模型进行。
最多每4小时一次开发状态检查；不做自动超参数/mean分支搜索。测试seeds不能用于调参或选checkpoint。

## D. 用户决定与未决项

### D1：使用新的共同版本，解除历史 runtime 依赖 — 已批准

用户已批准：将登记bundle的世界、资产、中立t0 reset、G1/G3/Placed/G4/D1–D8与903步时限作为**新共同benchmark基础**；Ours同版本重评，旧分数不沿用。双方eval禁RSI、G1/G2预置、物体clamp，deterministic mean，512次固定初态、seeds `20260829..20261340`。
中立t0 q与G4返航目标是明确公布的共同场景约定；除此以外，Ours robot reference/抓握q不可进入ViViDex训练。
G2需一项明确的新约定：以**认证触发时锁存的实测双腕位姿**为起点，用同一公共认证层做世界+Z15mm、姿态保持、8/5/8步骤；原判据不变。不得读Ours `cert_arm7`或母带交互行腕/臂q。
这改变归档中“交互首行抓握站位”锚点，**已获用户批准，实施时须作为版本overlay记录并供双方使用**；不能称旧合同未变。本轮没有改evaluator。
原始`progress.py`/`run_eval.py`保留；G2的env hook与共同wrapper记录独立hash。不能把certification助力称作纯policy自主执行，policy视频与认证段清楚标记。
canonical reset没有随机化，故该指标是**固定场景协议完成率**；其G3是倾倒几何条件，不是液体入杯率。不得与原版颗粒比例混列或声称场景泛化；若轨迹/初态完全相同，不使用独立Bernoulli假设制造很窄CI。
备选：用户给实际Ours run与同合同证据后采用历史合同；限时2小时，未有材料时不由implementation继续查历史。

### D2：冻结新MANO/几何输入约定 — 已批准，world按F/W1落实

用户已批准本节的mean、手部几何、时间/有效帧及prefix/suffix约定；**world alignment具体方案现已按F/W1获批冻结**。用户要求没有实际重要影响的杯瓶旋转差异不应继续成为阻塞。

推荐主分支 `proposed_raw_pose`（不额外加hands_mean），理由是最少附加未证实的解码运算，**不是误差/视觉/任务分数更优**。它仍是假定，不认证历史HaWoR。`+mean`候选保留为敏感性材料，不能用它的得分替换主分支；如额外训练，必须两分支完整列出。
保留v4左shapedirs处理、wrist-zero局部系、现有十点/拇指候选、固定A、米制原始尺寸、NLopt与alpha=.004。右手用真实MANO_RIGHT和同一构造规则，不能反射左手q冒充右手。
输入为142帧原始MANO与两物体pose；派生时间 `frame_id/15 s`，20Hz时位置/关节线性、旋转SLERP。所有有限物体pose原样参与，valid mask仅报告；不按mask删帧、填补、confidence加权。非有限数据报错。
**world alignment讨论修正（已按F/W1决定）：** reviewer接受不追求完整6D姿态对齐；此前把47.8998°差异直接强调成重大集成阻塞过强。杯瓶若近似轴对称，绕自身长轴的转角可能不改变外形和任务；它们并非任意旋转都等价，改变有向口部/长轴的倾斜仍影响倾倒。差异的有限功能检查现见F，不能凭完整四元数差异判断有害，也不能据此引入新的全面审计。
推荐将检查并入WP1原定场景回放，只确认物体中心/桌面高度、实际口部/长轴朝向与手物相对位置；不以四元数数值或轴向自转匹配为通过条件。轴向无关差异仅记录。若出现横躺/倒置、口部方向错误或手物分离等实际问题才阻塞相关运行。
另一个代码事实：原版`ObjectMimic`在`rewards.py:223,271`使用完整旋转距离，不自动消除物体对称性。物理上可忽略的自转未必在reward上无影响；优先保持一致的固定pose/mesh坐标约定，不静默删除旋转项或增加symmetry-aware reward。需要此类reward改动时属于单独方法决定。
以下中心+重力变换现已获F/W1的明确批准；不再是方法待批项：
采用**重力保持的共同世界变换**：由两物体首帧中心连线确定唯一yaw，以两中心均值定translation，尺度1；将同一个变换施于全部原始物体pose与双腕，保留原始旋转、帧间噪声及双手相对运动，不强迫各物体reference首帧等于设计reset。
本轮只读计算得到候选yaw **−2.783916°**、translation约 **[−0.638334, 0.157987, 0.004427] m**；两中心各误差约 **5.78248 mm**。这是新工程锚定候选，**不是已验证的历史world transform或完整6D对齐**。配置应从登记输入重算，不截断这些显示值用于运行。
WP1随场景集成确认pose、实际mesh和口部方向的对应；有资产转换依据时仅记录固定mesh-frame外参，不能变成拟合任务分数的补偿。不存在实际几何矛盾时，不因缺少历史局部系出处或不可辨识的对称轴转角单独阻塞；对实际口部/长轴方向或手物位置无法解释的错误才阻塞。初始中心≤3cm只是候选工程位置检查。
禁止照搬H2S2R逐物体独立锚定，禁止调尺度/逐帧offset。canonical reset与原始轨迹首帧剩余姿态差别必须显式展示；若确认只是设计reset与嘈杂reference不同，不把“reference等于reset”作为RL前置条件，也不预先声称轨迹可执行。
MANO world wrist从已登记decoder的完整root+translation输出构造，再与局部A一致地组合到palm；不能把MANO trans直接当Sharpa wrist位置。
为从共同home开始并返回，reference增加**固定2秒接近prefix和2秒返回suffix**，只从canonical home与ViViDex自算q插值，物体目标分别保持原视频首/末pose；不是复制Ours的653行时间表。142帧原始输入完整保留；训练操作段按后来获批F/S1合成并显式标记，结束后仅保持目标、不写物体状态。
若出现共同世界位置、实际口部方向或腕根组合的可见/可测错误，WP1停止受影响参考的训练并带证据回里程碑；仅历史局部系语义未知或四元数差异不足以触发停止。移动共同物体reset、独立重排两个交互对、重新设计感知或放宽成功判据属于新决定，implementation不得自行选择。

### D3：联合状态RL路线已批准；训练机制不能自动删减

用户明确批准先用联合训练尝试；比较目标是human-video → RL correction最终学会任务，视觉阶段明确忽略。下面联合58维状态PPO作为执行主线，不再要求用户重新选择架构/比较层级。增强/curriculum要尽力满足；此前自动no-augmentation投稿备用建议撤销。

**为什么推荐一个58维policy，哪些不是论文要求：** 原版验证的是单手/单臂手设置，未给出双手联合与分别训练的对比；不能声称它规定了联合58维或证明这种架构更好。58=两臂14+两手44，是当前机构总输出维数。采用两个各29维的policy后，最终控制机器人仍需组合成58维命令。

| 适配方案 | 可行性、优点与代价 | 当前建议 |
|---|---|---|
| 左右各自训练，另一侧固定/脚本；最后组合 | 能降低单次学习问题的动作维数，适合定位抓握/跟踪问题；但训练时对手行为与最终双手场景不同。共同视频时间可以帮助同步，不保证接触、相对位置和失败恢复协调；不能把单侧成功率当整体成功率。 | 可作为诊断/以后明确的预训练方案；独立训练后组合本身也是一种适配，不因保留单侧维度就更忠于原论文。 |
| 同一双手环境中两个29维actor共同训练 | 可保留单侧模块，也能利用双方观测/团队奖励；需要明确共享观测、critic、更新与对手非平稳性，实施上引入额外多策略接线。 | 可选，并非禁止；截止日前没有现成证据说明比单PPO更省时。 |
| 一个58维联合actor，完整双手状态与共同reward | 直接复用单PPO训练/存取流程，训练时看到真实双手耦合，无需先解决两个policy组合。高维探索与共享reward信用分配也可能更难，没有样本效率保证。 | **用户已批准为第一版主线**；理由是少一次训练-组合分布切换和少一套多策略训练逻辑，不是作者规定。 |

采用独立训练时，公平预算需统计两侧全部训练及任何联合微调；不能分别挑最好seed配成最优组合。调试时可以分别屏蔽动作/查看每侧reward，不必因此把正式方法改为两个policy。

**成功轨迹收集/视觉policy到底影响哪个结果：** 原流程为`state RL teacher → 固定teacher rollout/收集成功轨迹 → 用BC或diffusion训练visual student`。公开`tools/generate_trajs.py`加载PPO并predict/写数据，没有反向更新teacher；视觉训练另有入口。故跳过后两步不会改变已训练teacher的权重或其固定state-input评测，但会缺失视觉输入下执行的student，不能作为完整视觉baseline。
视觉student输入为机器人本体状态与场景点云，teacher依赖机器人/物体状态；student是另一份最终模型。成功数据覆盖、点云坐标表示和BC/diffusion学习都会影响student表现，效果不能假定等于teacher或必然优于/劣于teacher。论文分别评测state与visual层，不能混用分数。
**主表层级已确认：** 比较human-video → RL correction后的状态策略任务表现，视觉蒸馏及其专用数据收集明确排除。继续保存评测和复现所需数据。轨迹增强/curriculum属于状态RL训练，不能视为无影响；主线应保留并验证其作用，任何缺失只能明确报告而不能自动降配后当完整baseline。

执行主线：一份联合状态MLP、**58维绝对关节位置目标**（顺序`Rarm7,Larm7,Rfinger22,Lfinger22`），按共同限位把[-1,1]映射到PD target；不采用Ours residual、H2S2R synergy/FABRICS或冻结一侧。
离线双臂IK只解ViViDex腕目标，中立home/上一解初始化；不读Ours arm路径、不调用CuRobo guide。在线只有共同G2认证层可做规范腕偏移。
reward按 pinned ViViDex ObjectMimic分手移植，左右算术平均、共同时间轴、双方pregrasp达标后进入manipulation；保持手跟踪、物体位姿、接触、lift与终止结构，不加Ours成功奖励/相位锁/affordance项。
使用独立venv复用已装Isaac/Torch，新增**SB3 2.7.0**和薄VecEnv wrapper；原版SB3 1.1.0环境不动。这是明确记录的PPO实现版本迁移，非逐位原版复现。[2.7.0官方依赖声明](https://raw.githubusercontent.com/DLR-RM/stable-baselines3/v2.7.0/setup.py)容许现有Torch/Gymnasium版本，但实际联用必须WP1验证。
保留PPO参数：pi/vf `[256,128]`、Tanh、lr1e-5、gamma/GAE .95、entropy .001、vf .5、clip .2、4096全局rollout、minibatch256、5epochs、log_std_init −1.60、正交初始化、Adam eps1e-5；action bias按原工具的计算规则、改用ViViDex自算reference得到。禁止暗加RNN、PointNet、值归一化或自适应KL学习率。
WP1最初的固定场景短PPO仅作**development_only**接线/测量；同包开始移植增强与curriculum，WP2按本次F节修订的8–10工程小时估计完成训练就绪修复与验收，不能先搁置到最后。正式run保留阶段机制，可从固定stage0学习；未自然达到升级条件不等于代码删掉增强，应如实记录阶段占比和触发次数。
正式训练前一次审查双手参数、resolved diff和生效证据；未按期完成则报告真实缺口和成本，不自动批准/采用F1，也不让implementation临时发明替代增强策略。正式B与seed数已按WP1容量收敛为F中的单seed0、3,002,368步；新版本的实际开销和放行资格仍待WP2，不沿用原版60M合同。

#### 状态RL机制保留与验收要求

以下来自本地pinned代码抽查，不要求重审原版全部组件；移植差异在一次里程碑集中审核：

| 机制 | 原版实际行为与入口 | 双手适配的交付证据 |
|---|---|---|
| 轨迹引导reward与pregrasp/manipulation切换 | `models/rewards.py`、`models/control.py`；不是仅训练q误差 | source→适配项映射、小批次reward对拍、阶段状态trace |
| 初始位置/旋转的参考增强 | `HandReferenceMotion.reset`同步变换object和robot参考；`control.py`的Pour阶段0固定，阶段1/2采样XY，所读分支的yaw实际为0，即使`rot_aug=True` | 实际采样参数、前后目标和同步reset；非identity实例（适用阶段）。不能用增加未启用的随机旋转冒充忠实恢复，也不能照搬Adroit自由根位移到固定双臂 |
| Pour目标/操作段合成 | `reference.py`的`random_episode` Pour分支在抬高>0.12m后截断，再以0.02m位移步长合成移向目标及−120°倾倒，并同步变换手参考 | 原版来源与双手改动分别列出；按F/S1已批准映射实现并验证，不再重问方法选择。不能对两个物体各自照搬单手倾倒，不能用Ours处理后的轨迹代替 |
| 自动curriculum | `utils/eval.py`以25次内部评估的pregrasp_success均值>0.95逐次推进0→1→2；内部策略评估为stochastic，区别于最终deterministic外部测试 | 保留来源与升阶作用；双手pregrasp定义、内部验证seed/频率写配置。测试升级和不升级分支、保存重载stage；正式run不得强制预置G1/G2或靠最终测试集升阶 |
| 训练与最终评测分离 | 阶段/增强服务训练；最终使用D1的t0、无RSI、固定外部判据 | train/eval各自resolved config、actual reset/目标记录和计数；训练增强不改变最终共同测试场景 |

全局场景增强应共同作用于双手/双物体；目标条件合成若需改变相对运动，必须来自上述原版机制的明确双手适配，而非随意独立重排。几何/可达约束导致采样拒绝时记录比例和原因，不暗中筛选高分轨迹。只有配置开关、恒等变换或硬设stage值都不构成机制完成证据。当前没有宣称增强/curriculum已移植完成。

## E. 已完成的 WP1 prompt（执行记录；下一包见F）

> 已批准D1、D2非world部分、联合状态RL路线；视觉阶段明确排除。具体world候选仍待落实，增强/curriculum适配参数在首个机制里程碑冻结。不得重问已批准项；本prompt不授权自动采用no-augmentation主结果。

```text
你负责 ViViDex Pour17 WP1：在隔离副本交付“双手参考→物理控制→真实PPO更新→保存重载→外部评测smoke”的垂直闭环，并给出训练/评测吞吐量。不要再拆成多个局部几何审计。

范围与授权
1. 当前只做Pour17，其他三任务暂缓。Pour17用于验证之后复用的同一pipeline：retarget/控制/训练/保存/评测主干不得硬编码杯瓶路径、pour成功条件或视频帧号；这些放task配置和evaluator插件，当前只实现Pour17实例，不扩建通用框架。截止北京时间2026-09-16 05:00。执行BASELINE_PLAN.md active plan A–E；历史规格只作证据。D1、D2非world部分、联合58维状态RL已获批准，视觉policy及其专用数据集收集明确排除，不再请用户选择架构或比较层级。world具体候选仍按D2落实；增强/curriculum必须尽力保留并交付生效证据，不能自动删减换主结果。
2. 首次交付目标8–12工程小时：T+2小时只汇报实质阻塞；T+10目标闭环；T+12硬检查。不要逐步骤请求review。出现需改变方法/世界/输入/成功判据的事项才暂停依赖部分。
3. 本包最多开发短训练与测量，不启动60M或无人监督的正式长跑，不重启native seed1。本包结果统一development_only，不能进入论文主表。

必须复用的来源
- B=D:/UCBP/RL/Baselines/ViViDex。读B/AGENTS.md、REPRODUCTION_STATUS.md、BASELINE_PLAN.md active plan；native协议不变。
- 原始输入：登记pour17_baseline_bundle_20260829.tar.gz（SHA77edc03c…）内perception以及world/reset/evaluator；perception SHA28e13c24…；world manifest SHA2339f7a9…；canonical reset SHAd41cd5d6…；run_eval SHA40e6b157…；完整SHA以现有configs/status为准。
- _inspection仅可提供核对同hash的perception，不可复制其旧evaluator/world目录。
- 复用diagnostics/m1a_full_mano_decode_audit.py、m1a_sharpa_left_fk_candidate_v1.py、m1a_sharpa_single_frame_solver_candidate_v1.py以及configs中的固定规则；左raw分支q可复用artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_arrays_v1.npz。
- 右手MANO_RIGHT从现有资产定位并记hash；缺失即报告，不镜像左q、不下载不明模型。左边已验证的代数不重跑。
- 世界搭建只选择性移植D:/UCBP/RL/Baselines/RL-Correction-H2S2R/tasks/h2s2r_pour17/{cfg.py,env.py}的资产/物理/传感器/按名字关节映射部分；不可继承其phase1_g2 reset、clamp、synergy、FABRICS、trajectory/clock/reward、训练PPO和预置G1/G2 evaluator。
- 方法来源：B/upstream/vividex_mujoco的models/{reference,control,rewards}.py、tools/train.py、algos/policies.py及PPO配置。原版源码不修改。

允许文件范围
- 新功能仅B/adaptation/pour17_v1/：reference.py、geometry.py、scene.py、env.py、reward.py、augmentation.py、curriculum.py、sb3_vecenv.py、train.py、eval_entry.py、smoke.py及必要__init__/有针对性测试；命名可细化，不新建另一套通用框架。
- 新配置：B/configs/pour17_v1/；输出：B/artifacts/pour17_v1/wp1_<runid>/；更新BASELINE_PLAN.md和REPRODUCTION_STATUS.md实际状态。不改旧诊断、artifact、upstream、Ours/H2S2R代码或native协议。
- 远端新代码/venv/tmp/cache/output统一/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/；借用现有Isaac安装，禁止在系统盘大规模装依赖或降级共享环境。
- source provenance包含逐文件hash、复制来源和明确diff；根Git unavailable不修、不编造。

固定接口（按已批准D1–D3；不得现场自选方法）
- reference.npz：schema_version、task_id='pour17'、source_sha256、contract_sha256；source_frame(142)、derived_source_time_s=frame/15、control_time_s；right/left_finger_q_rad(T,22)、right/left_wrist_pose_wxyz(T,7)、joint_q_rad(T,58)、object_pose_wxyz(T,2,7)、source_valid/source_index、joint_names、阶段标签。T为20Hz派生轨迹加2s前/后缀；metadata逐段给精确采样规则。米、rad、xyz+wxyz、env-local坐标；向Isaac world加env_origin只能一次。
- 用已批准raw mean和现有十点/A/alpha；右侧相同算法但真实右模型/链。world方案目前待定，中心+重力只是推荐候选。场景回放只对实际中心/高度、口部/长轴和手物相对位置排错；47.8998°完整四元数差异本身、轴向无关转角或缺少历史出处不能作为阻塞。发现实际几何错误才报告；不做独立物体对齐、尺度搜索、逐帧offset或取Ours参考替代，不静默改旋转reward。
- 双腕由MANO完整root语义和A逆映射一致构造。双臂用限位内IK跟踪自己的腕reference；从共同home、之后上一解初始化。交付IK逐帧误差/失败mask；失败帧不得静默裁剪掉或替换成Ours q。reference有效性检查允许近似手部拟合，不要求十点零残差或PD回放任务成功。
- 环境：reset(seed)->真实obs；step(action)->obs,reward,terminated,truncated,info。action为(N,58)绝对PD目标映射；两手两臂都由同一policy控制。恢复canonical_t0，无G1/G2预置，无物体钳位/运行时teleport；只有reset可写初态。
- observation以固定schema拼接58维q/qd、两手掌和两物体pose/velocity、ViViDex自算hand/object目标与相对量、共同相位sin/cos；具体维数从上述字段推导并写schema/hash，禁止补零伪装原版358维。不读Ours robot参考、confidence、Dexonomy prior、cert_arm、训练奖励或H2S2R reference。
- reward按D3逐项移植并给source对应表；最初固定场景短测仅验证接线。并行接入augmentation/curriculum，依据D3机制表交付实例和状态记录；不能因为短测跑通就把增强/课程当可选项。以真实物理值算reward，不能从目标回填，不得只用q监督损失冒充ViViDex RL。
- 一个Isaac进程承载多个env，薄SB3 VecEnv桥；正确区分terminal_observation、terminated/truncated和bootstrap，不能让自动reset把terminal状态覆盖；seed实际接通，计数使用global transitions。SB3 2.7.0在隔离venv，保留D3参数，避免复制旧SB3自定义policy构造器造成API不兼容。
- evaluator侧提供make_env(num_envs,seed,device)、deterministic policy(obs)、reset_state()、apply_certification_offset(alpha)与实测object_0/1_pose、arm_q_right/left、pads3、wrist_right/left；输出单位/坐标写schema。外部evaluator独立驱动G2及903步上限。按D1批准的实测双腕锁存锚点执行共同G2，不能取Ours认证q；存certification trace。第一次先用登记scalar evaluator真实接入，不直接复制H2S2R的vector_eval。

执行与验收（一次完成，不逐点审批）
1. 启动前现场查GPU进程/空间，按D:/UCBP/GPU_POLICY.md评估GPU0/1并允许共享；不以其他compute进程单独拒绝，不杀进程、不改变他人锁。等待可并行做CPU reference与代码；不要安装本机Isaac当即兴备用路线。
2. 先1个env，再32env：成功加载全部真实资产，reset→step两臂两手都有真实受控响应、接触能从物理传感器读出；3个代表姿态确认runtime joint名字/方向/限位及腕位姿组合，发现新矛盾才扩大检查。记录输入目标、实际q、物体pose、接触，不用目标数组充当实际状态。
3. 完整双手reference运动学回放、开环PD物理回放和PPO rollout分别标记；提供固定相机视频/关键帧与逐帧跟踪误差。物理回放失败可以继续合法RL，但整体坐标错位、非有限值、控制一侧失效、资产/关节对不上必须停止修复。
4. 至少2次完整PPO rollout/update（8192 global transitions，32env×128step×2），有限obs/reward/loss，参数真实改变；保存后另一次加载，对固定obs deterministic action误差≤1e-6，并完成真实物理episode（可任务失败）。保存实际optimizer/step/config与RNG信息；不得宣称恢复了未保存的sim逐位状态。
5. 薄reward移植做小批次数值对拍；VecEnv做真实terminal/reset接线检查。这是防止新增移植错误，不重做既有MANO/FK/A测试全集。记录来源源码与移植差异。
6. 外部evaluator至少2个开发seed真实episode；额外用固定观测trace覆盖G2及终止边界，验证callback被实际调用，不要求短训练自然成功触发所有gate。trace/自检不计物理成功率。eval不能读取train reward来判成功。
7. 在授权空闲卡上测3×约3分钟端到端训练窗口，记录startup、reset、sim、PPO、eval耗时和显存。32env主档；允许追加128env一个档，保持4096 global rollout并注明每env horizon变化。包含一次长horizon执行测量，防止仅用快速失败episode高估速度。开发短训练总计不超过2M transitions或1小时训练测量时间，先到即停；排队时间单列。
8. 首包同步准备状态RL机制交付：原版增强/目标合成/curriculum到双手实现的对应表，已实现部分的增强前后reference与stage切换/恢复检查。未完成部分必须逐项写明确障碍和预计工时，不能填PASS或默认为无影响。WP2补齐后再验收正式训练资格；stage0本身可固定，但后续逻辑必须可执行。

交付文件
- reference.npz + reference_manifest.json（来源、解码/坐标/时间/IK、误差、左右手覆盖）；resolved_config.yaml、observation_action_schema.json、source_manifest.json、dependency_lock.txt；物理短视频与real_state_trace.npz。
- wp1_report.json：明确列reference、kinematic、physical、ppo_update、reload、evaluator状态，不能一个总PASS掩盖未执行项；所有运行命令、真实路径、返回码、耗时、失败原因。
- development final_model.zip、completed_steps.txt、开发eval逐episode记录；throughput.json给各窗口及保守r、1M/5M/10M/60M外推，含启动/在线验证/正式512episode评测开销估计与不确定性。
- training_mechanism_report.json：每项source、适配规则、implemented/exercised状态、增强/阶段实例路径、配置、待决差异、未完成原因与预计成本。不能以功能存在替代实际生效证据。
- 一条可复用train命令、一条加载policy/eval命令、一条从最后完整更新恢复训练的命令；更新现有状态文档。stdout同时输出简短结论和artifact入口。

失败与下一里程碑
- T+2h：若缺右模型、实际中心/口部/手物对应出错、真实资产不全或无法获得Isaac运行资源，报告具体路径/残差/错误、已完成接口和预计修复时间。继续无依赖部分；不把轴向无关旋转或历史出处未知升级为新审计队列。
- 可正常物理执行但零成功/高手部残差不触发方法搜索；交付原始失败数据供WP1里程碑定预算。禁止选更好mean、seed、摩擦或初态。
- 到T+12h仍无真实PPO闭环：交付全部产物与一个主要阻塞，不再自动增加下一小审计；由reviewer按T+18h机制里程碑、T+36h训练最迟线集中裁决。
- WP1完毕停在里程碑：由reviewer据吞吐量和机制报告冻结正式B/seed数及增强/curriculum双手映射。不得自动关闭这些机制或切F1替代主结果。无需再为普通修bug/改路径请示；任何方法、共同合同、reference来源变更才请求决策。
```

## F. WP1 reviewer裁决与当前WP2工作包 — 2026-09-12

**WP2 implementation里程碑（2026-09-12 17:54）：** [交付报告](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/WP2_REPORT.md`）与[机读分项](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/wp2_report.json`）已生成。黑桌、接触6组正反例、从头8192 PPO/update/save/reload、真实25回合callback和32×903 batch均已执行；累计训练8192步/144.061秒，未清零、未扩预算。参考第40步两侧误差仍超过5cm，内部pregrasp0/25，批量32回合均未过G1–G4，因此**不具备正式训练放行结论**。原左index141精确对应S1 index143，局部IK/实际层证据已列明；不把贴限或失败说成全局不可达。正式命令仅供review，未启动3M/最终512。下一步为reviewer集中裁决，不自动执行替代root/标定或扩训练。

**最新用户安排：** 等待本次implementation的WP2里程碑结果，再结合本节和同事retarget包集中制定下一步。本轮不追加替代retarget实验、不中断现有工作、不提前放行正式训练。当前综合状态见 `REPRODUCTION_STATUS.md` 的“当前baseline情况”；接收结果时依据真实参考执行/接触、训练机制生效与新版完整预算三项裁决，不再逐小步骤审批。黑桌已在最新本地产物中确认修正；16:51暂停记录表明此前仅暂停后续调度，新版PPO尚未开始，并非丢失训练更新。

### 评审结论与证据

**接收WP1作为开发闭环里程碑；不接收其为训练就绪或任务学会。** 本次reviewer仅检查文件、源码、既有数值轨迹，并对6个已失败/对照腕目标做同一IK的home初值复算；没有改功能代码、运行Isaac或启动训练。16个源码/配置文件的SHA与交付manifest全部一致；`reference.npz`、`real_state_trace.npz`、8192-step `final_model.zip`的SHA也与报告一致。无需重做这部分PPO接线证明。额外吞吐窗口累计57,344步属于开发成本，不能混入正式从头训练。

| 发现 | 具体证据与判断 | 下一包的处理 |
|---|---|---|
| 腕路径与抓取前目标存在实质问题 | `reference_manifest.json`：右168/269、左194/269个IK样本未达到既定5mm/0.05rad容差，位置误差P95分别约23.91/23.64cm。核心首样本index40右腕误差8.62cm。`real_state_trace.npz`在接近pregrasp边界的index38测到左右十点误差14.87/22.53cm；当前训练却在39步、两侧均<5cm时才进manipulation。 | 不能依据总关节误差中位数0.0035rad认定参考已可执行；手指44维会掩盖臂部问题。定位姿态/限位分支/真实碰撞/动态跟踪的区别，修代码错误，不调大成功门槛或投影失败目标来掩盖。 |
| 不是换回home初值就解决 | 对两侧index39/40/141的现有目标使用同一solver、相同250次上限和home初值，只保留index141右侧原本已通过的正例；其余5个仍失败。 | 此有限复算不证明目标不可达；不继续盲目增加迭代或重复FK代数审计。下一次验证必须在实际场景里回答腕/手为什么到不了工作区。 |
| 接触没有正例 | 269步open-loop的10个指腹通道全为false。物体有显著位移，但可能由其它手部/臂部碰撞造成，不能据此断言指腹传感器坏了。`_tip_contacts`和`_object_table_contacts`遇到缺失force matrix会返回零；报告另记table过滤不支持的警告。 | 必須验证目标物体接触/分离的实际正反例，缺失传感器是错误而不是“无接触”。物体-桌面、整手-目标物体、指腹-目标物体分开接线。 |
| RL训练语义未完整移植 | pinned `rewards.py`在pregrasp用掌心+五指尖，manipulation用五指尖，整手任意接触作总门控，五指接触另加分；当前`reward.py`两阶段都用十点均值，并把>0.5N指腹接触当总门控。现有NumPy/Torch对拍只覆盖自写pregrasp公式，不能证明源实现等价。 | 十点保留给NLopt retarget；RL reward/阶段统计恢复原版对应结构，验证全部关键分支。0.5N是共同G1判据，不应直接替代所有训练接触语义。 |
| 其它影响状态policy的差异 | 原版`HandObjectReferenceMotion.goals`含未来1/5/10个reference索引的object pose；当前286维观测只含当前目标。`get_warm_start`还有归一化动作+0.2偏置，当前train只用home。 | 补前瞻目标及源状态特征的对应表，按语义做具身适配；不得将home-only写成完整原版warm start，或因视觉阶段排除就忽略这些状态策略输入。 |
| 增强/curriculum仍未上线 | stage1/2只改目标、不改实际物体reset；自动25回合验证/推进/恢复没有连PPO；Pour合成缺失。 | 同包完成，不以采样器/状态类单测代替实际生效。固定基座的机器人home不应被随意整体平移，具体规则见下。 |
| 外部评测成本超出旧预算 | 已复用scalar Kit，2×903步耗565.3s；512回合约40.20h，不能按“最后12h足够评测”安排。32×903物理smoke为306.55s，只是批量化容量线索。 | 前移batch evaluator实现及真实计时；保留512和原判据。不要再做串行512的无预算启动。 |
| 视频不可用 | 已交付黑屏验证；两个evaluator回合无录像。 | 修显式相机/render并重跑，新增视频是同配置重新执行，不是原两条轨迹的追溯录像；不承诺逐位重演。 |

开发0/2不足以否定联合PPO；上述阻塞来自源码和真实轨迹，不来自挑分数。WP1的“PASS”只在各自声明的接线层级成立。

### 本次研究决定与用户批准

**2026-09-12 用户资源授权更新：** WP2 可使用 GPU0 或 GPU1；即使存在其他用户作业，也可在现场检查显存、利用率和主机内存后判断安全共存。此项取代上文历史的“有 compute 进程就排队”限制；不授权终止/修改他人进程、改主机设置或启动正式3M。当前隔离单作业入口保守要求所选卡至少24GiB空余，并先做小规模探针。

**已决定的执行取舍：** 保持一个联合58维状态PPO、raw MANO分支、原物理世界/共同reset/G1–G4及无视觉student边界。不另开单手训练或换solver/simulator路线。相机、接触、reward/观测源语义恢复、自动curriculum和批量评测属于下一包的集成交付；不要求每个普通修复再请示。用户已明确批准下列W1/S1按文中规则实现。超出这些规则的共同物理参数/判据、输入来源或新方法改变才上报。

**W1：现有共同放置已获用户批准，冻结为新工程约定。** 使用现有单一gravity-preserving yaw/translation，scale=1，同时变换两个物体和两手：yaw约−2.7839162558°，translation约`[-0.6383340163, 0.1579874289, 0.0044270754]`m，以hash绑定的`reference_preik_manifest.json`精确值为准。不是对两物体分别配完整四元数。按登记local +Y口部轴解释既有reference与第一物理step，杯/瓶轴夹角约5.35°/8.49°，口部位置差约8.37/14.42mm；这些数值依赖相同mesh-local-frame解释，不是完整资产认证。没有新证据要求追平47.8998°的完整旋转差。W1固定输入约定，不认证腕轨迹可达；实际表面/手物分离仍须WP2解决。正式run仍以WP2训练就绪验收为条件。

**S1：以下双手Pour目标合成规则已获用户批准。** 这是原版单手目标合成到双手任务的显式适配，须保留source和本段差异，不称原论文已规定：

1. 完整保留原142帧及自身MANO/物体输入。取瓶子相对初帧首次抬高>0.12m为cut；取cut之后瓶子口部轴相对竖直倾角最大的最早视频帧为pour keyframe。若不存在有效cut/倾倒段，报告源数据不满足，不按训练分数另选帧。
2. 从cut到pour keyframe，右侧沿原版0.02m步长构造到自身视频keyframe瓶子位置的目标移动；使用120°倾倒。倾倒轴由world up和“cut瓶子指向keyframe杯口”的水平向量叉乘定义，正向朝杯倾倒；退化时报告，不能任意转两物体。保留keyframe的瓶子目标高度，明确区别于原版固定接收容器时将goal z覆写成cut高度的行为。这个高度适配由同一视频给出，不能使用Ours机器人轨迹。
3. 右手腕/手部点随瓶子同一刚体增量变换，右指构型保持自身cut参考；左杯、左腕和左指从同一视频cut→keyframe同步重采样到合成时间轴，不能给左杯也执行单手倾倒。两侧始终由联合policy实际控制，没有脚本托杯或冻结执行器。
4. 合成末端增加25个20Hz参考样本的保持段，用于完整任务所需的持续倾倒；之后用0.5s的xyz线性/quat SLERP桥接回自身视频keyframe，并接回其后原视频的放回/释放段和已批准的2s返航。明确这是共同任务持续判据/放回阶段的适配；不是用Ours的成功轨迹修参考。每个合成/桥接样本标来源，不能伪造原视频frame/time。
5. 两侧目标仍是训练引导，不写入物体实时状态或作为policy动作覆盖。切换前检查端点连续性、口部几何、桌面/可达性和完整时长是否满足903步上限；名义目标若连倾倒几何都不成立就报告具体差异，不缩短持时、改目标物体reset、按SR优化倾倒轴或引入参考锁时。

**固定基座增强/课程映射（本次reviewer规则）：** stage0名义输入；stage1/2采用既有原版XY范围`x∈[-0.06,0.04], y∈[0,0.1]`、yaw=0。共同移动两物体的真实初始位置及双手/双物体操作段目标；机器人基座和canonical home保持固定，重新连接home到该episode目标的approach及返航。不能机械平移整个robot reset或把home参考也漂走，不能只平移目标。非名义样本几何不可用时记录原因/比例并报告，不静默缩小采样范围或挑高分样本。

pregrasp统计为两侧各自“掌心+五指尖平均位置误差<0.05m”在统一pregrasp时刻的AND；不要求共同G1已通过才升级，也不以两侧均值掩盖一侧失败。统一时刻按已批准2s approach对应的40个真实control steps/core首样本绑定，不能把0-based最后approach索引39误当已完成40步；reward/阶段更新先后须与该时刻一致并留边界trace。保留25个stochastic内部episode、严格均值>0.95、stage0→1→2。内部seed固定100000–100024，独立于最终evaluator。首次完整rollout后和每约1M global transitions的完整更新边界验证；阶段更新同时作用训练和内部验证env。保存/恢复每个env RNG、全局RNG、curriculum、已用步数和下一验证点；接口测试可用构造的成功/失败输入覆盖分支，但正式升级只用真实记录。源码频率对齐到完整更新的差异记入manifest。

**状态reward/观测的实施规则：** NLopt仍用已批准十点；RL pregrasp用五指尖加掌心，manipulation手跟踪仅五指尖。掌心使用现有A构造中四长指MCP均值对应的固定root-local点，目标/实际采用同一定义；不把失败IK的FK结果回填为“原目标”。左右算术平均、双方pregrasp的AND保持。整手接触总门控覆盖该侧手与自己的目标物体；指腹接触奖励、G1的0.5N阈值分离；桌面接触不能以missing=zero代替。新增观测schema版本，补源实现的实际掌心/五指尖、对应手物相对量、最终目标相对量、1/5/10 reference索引前瞻object poses；单位/坐标/截断/双侧顺序明确登记，输入只取当前真实state或自身reference。不能只把新维数写进说明而保留286的硬编码wrapper。

warm-start按原版可对应项保留：以canonical home的归一化动作作基值，对44个finger动作加入原版归一化+0.2并clip；Adroit自由根平移/独立腕关节索引不能照抄到7DoF臂，臂bias保留home。将这个具身差异和初始实际action逐项记录，不声称逐位原版一致；不调偏置择优。初始偏置不读取Ours q，不等于加载预训练模型。此前home-only开发checkpoint保留但不得用于新版续训。

### 预算、时间与放行条件

**正式容量计划：只安排seed0，requested=3,000,000，实际完整更新预算B=3,002,368（733×4096）global transitions；32 env×128 steps，最后更新的checkpoint。** 不恢复seed1701等开发模型，不自动加第二seed。选择基于算力与截止时间，不能作多seed稳定性结论。按r=31.09755，仅训练约26.82h；另计新增强/reset、25回合在线验证、恢复、评测。一次完整操作学习也可能需要远超此预算；不把有限预算低SR解释成方法固有失败。完整记录开发和正式成本，比较Ours时列清训练预算/观测权限/共同runtime。

- 当前review约北京时间09-12 14:30；WP2工程上限估计8–10h。开始后2h应有可见场景及接触正反例或一个明确底层错误；4h时应能判断名义approach受限于哪一层。已定位错误可以继续修，不能用“再多跑几次”代替结论。
- 目标09-13 00:30交付WP2、02:00启动通过验收的正式run；09-13 14:00仍是最迟启动界线。GPU排队或批准等待消耗余量，不顺延截止，也不改共享主机设置来挤占他人。
- WP2短PPO总上限65,536 transitions或1h训练墙钟，先8192确认新版接线，再用同版测开销。不必重跑旧版三个窗口；可按原计划额外测一次128-env纯容量候选，但未经新配置登记不替换32-env正式主线。训练性能排查只改本适配的采集/render安排，保持physics dt、decimation、传感器语义和控制周期。
- batch evaluator先32 slots×903测一次真实长回合，再按测得reset/判定/认证开销外推512；94.26 transitions/s仅暗示约1.36h，不是评测保证。目标完整512评测≤6h含余量。scalar已经复用Kit，需并行物理槽及每槽独立的原判定/G2 callback；不得让槽间共享latch、phase、seed或episode状态。
- 正式放行条件：W1/S1已获准；名义参考不存在未处理的坐标/接触/物理执行错误；源机制对应和真实生效证据齐全；新版短PPO/保存重载通过；测量显示B及在线验证、最终评测可装入剩余窗口。不要求先在开发PPO拿到任务成功，也不要求所有142帧零误差。
- 若4h后nominal reference仍明显穿桌/脱离工作区/卡在限位而无法解释，停止给该参考扩大PPO预算，提交一个可见反例及最小修复/方法备选；其余集成可继续。不能把候选“投影成可达”或移动共同reset后继续叫原版本。
- 09-15 05:00前结束训练、09-15 17:00结果封存、09-16 05:00投稿不变；实测评测超过6h则提前停训安排。确需改变512、预算、共同世界或新增预训练/放宽门槛，带具体成本/偏差回研究决策，不自动降配。无正式资格则F0；允许报告真实有限结果，但不造SR。

### 黑桌纠偏与中断恢复裁决 — 2026-09-12

用户指出画面仍是棕桌，implementation随后暂停新增仿真/PPO。reviewer确认黑桌历史在交接材料中存在，前述工作包也漏写了该显示约定。当前源码仍为棕色；本次仅更新计划，没有修改功能代码或恢复作业。

- **直接修显示遗漏，继续原WP2，不新增方法审批。** 依据 `D:/UCBP/reports/h2s2r_run04_black_table_physics_audit_2026-09-03.md` 的颜色补丁范围，将本适配桌面 `PreviewSurfaceCfg.diffuse_color` 改为 `(0,0,0)`；后续补录使用的WP1兼容副本也采用黑桌并标记视觉版本。相机不进入状态actor/critic；仅此显示字段变化不改变状态观测、reward或动力学。保留原视频、checkpoint和manifest，不因颜色重做训练或旧审计。
- **不是整体换环境。** 新版USD哈希和桌面尺寸/高度匹配只支持这些字段，不能认证完整物理等价。环境总表12.1另含H2S2R Run08/09的杯瓶0.5kg、摩擦1.0等不同分支设置，不得批量迁入。保持本包已登记的USD、碰撞/材质物理参数、质量、PD、reset、W1/S1和判据。若发现具体物理合同冲突，单列字段/来源/影响交reviewer；不能用“黑桌”标签决定整套物理参数。
- **最小验收并入已有相机检查。** 在颜色修改前保存当前WP2源码快照，记录颜色补丁差异，确认该补丁未触及物理/控制/reward/观测字段；源码总哈希会变化，应更新manifest，不要求其保持旧值。在下一次已安排的相机探针确认黑桌且场景可见，保存一帧及配置绑定即可；不用另开物理等价审计或重录所有历史产物。不受影响的CPU集成可同时继续。
- **先记录真实暂停点，不能假定已有PPO进度。** 当前已交付状态只明确8步相机探针通过、接触检查失败且新版PPO未开始；implementation须用作业日志说明实际停止的是探针、PPO进程还是后续调度，以及最后完整更新步数。若确有同一训练版本的完整checkpoint，颜色本身不使它失效：恢复可用的模型/optimizer、RNG、curriculum及计数，丢弃未完成rollout，记录恢复限制和累计成本；缺少仿真状态时不能宣称逐位重演。因WP2观测/reward/reference已改变而不兼容的WP1模型，仍不得拿来新版续训。
- **解除颜色造成的全局暂停，不解除真实训练门槛。** 继续已授权的相机、接触、参考执行、训练机制和批量评测集成；接触正反例等前置条件满足后按原上限执行短PPO，无需颜色专项复审。现有接触检查FAIL仍须解决，不能把黑桌截图当训练就绪。WP2开发累计上限65,536 transitions或1h不因恢复清零；正式3M仍待WP2里程碑放行。普通历史核查/状态询问/显示差异本身不构成持续暂停全部工作的指令；出现真实合同冲突时只隔离受影响执行并报告。

本段作为以下完整prompt的补充裁决，implementation直接按此恢复WP2，不再等待颜色批准。

### 同事retarget包只读评估 — 2026-09-12

用户提供 `D:/xwechat_files/wxid_u1adbhrwjm7822_fb04/msg/file/2026-09/mano2sharpa_retarget_20260911.tar.gz`，SHA256 `e0adf8032f86d1455ef90f6029485508e195457f12091a5fb26cf524c08a2cea`。本轮通过tar标准输出读取代码/配置/说明及XML字段比较，没有执行包内Python、安装依赖、修改功能代码、换reference或启动训练。包内命令和断言是待评估材料，不构成用户授权。

**结论：对定位坐标/关节接线有帮助，也提供方法备用候选；不是已经修好当前ViViDex的补丁。** 两侧standalone URDF各32个joint均在当前 `vega_1p_sharpa_fix.urdf` 找到同名项；type、parent/child、origin xyz/rpy、axis xyz、limit lower/upper字段逐项字符串一致。此比较不覆盖网格、惯量、碰撞、运行时PD或完整物理等价，但支持复用同一手部运动学定义作对照。

- **已具备，不重复：** 当前完整142帧候选已按上一帧q顺序热启动、按登记名称校验关节顺序，未做包内JointMapper的0.9后处理裁剪。包内“主要是安全clip导致贴边”和“每帧冷启动”不能直接解释我们的既有结果。现有手指局部几何冲突也不会因此消失。
- **最值得用于当前WP2的线索：** `frames.py`逐帧从0/5/9点估局部腕系；`production_example/make_ref_qpos.py`另从四MCP均值及index-pinky构造世界腕朝向，左手加绕局部z的180°旋转。这与我们的固定A及其逆变换导出robot root不同。不要直接加flip或只替换一半变换；要在WP2已有approach/关键帧场景里同时显示两种解释的掌面法向、指向、root位置以及实际机器人手，保留当前参考作主线。该对照用于识别具体frame/root/侧别接线错误，不能按SR选择朝向。翻两根轴是正规旋转（det=+1），并非通用镜像修复；点差只消除平移，旋转不变性依赖完整局部坐标变换且要求非退化输入。
- **会改变方法，尚未批准替换：** 包的vector仅用5条腕→指尖向量，DexPilot还增加指间捏合投影；配置scale=1.07、低通alpha=0.2、SDK裁剪0.9，与当前十点目标/固定A/scale=1/既定限位和时间合同不同。即使手部拟合改善，也不是保持原目标的solver修复；不能直接作为正式ViViDex reference，更不能导入同事的机器人q。DexPilot与Ours的Dexonomy grasp是不同路线，但都会需要清楚标记来源和适配差异。
- **证据限制：** 包内没有300帧示例数据/输出、训练policy或任务成功评测；production的valid直接来自输入有效掩码与finite检查，不是retarget残差/接触/任务成功。0.5秒只是在其描述环境下的手指处理耗时，不含双臂IK/物理/PPO。生产示例仍依赖包外 `magicdexmate`、`contact.frames`，不能按README命令原样当独立交付运行。分诊C段仅打印左手flip建议，B段没有计算机器人FK标定值；D段冷/热差还可能混有低通状态和抽帧效应。其输出不自动给出根因。

**截止日期下的建议：** 将坐标/root对照并入现有WP2腕部诊断，额外工程时间上限估计1–2h，不重启MANO/FK/A整套审计，不暂停独立集成。若证据指向明确接线错误，按已有普通修复权限处理并验证；若只能通过换向量目标、缩放、逐帧腕系或捏合先验改善，提交一个具体方法备选，由用户决定是否另设隔离对照。此材料评估本身不批准新的替代reference实验或正式长跑。

### 可直接转达给implementation的完整WP2 prompt

```text
你负责ViViDex Pour17 WP2：把已完成的开发闭环修成可正式训练的版本。这是一个集成工作包，按下面验收一次性交付；不要再拆成逐个小审计后等待批准。reviewer不代写代码、不启动训练。

先读D:/UCBP/AGENTS.md、D:/UCBP/RL/Baselines/ViViDex/AGENTS.md及REPRODUCTION_STATUS.md，再读BASELINE_PLAN.md的F节；E是已经完成的WP1，不是下一队列。原版POUR_TWO_SEED_PROTOCOL.md保持不变。

输入：artifacts/pour17_v1/wp1_20260912_02/的reference.npz、reference_manifest.json、runtime_probe.json、real_state_trace.npz、final_model.zip、wp1_code_source_manifest.json、training_mechanism_report.json、throughput.json；wp1_20260912_01的preik输入/manifest；现有adaptation/pour17_v1源码和pinned上游reference/control/rewards/util/eval。先复用哈希已核对的输入，不重跑MANO/FK/A旧审计。

允许修改：仅ViViDex/adaptation/pour17_v1/及configs/pour17_v1/，可增render_capture.py、batch_eval.py和本包必要验证入口。产物新建artifacts/pour17_v1/wp2_<id>/，旧WP1不可覆盖。更新BASELINE_PLAN.md和REPRODUCTION_STATUS.md，不改上游、Ours/H2S2R代码、登记bundle、共同物理参数/阈值。远端新文件/venv/cache/output继续放数据盘隔离根。先检查实际GPU进程和空闲资源；不碰他人任务/主机全局配置。

授权边界：联合58维状态PPO、raw MANO分支、无视觉student及保留状态训练机制已批准；用户也已明确批准F节W1/S1按文中规则实现，不再问。直接把获批规则落实到新版配置/参考和实际训练机制；它们的批准不等于现有候选已通过物理验收。本包不要启动正式3M长跑，完成后一次提交训练就绪证据。

先推进真实阻塞，最多2h给出可见场景/传感器结果，4h给名义参考层级判断：
1. 显式接通一台覆盖机器人双手、杯瓶和桌面的Isaac camera/render product。先验证连续若干帧包含真实场景、非全黑且不是静态缓存，再录完整回合。先保留WP1兼容运行副本（286维观测/原物理与控制语义，只增录像），重跑同8192-step checkpoint和原两development seeds，用新目录保存两条evaluator视频及对应逐步state/action/判据trace；标明是重新执行，不能声称录像恢复旧回合。不要拿旧checkpoint输入新的schema_v2，或把修复版行为标成旧失败策略原样重演。另录修复后的自身reference open-loop；目标叠加必须与真实物体区分。相机不进入actor/critic，正式训练默认不捕获画面。
2. 验证每侧真实手-自己的物体接触、指腹接触、object-table接触的正反例。只在独立开发诊断中可施加构造的接触测试，不能计task SR或设正式G1/G2。missing/不支持的传感器显式失败。训练整手门控、逐指奖励和外部G1的>0.5N分别实现，不能用总受力/桌面碰撞冒充目标物体接触。
3. 对approach末端、原index40及左index141同时记录desired wrist、IK FK、实际wrist/hand、臂关节限位和接触；用现有连续PD执行判别目标/IK/实际动力学差异，不只汇总58维关节中位数。已知home重试不能解决，不再盲加迭代。允许修帧变换/命名/角度单位/求解器实现错误及既定home/previous初始化处理；不得改A、mean、joint limits、物体reset、目标源或投影失败目标。不是要求全轨迹零误差，也不是要求训练前已完成G1–G4；是要求排除让训练目标失真的实际错误，并说明仍有近似误差是否能进入既定pregrasp门槛。

同包完成训练机制，而非另等一个研究任务：
4. 按F的状态reward/观测规则恢复pinned结构，交reward source-to-port对应、掌心/五指尖定义、全关键分支数值例。源对拍须包含pregrasp、未接触、只有非指腹手部接触、各指接触、目标/实际lift、桌面接触、手奖励门控、失败终止；比较源公式本身，不能自写同一公式互相对拍。补实际手点/相对量、最终目标与1/5/10前瞻目标，写schema_v2、创建对应VecEnv spaces和终态观测，不能保留286硬编码。warm-start按F记录可映射的+0.2 finger偏置和固定臂home差异。因schema/reward变化，新版PPO从头建模，旧checkpoint只用于补录像。
5. 按F实现固定基座stage1/2同步物体reset/操作目标，home和base不平移，approach/return重新连接。按已批准W1/S1完整双手合成规则接入；不把左右手分别当单手pour，不只留旋转开关。源视频和合成索引、时间、参数、IK/采样失败全部保留。展示至少一个非identity live reset及其对应双手目标，实际state读回验证；失败样本不能静默换成identity。
6. 将25个stochastic内部episodes、双手pregrasp AND统计、严格>0.95、0→1→2真正连进PPO。终态统计在reset前捕获，不能平均掉失败侧或用会被reset清零的extras。内部seed100000–100024；首次完整rollout后及约每1M的更新边界验证；一次stage变更作用train和internal eval。完成构造分支覆盖+一次真实callback运行；自然未升级如实记，不假造成功升级。checkpoint/恢复必须含env RNG、阶段、验证位置和实际步数；当前只写rng_state.pt不恢复并不满足。
7. batch evaluator复用登记progress逻辑，逐slot拥有903步时钟、G1–G4、latch/15mm认证动作和失败记录。先在相同固定trace（包括正/负gate和deadline/timeout）上验证scalar判据完全相同，再跑真实32×903 long-horizon，计入reset/模型推理/判定成本。增加batch不能预置G1/G2，不能把synthetic trace计成功。reset_manifest保存实际读回state和哈希，不只是cfg常量；旧evaluator元数据中的Ours reference仅用于登记共同判据，不能在baseline hidden path读Ours手/臂q；本次ViViDex reference/checkpoint/overlay的哈希另外明确登记。

验收与交付：
- wp2_report.json列“已修复/实测通过/未完成/待方法批准”，不能只给总PASS；给名义参考在训练门槛处的实际证据。world/fidelity若仍失败，提交可见具体反例和修复成本，不扩大训练。
- camera_validation.json + 两个重新执行的evaluator mp4/json/逐步trace + open-loop mp4/trace；确认内容/相机覆盖/帧数/时间，不把黑屏文件存在当成功。
- training_mechanism_report.json + source_to_port.md + resolved schema/config + live nonidentity实例 + curriculum实际callback/恢复证据；所有重要差异均有来源或F的研究决定。
- batch_eval_validation.json + 新版本throughput.json，给正式3,002,368步与512回合的含开销墙钟预测；不以94.26物理smoke速度当PPO或最终评测速度。
- 新版8192-step PPO/update/save/reload通过，开发本包总训练上限65,536 transitions或1h，不在旧版上重复窗口。保留全部开发结果，不选最好seed/版本。
- 新逐文件source/input/manifest hashes、完整命令及policy加载说明；如恢复不含完整simulator状态，明确说明而非宣称逐位续跑。

正式容量选择已由reviewer给定：seed0、requested 3M、实际3,002,368、32×128/global rollout4096、最后完整更新。这里只生成可审查的启动命令和最终配置；WP2完成停在一次训练放行里程碑，不自动启动长跑。目标北京时间09-13 00:30交付、02:00有条件启动，最迟14:00；不保证完成。普通修bug/接线无需逐项请示；超出已批准W1/S1的新方法、共同合同改变或明确失败才上报。不得用无增强版本替代主baseline。
```

## G. WP2评审：场景放置、reset与参考执行闭环 — 2026-09-12

**WP3执行交付，待reviewer决定（2026-09-12）：** [报告/视频入口](README.md#archive-access)（归档：`artifacts/pour17_v1/wp3_20260912_01/WP3_REPORT.md`）。根/公共reset与登记一致，但固定躯干l2/l3与桌面存在真实visual/collision相交；无生产场景修改、无新增PPO。已按本节停止依赖world的新参考物理执行，复用旧trace并完成保存左21点的非物理对照。唯一[共同world提案](README.md#archive-access)（归档：`artifacts/pour17_v1/wp3_20260912_01/common_world_revision_proposal.json`）为完整固定根env-local(-0.75,0,0)m；只提出未实施，臂可达性未认证。需reviewer明确决定后再做双方同版短验证/各自臂参考重建；这不是正式训练就绪或3M批准。

**接收WP2为新版训练机制/接触/录像/批量评测的集成交付，不放行正式3M。** reviewer读取总报告、分层参考误差、reset记录、颜色补丁及吞吐；抽查16个相关源码/报告hash与delivery manifest一致。6组构造的真实物理接触对照通过；从头8192-step PPO、真实25回合callback（0/25，stage0）、32×903批量执行均已交付。保存重载动作和RNG检查通过不等于完整simulator中断重演；这些已通过项目不再整套重做。

**用户截图提出新的具体场景疑点。** reviewer实际查看棕桌 `output/camera_v1/probe_probe_01.png`、黑桌旧策略 `output/old_seed_0_black_v1/eval_probe_01.png` 和新版参考 `output/openloop_black_v2/inspection_step40.png`。改色前画面中已出现桌板横穿躯干所在区域的视觉关系，故不能归因于黑色材质补丁；两相机视角不同，也不能用像素差估计坐标变化。此前黑桌验收仅覆盖显示可见，未认证此场景几何。

旧seed20260829录像属于WP1/286D兼容重执行，不是新WP2/544D策略效果。其报告的58个reset q与登记canonical逐项差为0；这是报告值核对，不证明整个reset物理正确。登记canonical规定root由USD固定根决定、不要写root，并明确 `(0,0.3,1.2)` 只是历史占位；不能把它照抄为修复。当前scene未显式设robot root；reset写关节状态/target及物体pose。reset报告尚不包含完整robot根/躯干与桌面读回，FK/runtime腕吻合只认证两者在当前放置下相符，不能独立证明该放置就是预期世界。

**残余参考问题已有物理证据：** step40右/左实际掌心+五指尖误差129.55/176.63mm；右目标→IK86.22mm、→实际124.64mm，左目标→IK29.79mm、→实际175.97mm。原左index141映射S1 index143，IK残差239.42mm、实际与该IK FK相差约0.42mm。不能继续统称“MANO不准”：场景几何、目标/IK误差、物理跟随必须分开。局部点对不相容事实保留，不重做证明。

**预算更新：** WP2 stage0实测56.86 transitions/s，3,002,368步直接外推14.67h、加25%18.33h；512批量评测加25%约2.10h。仅40步失败episode和未成功G2认证的测量，不保证后续阶段。正式启动/停训/结果封存既有截止不顺延，不以乐观吞吐吞掉评测缓冲。

### 可直接转达给implementation的WP3 prompt

```text
执行ViViDex Pour17 WP3：把用户指出的机器人—桌面重叠与reset疑点，和现有参考执行失败收敛成一次可决策的交付。先读两级AGENTS、REPRODUCTION_STATUS和BASELINE_PLAN G；F保留W1/S1与机制规则。正式3M继续不放行；本包不追加PPO，不为低分重复旧训练和接触整套验收。

输入固定为wp2_20260912_01交付manifest、WP2_REPORT.md、reference_execution_layers_v1.json、runtime_probe/登记canonical与world、颜色补丁和棕/黑原画面；旧策略录像为286D兼容重执行，新参考为544D版本，禁止混标。同事retarget包仅用于后述坐标解释对照。

允许范围：ViViDex/adaptation/pour17_v1及configs/pour17_v1中的必要场景诊断/既定合同接线修复；新产物artifacts/pour17_v1/wp3_<id>/。保留WP2原件，记录source-before/diff/hash。不得改上游/Ours、登记bundle、桌面尺寸/高度/位置、机器人根、质量/碰撞过滤/PD、reset q、A/mean/限位、G1–G4或W1/S1来试分数。真实普通接线错误可以修；若改变登记物理/初态或方法，先做具体可审阅方案再请求决定。先现场检查GPU资源，不影响他人。

1. 先用同一相机的正侧/俯视检查canonical reset和短时间保持。只运行2个槽、每次最多40控制步，覆盖首次reset、保持、再次reset；明确action为home对应实际关节target而不是把normalized全零误叫home。完整记录render帧对应的completed-step、参考index、动作及写入/真实读回时序。
2. 读回每槽env origin、Robot根prim/固定根body/arm_center/相关躯干/双腕的世界与env-local pose、58 q/qd/target、桌面collider的实际transform/尺寸、物体pose。对照同一USD authored transform、固定约束、scene初始化和登记初态；检查是否漏加/重复加env origin、覆盖USD根、过期tensor或joint target未同步。不要只比较q数组，也不要把历史占位root直接写入。明确canonical home是获批中立起点，不是Ours near-grasp；若用户预期的旧画面来自另一reset分支，记录版本差异，不直接切过去。
3. 针对画面疑似相交的躯干/臂/手，区分visual mesh与collision shape。AABB只筛候选；用实际碰撞几何相交/分离证据和目标pair接触信息确认、附侧视或只改渲染的碰撞形状叠加。固定躯干没有碰撞形状/碰撞被过滤时要明确，零contact不能证明无穿透。输出到底是视角遮挡、只visual重叠、实际collision交叠还是运行中运动造成；不能把这些情形统称物理PASS。
4. 如发现实现与已登记场景/初态的具体偏离，直接修复并重做同样短reset/保持；如果忠实执行登记世界仍有需要改变的场景几何，输出一份字段级共同world修订建议（依据、精确变更、对两方法/reset/W1/reference/旧checkpoint的影响），不现场挪桌/挪机器人或关闭碰撞蒙混。world若需要新决定，暂停依赖它的物理执行，独立的CPU参考分析可继续。
5. world/reset可解释后，用现有PD连续执行自身reference至step40及必要时S1 index143；记录desired wrist→参考IK FK→实际q FK→Isaac腕、掌面/手指方向、root offset、实际手与桌的接触和关节target/实际误差。不扩大IK迭代/随机重启。把同事包的腕系解释在这些关键帧作非物理叠加，保持原q/reference；完整标记其局部归一化和世界root之间的约定，不能直接添加左手180度或导入同事q。若仅改变方法才改善，给一个具体备选；不继续叠加多个待定分支。

交付：wp3_report.json/MD，reset_world_readback.json（每槽/每时刻），scene_geometry.json（visual/collider区别和证据），短reset/保持视频、step40层级trace/可见反例、修复diff/hash；若需方法/世界修订，给精确候选和影响，不能只说“坐标不对”。这些是建议接口名，可按现有产物封装，字段含义不可省略。

时间与停止：2h内完成场景/reset的具体判定，4h内整包交付（工程估计，含短物理检查，不含训练）。普通实现错误修复与短验证合并交付，不逐步等批准；2h后仍无法判定时报告缺少的唯一关键读回及成本，最多用剩余2h获取，禁止无限审计。明确需要共同world/方法改变就提交可审阅方案；未解释的物理/参考问题不追加PPO。已有接触、reward分支、8192更新等证据只在相关改动影响时重验。
```

完成后由reviewer决定：实现修复后的正式就绪验收、共同world修订，或明确标记的retarget适配备选。不能以“代码忠实于旧bundle”替代场景合理性，也不能仅凭疑似穿桌图宣布所有实验无效。现有checkpoint保存为开发证据，不承诺未来改版可直接续训。

## H. WP3裁决：批准单一共同根候选进入短验证 — 2026-09-12

**交接截止快照（2026-09-13 01:38:28 Asia/Shanghai）：** DEV_TO_FIRST_CURRICULUM_V1获批且正在运行；服务器progress=577,536、stage0，524,288两组固定诊断已执行，最终结果尚未封存。此时不是下方旧状态中的“待预算批准/无遗留进程”。文档被同步后，接手AI agent必须先向Dennis索取训练结束后的报告、最终checkpoint/state、预算与hash，不能根据此快照重启或续开作业。该要求同时见README、AGENTS和REPRODUCTION_STATUS；精确运行证据在verification/handoff_20260913_snapshot.json。本次只同步，不新增方法/训练授权。

### 当前执行：DEV_TO_FIRST_CURRICULUM_V1已批准并运行中（2026-09-13）

reviewer核对追加报告、64窗口及固定评测记录；checkpoint SHA256=92868d6525087ae36da2acb09982432af074d4a6d2c5208a40de7f4039b19483、delivery_manifest SHA256=d57ab40e4232ef6aac22358ecdc19475decbfba1c11a496aa672db8830c94386均一致。确定性25回合均到40步，右/左中位96.85/163.68mm，臂桌接触记录0/25；其残余误差不能主要归因于已观测碰桌，也不能把无记录接触升级为所有子步无碰撞认证。stochastic接触9/25并伴随较大误差，提示探索相关执行问题，但尚不识别因果，不足以直接降std/改PD。最后8窗口边界误差中位的中位数右166.37/左202.31mm，前8窗口171.39/212.65mm，下降较慢但未证实停滞。仍无联合通过，保持未学会标签。

**研究选择：保留方法，不实施桌约束IK，不重做目标来源审计；建议一个更完整学习段到既定1,003,520检查点。** 这是既有curriculum调度中的真实方法里程碑，不是宣称训练到该步一定升级。仅继续到262,144始终stage0不足以检查百万步调度。本轮不改变原方法/评价合同、不放行正式3M，不因已有改善就自动增加预算。

**已批准的当前包DEV_TO_FIRST_CURRICULUM_V1：** 从当前seed1701 checkpoint/state恢复到总计1,003,520 transitions，最多新增741,376（181个4096窗口），新增训练墙钟上限4h，先到为止。全历史开发总计最多1,011,712，保留每次额度与实际消耗。按79.3299 transitions/s估计新增训练2.60h，25%余量约3.25h；共享负载、阶段和内部验证可能改变吞吐，4h是停止上限不是承诺。用户已明确批准本科学预算；2026-09-13隔离作业已启动，正式3M仍未放行。

**当前正在执行的要求：**

1. 复用已验收恢复流程，保留actor/critic/optimizer、seed1701、262,144计步、算法状态、stage0和next_validation1,003,520；新sim显式reset并重绑最后观测，不声称恢复PhysX bit-state。world/reference/物理/源方法保持已封存版本。只允许必要extension额度/调度记录及已授权GPU运营接线，禁止重置旧账本、改训练配置、启用formal-approved或自动换seed。
2. 按D:/UCBP/GPU_POLICY.md做10秒资源准入及运行监督。GPU0/1和符合余量的共享均允许，不以存在PID拒绝；结合训练实测2621MiB及既有评测/初始化峰值预算，若内部验证与父进程共存须合计显存/RAM需求，不只套单进程峰值。准入后只在本任务子环境设置RL_ISAAC_NO_GUARD=1，记录物理UUID和cuda映射；不碰其他track或他人锁/进程。资源告警按该文件在本任务安全边界保存退出，未授予无限重试。
3. 连续学习，不因stage0、0成功、接触或单窗口波动中断。到总步524,288作一次中间交付（进度消息，不等待reviewer），运行同口径固定25 stochastic+25 deterministic stage0诊断；这50回合只测量，不改变curriculum。沿用正常终止、固定seed/采样seed。保留既有前2回合逐步诊断，对其余接触回合只增加首接触step/link的轻量记录，不另开碰撞认证或全套视频制作。
4. 到总步1,003,520完成最后完整更新及既定25 stochastic curriculum验证，由原>0.95规则决定stage，随后结束本包；不跳过、不人工提前、不因验证失败延长。若4h先到，则保存最后完整update，明确未到既定验证。最后模型再做同口径双模式50回合stage0开发诊断，与已封存262,144评测及本包中点比较；固定诊断合计100回合、墙钟合计限20min，既定curriculum25回合计入训练墙钟。新stage若刚设置但未继续训练，必须明确说明，不能声称训练过该阶段。
5. 交最终policy/state、恢复边界、额度/实际步数和墙钟、同分母左右误差/联合通过/提前接触/臂桌接触、curriculum事件及源码/world/hash。只选最后完整update，不从中点择优。非finite、身份漂移、控制接线错误或资源条件不满足是停止原因；低分按计划交付。输出新development_curriculum_<id>产物，更新现有计划/状态，不改其他track。
6. 到包末若联合仍0且固定配对误差趋于停滞，不自动再追加同配置；若有通过或明确改善也不自动启动3M。reviewer据这一次完整里程碑决定后续方法与正式可比实验。正式放行不以“必须先成功”作为发表低分结果的隐性筛选，仍需共同runtime/物理版本和可执行评测合同；开发seed1701不能改名成正式seed0。最终评测/整理/写作至少保留提交前18h，本包不能滚动延长侵占该时间。

**启动记录：** 新runtime为`wp4_20260912_01/curriculum_runtime_20260913_01`；约10秒准入选物理GPU0（UUID`GPU-1d992e9e-0787-f30e-3b2b-b5ea38a7897a`），进程内cuda:0，首个完整更新266,240。部署包SHA256=`7f3c83bc34a06711dd3f4fd933fe7e30a201d2dbcec756bd7b11ced118a82eb7`。作业在运行中，本段不预写最终结果。

### 历史交付：DEV_EXTEND_262144_V1已批准并执行完毕

2026-09-12 reviewer读取development_20260912_01报告后批准DEV_EXTEND_262144_V1；implementation现已完成[追加开发交付](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/DEVELOPMENT_EXTENSION_REPORT.md`）。seed1701从57,344恢复到262,144，新增204,800 transitions / 2,581.625秒训练墙钟，达到步数上限；旧65,536开发额度与本次extension额度均不得重复消费。固定双模式首尾评测、实际policy臂桌接触、曲线、checkpoint及恢复边界均已保存。确定性边界误差中位右/左261.46/285.82→96.85/163.68mm，stochastic 299.53/253.51→195.76/218.96mm，但联合预抓取仍0/25、训练累计0/7,343，且stochastic臂桌接触1/25→9/25。接收为开发证据，不放行正式3M，不自动追加相同配置训练。

**研究判断：** 57,344 global transitions=14个4096窗口，不是充分训练失败证据。40步边界回合右误差中位从342.77降至236.58mm（约31%），左257.23→227.43mm（约12%），有噪声且末段仍有变化；这不是同一固定测试集的因果效果，也不是任务成功。0/1627仅为训练回合联合预抓取结果，不能冒充最终policy任务SR。提前接触不是臂桌接触；本轮没有测policy的臂桌接触，因此不能直接沿用固定q回放的碰桌结论。内部验证仅在4096执行，下一约定点1,003,520，stage始终0也不是14次独立验证失败。保持当前方法比立即加桌约束IK更有依据，但不保证多训能解决源噪声/物理问题。

**已执行包：DEV_EXTEND_262144_V1（以下保留为授权与验收记录，不得再次启动）。** 保持seed1701及当前world/reference/PD/reset/reward/augmentation/curriculum/学习率/网络不变，新world模型已从57,344继续至262,144，共新增204,800 transitions（50个完整窗口），新增训练墙钟2,581.625秒，先到步数上限；全部历史开发总账为270,336（含旧WP2的8192）。执行使用独立、显式development extension额度，没有使用formal-approved，没有复位旧账本。正式3M未启动。

**已执行时采用的完整要求：**

1. 首先对已经选定的57,344最终checkpoint做stage0诊断评测：既定25个seed100000..100024、stochastic模式和正常终止；另外相同25个seed deterministic模式，两个模式分开记录，不能挑较好模式报主结果。绑定现有新world，记录到40步率、左右边界误差、联合通过、提前接触及终止时刻；对前两个固定编号回合额外记录实际关节/目标与前臂—桌pair接触，可复用已有传感器工具。不重做接触正反例或拍摄全套视频。所有50回合是开发诊断，不更新curriculum、不用于最终任务SR。发现确切错误接线/身份不符先停修，0成功或碰桌本身不成为再次等待reviewer理由。
2. 复用同一最终checkpoint的actor/critic/optimizer、计步和已保存训练状态继续学习；使用原resume支持，检查PPO计步57,344、stage0、next_validation1,003,520和world身份；新模拟器显式reset后使SB3最后观测与新状态一致，保留optimizer/算法RNG及记录实际恢复边界。不得拿旧sim obs接新reset、不得伪称PhysX bit-state无缝恢复。若现有恢复实现不能安全对应，报告一个具体接线缺口，不自行从头重抽seed或复用旧world。
3. 不改原curriculum调度：262,144前没有下一次既定百万步验证，预期stage0不能被当成新增失败证据。连续执行到新额度或明确运行故障；保留同口径4096窗口统计，记录训练回合中实际联合通过，而不是根据调度尚未触发判定没有通过。不能按reward变化选择checkpoint或改方法。
4. 对最后完整update重复上述固定双模式50回合，和57,344 checkpoint做同编号配对比较。首尾两次评测合计最多100回合，新增评测/启动墙钟总上限20min；若到时未完成，保留部分、明确分母，不越额或隐藏失败。展示左右误差/终止构成变化及实际策略碰桌证据，尽量分清均值动作质量与随机探索。全包预计约60–80min加GPU排队；不把排队当训练时间，不干扰其他作业。
5. 同包交最后checkpoint、extension预算账本/恢复边界、首尾固定评测原始回合、学习曲线、接触记录、命令和hash，放新的artifacts/pour17_v1/development_extension_<id>。如无通过且配对误差不改善，停止追加相同配置训练，下一次reviewer根据实际policy证据决定一个必要适配；有改善也不自动启动3M。正式结果仍需共同物理合同和独立正式训练/评测；该开发扩展不能包装成正式seed0。

**执行结果（2026-09-13收尾）：** [报告](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/DEVELOPMENT_EXTENSION_REPORT.md`） / [学习曲线](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/extension_learning_curves.png`） / [固定回合配对图](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/fixed_endpoint_paired_errors.png`） / [机读合同](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/extension_contract.json`） / [哈希清单](artifacts/pour17_v1/development_extension_20260912_01/delivery_manifest.json)。最终checkpoint SHA256=`92868d6525087ae36da2acb09982432af074d4a6d2c5208a40de7f4039b19483`，state SHA256=`6de5b4073d0b9ea9effb0b7323fc5cb8fda8d7ac8e9fd77479437d3ca38c08c5`。四次固定评测命令墙钟合计124.310秒；训练和评测均正常结束、无遗留ViViDex进程。结果显示误差改善但没有联合通过，不能自动放行3M或选择新的方法；等待reviewer决定下一包。

### 已执行完毕：接收续包，直接进入有限开发学习（历史裁决，2026-09-12）

**本入口已执行交付（2026-09-12）：** [开发学习报告](README.md#archive-access)（归档：`artifacts/pour17_v1/development_20260912_01/DEVELOPMENT_REPORT.md`） / [学习曲线](README.md#archive-access)（归档：`artifacts/pour17_v1/development_20260912_01/learning_curves.png`） / [哈希清单](README.md#archive-access)（归档：`artifacts/pour17_v1/development_20260912_01/delivery_manifest.json`）。train/internal_eval已接同一WP4 overlay且实际根/canonical读回一致。从头seed1701同run完成8192里程碑并继续到57,344新增步，751.205秒，按步数预算停止；累计开发65,536步/895.266秒。1627训练回合联合预抓取0通过，40步失败1185、提前接触442，stage0；4096更新处内部25回合0通过。保存/加载检查通过不代表任务成功。未实施桌约束IK或改变物理/目标，未新增固定动作诊断；正式3M不启动。以下执行规则保留为本次授权记录，不是再次启动相同预算的许可；等待reviewer下一包。

**裁决：接收来源核对和固定动作受阻证据；暂不批准实施OWN_APPROACH_TABLE_CLEARANCE_V1；放行冻结当前方法的新world有限开发PPO。** 下方旧续包及“三项学习条件”保留为历史，当前不再要求先证明一条通过40步门槛的开放环路径。这是reviewer修正开发放行条件，不改变仿真内2s/50mm终止、奖励、curriculum或正式评价合同；不是宣告碰撞解决、存在已认证路径或正式baseline就绪。

**理由：** 续包来源表、approach_blocker/result、env四项hash与交付一致；原MANO+W1已经有桌内点，没有确认需要修复的传输错误。q40理论六点18.53/28.19mm仅说明空间误差可小，仍未认证无碰撞。三种已测固定动作受阻是真实风险，不能证明整个58D策略空间无解。源码env._pre_physics_step直接将actor动作映射为关节target，_get_observations/_get_rewards用空间目标，train的初始均值来自canonical home，不读取approach IK q作为跟踪项/示范。因此只改0..40的IK q、保持所有目标/动作/初始化不变，不会把避桌能力传给当前PPO；提案主要增加可行性回放证据。25mm硬裕量和整段扫掠认证也不是既定任务成功条件，不应继续升级为开发学习前置。

**本包可直接转达implementation：**

1. 不再重做MANO/W1来源、V10对拍、固定动作或全程碰撞认证；不实现上述桌约束IK。保持WP4根(-.75,0,0)、canonical reset、PD/材料/碰撞、2s时长、原hand/object目标、W1/S1/A/mean、58D动作和原reward/curriculum。接收桌内目标噪声与接近碰撞作为本次开发试跑的显式未解决风险，不能写成修复或可行性PASS。不加BC、成功轨迹初始化、参考残差控制或对方q。
2. 先完成必要的新world训练入口接线，工程目标≤45min。当前train.py的build_env只调用isaac_scene_config，未调用WP4 apply_root_overlay；其internal_eval子进程也未传overlay，不能直接拿新参考跑旧根。复用wp4_world.apply_root_overlay，在train.py和internal_eval.py等本次实际使用的建环境入口增加显式`--root-overlay`并传递同一个路径，必须在创建Articulation前应用。训练、内部25回合验证、保存/加载校验绑定相同root overlay/reference/source/physics身份；创建当前训练进程时读回根与canonical并确认无错版本。允许这些必要接口接线和相关回归，不重建方法或改变旧artifact。
3. 正常训练必须probe_only=False、external_evaluator_controls_termination=False、stage0起步，不能沿用WP4外部终止诊断模式或屏蔽提前接触/40步失败。内部验证沿用实际25回合/既定seed和curriculum规则，必须运行在同一新world；不伪造阶段升级。
4. GPU可用后直接启动一个从头development seed1701、32env、n_steps128的连续PPO run；不加载旧checkpoint，不传formal-approved。先到8192完成更新保存并报告进展，若无真实执行错误就继续同一run，无需reviewer再许可。最多新增57,344 transitions或训练墙钟3455.939秒，先到为止；保留最后完整update、实际消耗和内部验证耗时。物理诊断历史700/2048，剩余1348仅供必要短接线读回，不是要求用完的额度，亦不取代PPO预算。
5. 训练中优先记录左右pregrasp六点误差分布、40步边界到达/联合通过率、提前接触终止比例、阶段、episode长度、吞吐和可取得的臂桌接触；按已有更新窗口报告，不能用总reward上涨替代任务进步。0成功、stage0或仍有接触本身不触发停训；非finite、爆炸、根/reference身份不符、控制输入未生效或计费/资源冲突等明确故障立即停。不得为改善曲线改参数/目标/seed或关闭检查。
6. 同包最终交付代码接线diff/hash、冻结world/reference/有效参数、最后checkpoint和准确加载命令、实际预算、学习曲线及失败分布，放新的artifacts/pour17_v1/development_<id>并更新现有文档。试跑卡stage0且无改善也如实交付，不追加第二条IK路线或自动扩正式3M；下一次reviewer依据真实学习结果决定物理/路径/输入适配，而非再等全套回放通过。此处只发布执行授权，reviewer未启动仿真/PPO。

### 已交付的旧执行入口：WP4交付后的续包（历史，当前以上节为准）

**本入口执行交付（2026-09-12，待方法决定）：** [续包报告](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_followup_20260912_01/FOLLOWUP_REPORT.md`）。源frame0/30直接MANO world→W1→最终/运行目标无分歧；左30仍有桌内点，另用保存的正常左81对照，无额外解码。未发现生产接线bug。自身q40纯FK六点18.5/28.2mm，但新增两槽40步固定动作（直接保持/关节插值）均有末步R_l5/L_l6—桌接触、误差约120/129mm；直接保持末5步关节变化≤1.54e-5rad。当前自身接近可行路径条件未满足，不上升为全局不可达。新增80槽，总700/2048；PPO尚未启动，学习余量未消耗。唯一[方法提案](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_followup_20260912_01/minimal_revision_proposal.json`）为自身approach臂IK增加桌分离约束，根/reset/PD/2s/目标/门槛不变；未实施，等待reviewer。不是因全轨迹IK失败或0成功而暂停。

本入口供用户转达implementation，优先于H下方历史WP4 prompt。目标是修复确定影响训练的目标/接线错误并进入有上限开发学习，不重做已完成WP4。上一轮直接“不追加PPO”对应已交付的上传/连续短测；本续包采用下方条件学习授权，满足条件后不再等待一次PPO许可，正式3M仍未放行。

1. 阅读两级AGENTS、REPRODUCTION_STATUS和本入口，按H现有来源对照规则追踪源frame0及一个正常帧的MANO world→W1→最终目标→实际reward时刻；先用已有数组，必要时仅解码这两帧。确认的单位/侧别/索引/重复变换/观测接线bug可在本包直接修，回归受影响路径并生成新版本。不得把来源噪声或已批准工程约定静默改成另一种方法。
2. 接着只判断当前2s接近阶段是否有可行状态/路径。复用WP4接触和分层trace；必要短测沿用2048槽转移总上限，已用620，剩余最多1428，不另开完整回放认证。不因全轨迹IK失败、回放抽动或0成功单独禁止学习；若有确定阻断阶段的碰撞/不可达问题，则交首次阻断位置、证据及一个最小修订提案。不得擅自改根/reset、PD、摩擦/质量、碰撞、2s时长、A/W1/S1或奖励/阶段规则。V10差异已经定位，不再重做对拍，不导入其成功q，不把历史V10认作当前共同物理合同。
3. 满足H下文三项启动条件后，显式绑定WP4新根及本轮冻结参考/源码，从头运行development seed1701 PPO；不加载旧world checkpoint。先8192，再继续同一run至本轮最多新增57,344 transitions或训练墙钟3455.939秒（1h减去WP2已用144.061秒），先到为止，实际新增量实时扣账。不得首8192仅因0/25就停下等reviewer。训练过程中不改变版本/目标/方法；非finite、物理爆炸、错误接线/版本漂移等真实故障立即停。
4. 源头定位工程目标1h；新增工程活动最迟2h给出“已开始开发学习”或“一个确切阻塞及其最小修订”。GPU排队单列，不干扰其他作业，不把等待当完成；没有硬阻塞则按条件自主推进。仅改本baseline adaptation/pour17_v1、必要configs/pour17_v1及既有计划/状态，产物放新的artifacts/pour17_v1/wp4_followup_<id>，不覆盖WP4交付或其他方法目录。
5. 最终合并交付：来源首分歧/无分歧结论及相关差异；接近阶段可行性证据或具体阻塞；若训练则交实际步数/墙钟、checkpoint与加载入口、左右pregrasp误差/通过率、终止分布、阶段和吞吐曲线；源码/参考/world/物理参数身份与hash。仍可能得到0成功，诚实报告。不重做decoder/FK/A代数验证、已通过接触/奖励审计或整套视频制作，不以对方runtime未上线阻塞本次开发试跑。此处是执行prompt，不是任何步骤已完成的记录。

**WP4短测执行入口（2026-09-12，非下方新增学习完成记录）：** [报告、视频及hash](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_20260912_01/WP4_REPORT.md`）。两槽固定根/重复reset/静态47.725mm分离通过；自身参考重建并完整连续执行271样本。实际step40六点误差右138/左196mm，存在index29起l6—桌动态接触；未改目标或PD。620槽转移、新增PPO=0。下方学习补充在执行期间出现，本轮最后的直接上传指令明确“不追加PPO”，故来源对照/开发学习未执行，不误标为已完成；保留该补充供下一执行决定。另一方接口见[共同world交接](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_20260912_01/COMMON_WORLD_HANDOFF.md`），对方读回仍pending。

### 当前执行收敛：优先训练阻塞，同包进入有上限学习试跑

**历史V10低层回放对照，2026-09-12只读结论（不混用V12）：** 用户提供`reports/H2S2R_POUR17_SUCCESS_REFERENCE_20260910/current_runtime_source`及`reports/H2S2R_STAGE_B_RIGHT_LOW_LEVEL_REPLAY_ENTRY_FIX_20260910`。这是成功源58关节命令236control回放，绕过PPO/FABRICS，不是H2S2R自主policy成功或全程无碰撞认证。抽查5份V10源码hash与identity一致；ViViDex env/scene/geometry/wp4_rollout四份hash与WP4交付一致。已通过只读SSH读取用户指定远端canonical_reset_v1.json，SHA256=d41cd5d6ab19f55d583dc3fcd458db89b48dc52e069e8d1be1d61e60ad64e679，与V10归档和WP4 base canonical相同。

| 比较项 | 历史V10低层回放 | ViViDex WP4 | 判断 |
|---|---|---|---|
| canonical reset关节 | q0/qd0来自同一canonical，按58个名字顺序核对 | 两槽首次reset读回与V10 q0最大差0rad | 没有“V10预先抬肘”的reset差异；右臂约[-45,-45,0,-90,0,0,0]°，左[45,45,0,-90,0,0,0]° |
| 固定机器人根env-local | inherited dexmate_env_cfg.py设置(-.5,0,0)，单位旋转 | 已批准短测(-.75,0,0)，单位旋转 | 差25cm；不能把canonical中的旧占位说明当实际spawn证据 |
| 实际arm_center | 用trajectory.npz的实测q和同名腕pose反算约(-.583613,0,1.278434)m | 实测(-.833613,0,1.278434)m | V10第0/40/178步、两侧共6次反算相对WP4均为(+.25,0,0)m，位置分量偏差<0.001mm，旋转矩阵差<1.2e-6；这是基于现有URDF的离线锚点推断，不是新PhysX根读回 |
| 桌与腕坐标接口 | 桌1.2192×1.8288×.04m、顶面.87m；左右hand_C_MC，减env_origin，wxyz | 相同 | 未发现这里存在高度、link或四元数顺序差异；不等于MANO目标链已正确 |
| 臂PD | 回放显式configure_physics，运行记录j1 kp/kd=9765.2/39.9，j5=286/4.3 | WP4绑定源码j1=28062.54/114.592，j5=7579.84/114.592 | 不同动力学；H2S2R有PhysX读回，ViViDex此处为交付源码证据，不能说控制器参数相同 |
| 摩擦/质量 | table/body/pad摩擦1/1/1，两个物体质量各.5kg，物体摩擦1、average组合 | 源码table/body/pad摩擦.5/.2/3，瓶质量.53kg、摩擦3、multiply组合 | 不能以相同USD或黑桌推断物理合同一致；未改任何参数 |
| 接近时长与命令连续性 | 入口标记0..178为approach，约8.95s；236样本臂命令最大相邻增量2.594° | approach40control=2s；最大47.535° | 路径和时长不同，不能用此对照单独判定controller或坐标bug |

**对当前执行的影响：** 先保留已通过静态分离的WP4根及共同reset，不凭历史回放改回-.5或抬肘；优先收敛自身hand target来源和2s接近阶段的实际可行性。若目标链无误而路径确实被桌阻挡，下一方法候选应是自身臂接近路径/肘部姿态的最小适配，而不是借用成功q；改变共同reset、根、PD/物理或既定接近时长均须明确决策，本次对照不授权这些变更。最终共同runtime需要把实际PD/材料等纳入一致性，不能仅对齐根/桌几何；不把历史V10直接宣布为当前权威Ours runtime。已查看历史side首帧及step200 side/top抽帧，视角有遮挡；trajectory只有手指接触等数据，没有全程前臂—桌pair记录，保留未认证边界。未导出/引入对方命令到ViViDex，未执行仿真或PPO；不增加全面回放认证为开发学习前提。

**WP4交付后的抽动解释与优先级澄清（只读复核）：** 完整trace表明39→40左臂j3的参考命令跳47.535°，同期左腕位置目标增量7.726mm；102→103左j7、103→104左j4命令分别跳42.922°/43.358°。这是20Hz参考q回放加原PD，不是学习policy输出；index29起又有前臂—桌实际接触。不要将其笼统归为坐标错误或PD bug，也不要以“先把完整回放变平滑”新设训练门槛。现有actor自主输出关节位置，并不强制跟随IK q：关节回放跳变与训练手部reward目标错误必须分开处理。沿用下方来源对照和当前阶段可行性的有限范围；只有发现平滑、可达、无阻挡命令下仍异常振荡，才追加针对性的PD/执行器检查。不得凭视频直接调PD、删碰撞或改变参考时间/奖励规则。此补充没有启动新作业，也没有把本次明确“新增PPO=0”的交付追认成学习已执行；正式3M仍未放行。

**最新用户要求与reviewer取舍：** 不再以所有问题解决/所有验证完成作为训练前提。本补充优先于H下方原WP4的统一停止/新增PPO=0安排；不新开一串WP编号，不打断当前根修订/参考执行。范围收敛为“world不再物理失真、hand target来源/变换正确、当前阶段存在可学习路径”，其余验证后移。已获准的共同根、A/mean、W1/S1、reward/curriculum和G1–G4均不借此改变。

**当前本地中间产物（不是WP4最终验收）：** `wp4_20260912_01/static_separation_gate.json`报告根/固定躯干分离通过；`reference_build_v1/reference_rebuild_manifest.json`记录新world下右/左IK P95位置误差约12.62/37.47mm（旧约24cm），step40约2.046/5.159mm，S1 index143左约3.342mm（旧约239.42mm）。仍有右30/左92个样本未过5mm/角度IK容差，但不能只按失败计数禁止训练；这些不是实际手跟随误差。这个结果支持共同根是大量旧腕部残差的共同原因，不支持它解释未移动的桌内hand target。

**hand target的定位重点必须纠正：** 当前训练的十点目标来自MANO landmarks，经局部A及配套逆变换的world root重组，不是NLopt输出q的FK，也不是arm IK的FK。源码源帧关系为 `p_S=R_A p_H+t_A`、`R_WS=R_WH R_A^T`、`t_WS=w_world-R_WS t_A`，因此正确组合时 `R_WS p_S+t_WS=R_WH p_H+w_world`，再做W1。A在该点目标回投关系中抵消；这不认证A的跨模型解剖质量、掌心定义或插值实现，却足以说明不能无证据地把“目标点被推到桌里”归因于手指优化器。第40个control step的核心首样本对应源视频frame0，不是视频frame40；S1合成尚未在这个起始样本发生。H2S2R object trajectory正常不等于hand target有正确性对照，不作为oracle或对方robot reference输入。

**当前工作包只补一次直接来源对照，工程目标≤1h，可与已授权物理步骤并行：** 针对源frame0的异常点及一个既有正常对照帧，记录①原始MANO世界点；②直接施加已批准W1后的点；③当前局部→world→S1/重采样输出；④对应实际六点reward目标/时刻。使用相同原始输入、单位、手别、原始帧；优先已有数组，缺少必要原始世界点时允许只为这两个源帧调用既有已验证decoder获取值，不重跑decoder/FK/A整套验证，不重解手指。表中保留原始坐标和变换来源、每段z/误差、杯/瓶口及桌面作参照。新解码/数据仅用于诊断，不自动覆写原参考。

- 若②与③不同，定位首次分歧并直接修索引、单位、重复变换、左右手、时间或reference/观测接线错误；只回归相关路径。
- 若②已在桌内而③一致，问题位于输入手—物体几何与已批准world放置组合，不能继续盲改A/IK，也不能从object trajectory正常断言MANO准确。区分已知近似噪声是否可接受与是否需要输入合同/目标方法修订，提交一个具体选择，不再追加泛泛审计。
- 某个目标点在桌内不自动等于5cm六点均值门槛物理不可达；不用精确零误差或所有点离桌作为额外启动门槛。但若进入既定学习阶段要求持续穿透/目标整体不可达，应先修原因或提出具体方法决定，不能靠调门槛掩盖。

**只保留三项学习试跑启动条件，implementation依据证据自行执行，不等待额外审批：** (1)已批准根的真实放置/固定躯干分离成立，当前阶段无已知破坏执行的物理错误；(2)上述直接来源对照排除未处理的单位/坐标/帧号/侧别错误，实际六点reward与reference同一时刻，输入没有对方q；(3)现有IK/连续PD的关键帧证据表明存在接近目标/抓取的可行状态或路径，未发现既定pregrasp门槛必然不可达，动作/观测/reward finite且关键接触读数有效。无需所有IK样本达到5mm，无需open-loop完成任务，无需已有自然curriculum升级或正G2–G4，无需等待对方runtime上线、完整恢复演练或全部新版视频。不能以“尚未证明不可达”单独代替可行性证据。

**条件满足后，同一工作包直接进入连续开发学习试跑：** 使用冻结的新world/reference/source版本从头建PPO（development seed1701，保留seed标签），不加载旧world checkpoint；先8192步确认新版更新/保存，再继续同一次run至当前剩余额度。沿用既有开发总上限65,536 transitions或1h训练墙钟，WP2已8192步/144.061秒，故本次最多新增57,344步，时间与后续实际新增用量累计扣除。不得按试跑结果换seed、关闭curriculum或改目标/奖励。过程记录pregrasp双侧误差分布、通过率、episode长度/终止原因、接触和每次真实阶段更新，用学习曲线判断是否在进步；即时低SR或0/25本身不暂停。出现非finite、物理爆炸、输入版本漂移等真实错误才立即停止。

试跑冻结的同一版本可以继续收集该run数据；并行只做日志/评测/文档工作，不能边改训练方法或reference边把checkpoint合并成一条曲线。试跑为开发证据，正式seed0从头3,002,368预算和最终512不变，未自动放行；若试跑无实现错误且有有效学习信号，则下一次评审直接决定正式启动，不再补一串小验收。到额度仍卡在同一失败阶段且无可见改善，提交主要阻塞与一个方法备选，不宣布方法本身失败。

**明确后移：** 剩余局部几何零误差/限位频率优化、同事包完整左右手对拍、全部轨迹零碰撞认证、每种增强样本全面可达认证、已通过reward/接触/保存重载的重复测试、完整simulator中断重演、额外相机和历史视频补录、最终512及论文级整理均不再阻塞此开发试跑。最终评测正确性和共同world一致性仍是论文比较前要求；保留已有验证并按实际相关故障恢复检查，不能把后移写成永久不需要。

**时间止损：** 沿用WP4剩余工作，不重置工程时钟；来源对照目标1h，最迟新增2h给出“已进入连续学习/一个明确训练阻塞”而非又一份组件清单。有真实阻塞就集中报告；无阻塞按上面条件直接学习。本轮reviewer只改计划文档，不启动训练或替implementation改代码。

**Reviewer决定：批准 `CW_ROOT_NEGATIVE_X_075_V1` 进入隔离短验证，以及双方各自的臂参考重建；不放行正式3M，不把该候选标记为已验证的正式共同world。** 用户本轮明确将这项决定交reviewer，故不再请求同一项批准。此前禁止改变根放置的规则仅对本段精确候选作例外，其余边界保留。旧登记bundle和历史产物不回写；本轮reviewer只读与改文档，没有执行候选、修改功能代码或启动仿真。

### 接收的证据与研究判断

- 接收WP3“登记场景真实相交、公共reset未发现接线偏差”的结论。固定l2/l3有visual三角形相交、启用的cooked collision证据与两槽持续pair接触；不再归为相机遮挡。首次/重复reset的q/qd/target及root/body一致，不能再让implementation重复寻找已排除的reset bug。reviewer查看侧/俯视首帧，并抽查提案、几何/重置/腕系报告及场景脚本5个hash均符合交付manifest；没有重新运行WP3，也未重复核验全部176个旧文件。
- `-0.75m`由固定躯干包围范围沿负X分离后选一个取整值，报告两槽预测最小间隔47.725mm；其选择不依赖task SR。该证据足以支持单一短测，不保证实际cooked形状/克隆后的分离、手臂扫掠安全或可达性。不给偏移网格搜索或按分数调root的授权。
- 此修订修正共同场景，保持ViViDex输入/目标方法。不能据此说所有旧参考误差都由躯干碰撞导致，也不能把旧world训练结果作为正式可比数据。左右两方法的policy和reference来源继续隔离。
- 特别保留：原step40左ring/pinky落入桌体的目标几何不因机器人平移消失；W1/S1操作目标、A、local finger q均不变。WP3左腕解释对照仅约3.43°/9.58mm差异，没有简单缺失180°的证据，本轮不换同事的腕系/DexPilot或手指目标。

### 精确批准范围与双方接口

共同机器人env-local固定根位置为 `(-0.75,0,0)m`，旋转保持原值；组合后的固定world anchor为 `env_origin+(-0.75,0,0)`，只施加一次。允许在隔离配置/派生world层正确接入spawn与固定约束；不能把视觉prim挪走而保留PhysX固定锚点，不能reset时另行传送或每步强制写root。

保持桌、物体初态/物理、58个canonical q/qd、控制周期、PD、碰撞过滤、G1–G4不变。机器人home的世界位置随根移动，W1和自身操作段的世界目标不移动；以新实测arm_center/home重新解双方各自的arm IK，重接2s approach/return，并重建依赖旧anchor的G2认证IK。finger-local参考及十点retarget保持。新artifact必须绑定新world版本、base bundle hash、root overlay、canonical规则及实际readback；actor/critic禁止读取对方q/reference。

双方使用相同**共同world字段**和实际读回，而非强求方法源码/reference的hash相同。ViViDex implementation仅改本适配，另一方法由其implementation在自身隔离副本执行；不得顺手修改/停止对方当前运行。ViViDex短测不必等待另一方法上线；另一方状态缺失时报告pending，不能宣称双方共同runtime已冻结。双方最终比较均须在通过验收的新world从头训练/重新评测，不能复用旧分数或自动续训旧world checkpoint。

### 可直接转达给implementation的WP4 prompt

```text
执行WP4：按BASELINE_PLAN H批准的唯一候选CW_ROOT_NEGATIVE_X_075_V1，完成共同根接线、两槽短验证和ViViDex自身双臂参考重建。另一个方法由其implementation执行相同world合同与自己的参考重建，不共享机器人reference。这里已获reviewer批准，不再等根候选许可；正式3M仍未放行。

先读两级AGENTS、REPRODUCTION_STATUS及H，复用F的W1/S1/训练机制。输入为wp3_20260912_01的提案、几何/reset/readback和wp2_20260912_01的已绑定自身参考、URDF、源码。所有输出放artifacts/pour17_v1/wp4_<id>/；仅修改本适配及configs/pour17_v1，派生world overlay/canonical文件放新产物，不改原bundle、上游或Ours。先保存源码快照和hash，现场核实共享GPU资源，不干扰他人。

1. 只实现root env-local=(-.75,0,0)，旋转不变。结合Isaac固定根解析验证实际锚点，不能仅设置一个cfg字段就当成功。新版本显式绑定base bundle+root overlay+canonical root规则，旧root=0仅保留为旧版本。不得附带改PD/碰撞/物体/桌/reset q、A/mean/手指限位/W1/S1或随机化范围。
2. 使用WP3已有读回和几何工具做两槽首次reset→home target保持→再次reset；在无policy动作下确认env origin只加一次、固定根和所有固定body相对旧状态平移-.75m、旋转/q/qd/target保持约定，桌与物体未被一并平移。检查实际fixed anchor、两槽隔离及相机侧/俯视。固定躯干须无visual/cooked collision交叠且原pair持续接触消失；几何裕量实测应满足提案>20mm目标。不得用缺失query或零contact单独宣告分离，WP3已知移动body query不可信的路径不可复用为动态无碰撞证据。
3. 读回新canonical arm_center/home wrists/hand points，重新建立自身IK anchor并生成完整自身arm参考和前后缀（G2认证IK也使用新anchor）。不平移操作段世界目标、不投影失败目标、不导入同事/Ours的q，也不改变当前solver预算和初始化规则。交新旧目标/手指内容一致性和新arm结果，统计两侧可达误差、限位、失败帧、完整horizon及桌内目标警告。
4. 根/静态几何通过后，在同一双槽环境连续PD执行新的自身参考，重点step40和S1 index143并覆盖完整271样本序列及返航；保留目标→IK FK→实际q FK→Isaac腕、六点误差和臂/手—桌面的实际接触/可视证据。允许必要GPU动态几何读回，不能借不可信scene-query确认安全。同步报告旧桌内手目标是否仍影响接近、接触或5cm门槛；不得静默抬高目标或放宽门槛。展示手-自身物体/物体-桌传感器在新world下仍有明确正负证据，复用已验收接线，不重做整个奖励审计。

交付一次：wp4_report.json/MD、共同world字段/版本/hash、reset_world_readback、scene separation证据、reference重建manifest、分层trace和短视频、相关代码diff/hash、测得墙钟及主要残余阻塞。为另一方法提供同一root/canonical合同和readback字段说明，其implementation只在自身副本生成自己的参考/报告；未知对方运行目录不得猜写。ViViDex不必等对方结果才交付。

时间估计1–2h接线与重建，短物理验证约15–30min，整个工作包上限3h。默认两槽、物理诊断累计不超过2048槽转移；失败先修具体接线错误，不把换root参数当修复。根/固定锚点不符合合同、仍穿桌、明显新臂/手碰撞或参考失败均交具体证据；未经新决定不换第二个偏移、不新增方法。静态根验证通过即可继续依赖它的参考步骤，无需中间审批。新增PPO=0，正式3M和最终512不运行，旧开发checkpoint不续训。整包完成后一次提交训练就绪/残余方法决策材料。
```

下一里程碑依据：根修订是否真实且无重复补偿、参考及实际执行是否达到可训练条件、另一方法共同world绑定是否可验证、剩余训练/评测窗口是否足够。static PASS不自动覆盖训练门槛；没有新矛盾不再重复decoder/FK/A旧审计。若只有world通过而桌内目标/腕残差仍显著，下一决定针对该剩余问题，不自动撤销正确的场景分离或扩大PPO预算。

## 历史规格与审计记录（非 active plan）

以下旧 §1–§5 保留来源/技术细节。其“下轮只做某个诊断”“不进入下一阶段”等排期限制已被上方deadline plan取代；事实和输入隔离仍有效。批准状态以上方F节及文档开头最新用户决定为准；批准后另记新版本，不回写旧PASS。

## 1. M0 审计结论

### 1.1 版本与运行来源：BLOCKED

本机和远端均存在多个相关 checkout；分支不同或工作树脏本身不构成错误，但也不能
自动代表正式实验版本。以下记录均由只读 Git/文件检查得到。

| 候选 | 路径 | branch / HEAD | Pour17 相关未提交状态 | 可得结论 |
|---|---|---|---|---|
| 本机 RL-Correction | `D:/UCBP/RL/RL-Correction` | `Step2_NoisyRecon` / `03b1a8317befc7abbe236f0e35fd5e6d2c87168b` | 未见 `tasks/pour` 变更或该任务目录 | Step2 重建 checkout；不能据此推出 Pour17 runtime。 |
| 本机 Step4 worktree | `D:/UCBP/RL/Worktrees/RL-Correction-Step4-Publish` | `task5/pour-bimanual` / `ff542749c9b1a7244dbbf615bdd82c9a9fd5ebfc` | `tasks/pour/{README.md,dexonomy_adapter.py,reference.py,tests/test_dexonomy_adapter.py,tests/test_reference.py}` 修改；`stage_reconstruction.py`、`step2_adapter.py` 及其测试删除。相对 HEAD：89 insertions、904 deletions、8 files；`git diff --binary -- tasks/pour` 的 SHA-256 为 `717feaebdda822c20343677cba817465940f865c65f81dd678f0aeea4e279534`。 | 仅为一个本机候选；HEAD 不能描述其工作树。不得因目录名 `Step4-Publish` 认定它是正式运行版本。 |
| 本机 H2S2R baseline | `D:/UCBP/RL/Baselines/RL-Correction-H2S2R` | `baseline/h2s2r-dexmate-single-side` / `ead03335336857cfb0585e6e4ee1ebcb36e98cb1` | `tasks/h2s2r_pour17/` 与 `rl_rebuild/baselines/h2s2r/` 为未跟踪的 baseline 实现文件。 | Baseline 候选，不是 Ours 训练来源的证据。 |
| 远端可访问 RL-Correction | `/home/kailang/experiments/rl_correction` | `Step2_NoisyRecon` / `03b1a8317befc7abbe236f0e35fd5e6d2c87168b` | 多个 reconstruction/retarget/HOI 文件修改，含未跟踪 adapter；只读搜索未发现 `tasks/*pour*`。 | 与本机同 HEAD 的 Step2 候选，且工作树也未清洁；不是可验证的 Pour17 runtime。 |

远端进程检查只发现 H2S2R monitor 服务及其历史 estimated-baseline monitor；未发现
Ours Pour17 trainer/evaluator 进程。交接包记录的 `/home/lyh/Project/RL_Correction`
在当前可访问主机不存在。可访问的 `/home/kailang/experiments/task5_diagnostics` 和
`/home/kailang/experiments/rl_correction` 也没有发现 `eval_best.pth`、`run_eval.py`、
`world_manifest.json` 或 `canonical_reset_v1.json` 的 Ours task runtime 副本。

**因此：M0-A「当前 Ours 权威世界/任务/reset/evaluator/运行版本」= BLOCKED。**
解除条件是由实验所有者提供或允许读取实际 run manifest/启动命令、源码根目录、已解析配置、
checkpoint、world/reset/reference/evaluator 的文件哈希，以及（如有）相关未提交 diff 或逐文件哈希。
仅给 Git HEAD 不足；实际使用未提交改动时，必须同时归档相关 diff 或受影响文件 SHA-256。

这个 BLOCKED 有且仅有两条可审查的解除路径，二者不得混写：

1. **历史运行复原**：定位实际 Ours train/eval 的启动入口、解析后配置、源码根和输出记录，并以
   world/reference/reset/evaluator 哈希及相关 dirty diff/文件哈希复原其真实实验版本。
2. **新共同 benchmark 冻结**：若历史运行无法追溯，明确把经哈希记录的交接包冻结为一个**新的**
   共同 benchmark；随后 Ours 与 ViViDex 都在该版本重新评测。它不证明交接包是旧 runtime，也不得
   继承或引用旧 Ours 分数。

### 1.2 可复核的交接包候选合同

本机交接包 `D:/UCBP/pour17_baseline_bundle_20260829.tar.gz` 的 SHA-256 为
`77edc03ca5be028fcbd8b6d978fd1a4072582b5c9904d603f5c86b29a70210f6`。远端解包目录
`/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17` 与本机归档
的下列文件逐一 SHA-256 一致：

| 候选组件 | 相对路径 | SHA-256 |
|---|---|---|
| 世界运行时清单 | `world/world_manifest.json` | `2339f7a9ca3eda294908e392744c798c06fbbbc0edd79250bd5e43aaa828dcd5` |
| canonical reset | `world/canonical_reset_v1.json` | `d41cd5d6ab19f55d583dc3fcd458db89b48dc52e069e8d1be1d61e60ad64e679` |
| v2 母带 | `reference/pour17_reference_v2__2ed81358__653rows.npz` | `a164d34e3ad0f86ca7aaae24f425a0f065474fe438fa2d2779c55edcaaa5865f` |
| 统一评测入口 | `evaluator/run_eval.py` | `40e6b157e836cb3ea3cd754791c1231ced7e9ef633c184d09a6ebe125f7d6d6b` |
| gate/死线本体 | `evaluator/progress.py` | `abdbfad4cda8c9f15ea96ec5feb781c27ee958ad575454f068272f433b6fb03e` |

该证据只证明本地/远端**交接包彼此一致**，并不证明其与当前 Ours run 一致。

候选共同世界来自 `world/world_manifest.json`：Isaac/PhysX，`physics_dt=1/240 s`，
`control_dt=0.05 s`、decimation 12；运行时 USD 是
`vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`（而非 `stance0803`）；58 个受控关节顺序
为 `[R_arm7,L_arm7,right_fingers22,left_fingers22]`，世界表面 `z=0.87 m`。候选杯为
object_0/Aux（0.150 kg、摩擦 0.5/0.5），瓶为 object_1/Object（0.530 kg、摩擦 3.0/3.0）。
这些数值在权威 run 对拍前均为 **candidate only**；ViViDex 不得修改它们。

候选统一评测为 `evaluator/run_eval.py` + `progress.py`：t0、无 RSI、deterministic mean、
512 episodes、外部 horizon 903 control steps。G1 为双手各 3/5 指垫 `>0.5 N` 连续 10 步；
G2 为 evaluator 发起的双腕世界 +Z 15 mm 认证（8/5/8 步），通过条件包含双物体上升
至少 5 mm 与手物相对滑移小于 8 mm；G3/G4 和 D1–D8 必须直接由同一 evaluator 执行。

### 1.3 共同输入、方法派生与禁用项

| 候选输入 | 分类 | 证据 | 本合同草案中的处理 |
|---|---|---|---|
| 原始 `pour17_raw_video.mp4`、相机 `K/c2w`、`world_fused.npz` 人手/物体重建 | 1. fair common input | bundle `README.md` 与 `make_pour17_perception.py` | 允许作为共同上游感知来源；不得把相机坐标直接猜作 Isaac 世界坐标。 |
| `*_mano_trans` `(142,3)`、`*_mano_rot` `(142,3)`、`*_mano_pose45` `(142,45)`、`*_mano_betas` `(142,10)` 和 `*_mano_valid` | 1. fair common input | `perception/pour17_perception.npz` 现场字段检查；生成脚本从 `world_fused.npz` 提取 | 可供 ViViDex 自行从 MANO 生成人侧关键点；左/右数据不得互换。 |
| object_0/1 位姿 `(142,7)`，单位米、wxyz、`gravity_z_up_world` | 1. fair common input | 同上，且生成脚本明确写入约定 | 可作为 object reference；有效帧处理尚未冻结。 |
| `left/right_hand_q` `(142,22)` 与 `left/right_wrist_pose_wxyz` `(142,7)` | 5. uncertain / needs decision | 生成脚本注明来自 `ref_qpos_{side}.npz` 的 DexPilot 重定向；`hand_joint_names` 是 right 22 名 | 不作为 ViViDex 的人侧输入或训练 reference；其是否可作独立共享-retarget 控制实验需另行决定。 |
| `reference/pour17_reference_v{1,2}*.npz` 的 arm q、finger q、confidence、cert_arm7 | 2. Ours-specific derived information | bundle `README.md` / `TAPE_DECISION.md` 明示 arm IK、Sharpa hand、confidence 与认证位形来自 Ours | 禁止用于 ViViDex 输入、初始化或 reference；仅 evaluator 读取其必要几何锚时按候选合同使用。 |
| object valid/scored flags（object_0 valid 63/142；object_1 valid 59/142；全部 scored） | 4. task/perception metadata，处理策略未定 | `object_valid_measured.npz` / `ovm_v1_20260826`；本次字段计数检查 | 可以保存为审计元数据；不得变成 confidence gate。丢帧、插值、剔除或保留全部帧必须在实施前单独冻结。 |
| ViViDex MANO→Sharpa 关键点/关节 reference | 3. ViViDex-specific derived information | 本规格 M1-A 的后续输出 | 只能由允许的共同 MANO 输入和冻结的适配器生成。 |

### 1.4 时间、缺测、坐标与左右手审计

- `timestamps` 全为 NaN；可验证的索引只有 `frame_ids=0..141`。生成脚本声明源视频 30 Hz、
  重建 15 Hz / 142 帧。**绝对时间戳 UNKNOWN**；在 M1-A 不得伪造时间戳。待对该采样率再核验后，
  可把 `t_s=frame_ids/15` 记录为派生的相对时间，而不是观测时间戳。
- 两侧手与 MANO valid 都是 142/142 true；这不替代物体有效性。object_0 与 object_1 的
  valid 分别是 63/142 和 59/142。缺测填补规则 **UNKNOWN**。
- 坐标元数据是米、wxyz、`gravity_z_up_world`；`c2w` 另外标为 OpenCV
  `x-right/y-down/z-forward`。重建世界到候选 Isaac env/world 的刚体变换未在交接包中冻结：
  **UNKNOWN，不得猜测或镜像。**
- pairing 已明确：left→object_0/cup，right→object_1/bottle。左 MANO keypoint 输出的
  精确 21 点名称/顺序、MANO 左右镜像约定和 MANO-root 到 wrist-link 外参，没有在该 bundle
  的可执行 schema 中给出：**UNKNOWN**。

### 1.5 warmup 与 G2 的共同口径

交接包 `evaluator/INTEGRATION.md` 建议评测 `POUR_HOLD_K=0`，同时记录 Ours 训练曾用
`POUR_HOLD_K=15`；文档明确要求双方确认。当前无法读取 Ours 真实训练/评测配置，故共同
warmup 值为 **UNKNOWN**。不得默认 0 或 15，也不得把物体钳位/认证偏移作为 ViViDex 独有帮助。

G2 的唯一候选口径是 evaluator 主动触发，而非 policy reward：以抓握站位为基准，双方均按
世界 +Z 15 mm、姿态不变的规范执行认证。是否已由当前 Ours runtime 采用同一实现仍为
**UNKNOWN / BLOCKED**。

## 2. M0 比较合同草案（待审核）

当且仅当 §1.1 的权威 run 对拍完成，下面合同可冻结：

1. 两方法使用同一经哈希确认的世界/USD、资产、物理、canonical t0 reset、控制周期和
   `run_eval.py` evaluator；不得改尺寸、质量、摩擦、reset 或成功阈值。
2. 共同最早上游表征为原始视频/重建、MANO 参数与物体 6D 轨迹，不是 Ours arm/finger
   reference。ViViDex 自行做 MANO→Sharpa 和后续腕/臂映射。
3. Ours 的 GraspPose、CuRobo guide、优化 robot reference、残差输出、confidence gating、
   `cert_arm7_*` 和任何已筛选 Ours 轨迹均排除在 ViViDex 输入之外。
4. 方法各自的 observation/action/reward 可以不同，但 evaluator 的输入只应为环境实测量。
   不把 `progress.py` 中 `W_OBJ/W_HAND/LEASH/MS_REWARD/WAGE` 等 Ours reward 常量移植到
   ViViDex。
5. 评测 warmup/hold 值、G2 调用实现、有效帧政策以及重建世界→Isaac 变换必须共同固定并
   在 run manifest 写明；未固定前不可进入训练或宣称公平比较。

## 3. M1-A 左 MANO → 左 Sharpa：施工规格（未实施）

### 3.1 原版 ViViDex 可复用部分与缺口

官方 MuJoCo checkout 的 `common/utils/process_dataset.py` 是已找到的人手/物体预处理入口：
它以 `ManoLayer` 从 DexYCB MANO 生成 3D hand joints、object translation/orientation。
运行时 `hand_imitation/env/models/reference.py` 和 `control.py` 只**消费**已计算的
`robot_qpos`、`robot_jpos`；`rewards.py` 再对机器人关键点做跟踪。

在 `upstream/vividex_mujoco`（commit `9790140170d8be49b828ab214ce3537a2475fcea`）和随附
SAPIEN checkout 的只读搜索中，**没有找到可直接复用、可执行的 MANO/hand-joints → Adroit
`robot_qpos` 重定向入口、实现或配置。**`process_dataset.py` 不是机器人重定向器。这只说明
公开 checkout 缺实现，**不**说明 ViViDex 没有定义重定向目标。

论文 §III-A、式（1）已定义原方法：对每帧机器人关节 `q_t`，最小化人手与机器人对应关键点的
位置误差，加上相邻帧 `alpha ||q_t-q_(t-1)||^2`；关键点为每根手指的 fingertip 和 middle-phalanx
位置，`alpha=4e-3`，首帧以关节限位中点初始化，并以 NLopt 求解。该定义是后续重建 retargeter
的 baseline 目标；公开代码缺的是该目标的可执行实现，而非方法授权。

已找到的预处理入口依赖 `torch`、仓内 `mano.manolayer.ManoLayer`、DexYCB toolkit、
`scipy`、`pyquaternion`、Open3D、trimesh 和原始 DexYCB 数据；这些依赖只能用于解释其
人手/物体预处理，不能补成缺失的 Adroit 重定向器。未来左 Sharpa 适配还需要经权威世界
确认的左手 FK/URDF 或 Isaac articulation、MANO_LEFT model、以及冻结的 landmark map；
它们不是该官方入口已提供的可执行依赖。

因此，原版“运行时 reference/目标消费”可以借鉴，但原版 MANO→机器人目标函数不能直接复用到
Sharpa，原因是 Adroit 为自由根 + 22 finger actuators，而候选世界是固定双臂 + 两个 22 DoF
Sharpa。把 Adroit `robot_qpos` 当作 Sharpa target 会混淆机构、根位姿和关节限位，禁止采用。

最小适配建议是重建一个**隔离的、ViViDex-owned**左手 retarget adapter：输入只取左 MANO，
输出只取左 22 finger reference；腕/arm IK 不在 M1-A。它复现论文式（1）的“对应关键点位置
匹配 + 相邻帧平滑 + 限位中点初始化 + NLopt”结构，并仅为 Sharpa 的 FK、自由度、解剖 landmark
和局部坐标基作必要 embodiment 适配。不得以五个 tip 或可选 MCP 集合替代论文的
tip + middle-phalanx 集合；也不得改用 DexPilot/Ours `left_hand_q` 来规避该缺口。

### 3.2 已知输入/输出契约

| 项 | M1-A 规格 |
|---|---|
| 输入 artifact | 候选 `perception/pour17_perception.npz`，并记录其 SHA-256 与 schema 字段清单。 |
| 输入字段 | `left_mano_trans (T,3)`、`left_mano_rot (T,3)`、`left_mano_pose45 (T,45)`、`left_mano_betas (T,10)`、`left_mano_valid (T,)`；T=142。 |
| 单位/角度 | 位置米；MANO pose/rot 的具体参数化须由 MANO layer 代码和 source schema 对拍，当前不可只凭数组形状认定。Sharpa q 为 rad。 |
| 坐标/四元数 | 物体与已导出 pose 是 world, xyz+wxyz；M1-A 手指目标应在手局部几何中拟合，不能使用未冻结的 world→Isaac 变换。 |
| 时间轴 | 在确认生成脚本的 15 Hz reconstruction 声明与 `frame_ids=0..141` 一致后，输出保留 `source_frame=frame_ids` 与派生相对时间 `t_s=frame_ids/15.0`；绝不把它写成或补成观测 timestamp。20 Hz control resampling 留给后续统一时间轴决定。 |
| 有效帧 | 输出 `valid` 必须逐帧等于输入 `left_mano_valid`。本阶段不 nearest-fill、插值、删帧或 confidence gating；这些是待决方法/数据政策。 |
| 输出 artifact（拟议） | `m1a_left_reference_v1.npz`: `schema_version, source_sha256, source_frame(T), t_s(T), valid(T), left_finger_q_rad(T,22), joint_names(22), objective_id, objective_config_sha256, target_keypoint_names, diagnostics`。不含 arm q、Ours reference、认证 q 或 reward。 |

候选左手 22 joint 名/顺序来自 `world/world_manifest.json` 的 controlled order 的索引 36–57：
`left_thumb_CMC_FE, left_thumb_CMC_AA, left_thumb_MCP_FE, left_thumb_MCP_AA, left_thumb_IP,`
`left_index_MCP_FE, left_index_MCP_AA, left_index_PIP, left_index_DIP,`
`left_middle_MCP_FE, left_middle_MCP_AA, left_middle_PIP, left_middle_DIP,`
`left_ring_MCP_FE, left_ring_MCP_AA, left_ring_PIP, left_ring_DIP,`
`left_pinky_CMC, left_pinky_MCP_FE, left_pinky_MCP_AA, left_pinky_PIP, left_pinky_DIP`。
对应软限位必须从同一 manifest 的 `robot.controlled_joint_limits_rad[36:58]` 按名称读取，不能把右手
数组或 URDF 文本顺序当作控制顺序。该来源需在权威世界确认后重新哈希。

### 3.3 论文式（1）的重建与 Sharpa 必要适配

当前只读证据见 [`M1A_INPUT_GEOMETRY_AUDIT.md`](M1A_INPUT_GEOMETRY_AUDIT.md)：候选 decoder 在 v4
冻结条件下已与指定官方 layer 数值对拍通过；当前可见代码支持 axis-angle，但历史 HaWoR 生产语义仍
UNKNOWN；五 tip、四个长指 PIP–DIP 中点及固定局部 `A` 均只是 PROPOSED 候选。reviewer 指定的
thumb 候选 `proposed_thumb_internal_segment_midpoint_v1` 已完成静态结构核验但仍待批准，解剖与作者原始
映射语义仍 UNKNOWN。候选左 Sharpa 22-joint FK/限位接口已通过内部一致性验证，但未与权威 runtime
articulation 对拍，十点也只获准作为隔离诊断测试点。因此这些证据仍不足以批准正式/完整 M1-A 实现。
reviewer 后续仅授权下述冻结合成案例和六例真实 MANO 单帧隔离诊断，
不能外推为完整轨迹或合同批准。

对每个 `left_mano_valid[t]` 为 true 的帧，待实现目标应保持论文的形式：

`min_(q_t in [l,u])  sum_(j in J) || FK_Sharpa,j(q_t) - A(p_MANO,j,t) ||^2 + alpha ||q_t-q_(t-1)||^2`

其中 `J` 必须由五指的 **fingertip 与 middle phalanx** 对应点组成；首帧 `q_0=(l+u)/2`，
后续帧以上一解为平滑项参照和求解初值，优化器为 NLopt。`A` 只能是经核验的 MANO-hand-local
到 Sharpa-palm-local 固定坐标约定/尺度关系；它不是未冻结的 reconstruction-world→Isaac
变换，也不包含 wrist/arm motion。

Sharpa 所需的适配仅限以下机构事实，均须在配置中显式冻结：

- **FK 与点帧**：候选左手 URDF 是
  `RL/Worktrees/RL-Correction-Step4-Publish/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa_fix.urdf`，
  其中存在五个 `left_*_fingertip` links 和 22 个可动左指关节。它可用于候选几何规格，
  但尚未证实等于权威 runtime USD/articulation；正式实现必须改为已冻结世界的模型并记录哈希。
- **控制顺序与限位**：候选来源是 bundle `world/world_manifest.json` 的
  `robot.controlled_joint_names_in_order[36:58]` 与按同名索引读取的
  `robot.controlled_joint_limits_rad[36:58]`，而不是
  URDF 文本顺序。权威版本未确认前，这也是 candidate only。
- **MANO FK**：候选 `MANO_LEFT.pkl` 已记录 SHA-256
  `c4022f…6a30`；候选 NumPy decoder 已在两种显式 mean 配置和冻结案例下与指定官方 MuJoCo
  `ManoLayer` 数值一致，证据绑定也已通过。当前可见平滑/导出/消费代码按 axis-angle/rotvec 处理，
  但没有与历史 pour/17 run 绑定；历史 HaWoR 的模型、mean、left-shapedirs 和完整 root/translation
  语义仍为 UNKNOWN。
- **解剖 map**：[`M1A_INPUT_GEOMETRY_AUDIT.md`](M1A_INPUT_GEOMETRY_AUDIT.md) 的修复版已核验 public
  layer 的 21 点顺序与候选 Sharpa URDF 的同 frame 几何端点。五 tip 和四个 PIP–DIP 中点为
  **PROPOSED**、且均非 COM。指定 thumb 候选为 MANO `(J2+J3)/2` 与 Sharpa
  `left_thumb_PP:[0.0195,0,0]` 的运动学关节线段中点；其 selector、直接父子链、URDF 端点和有限扰动
  FK 已通过结构核验，状态只能写作“指定候选已完成结构核验，仍待批准”。`J1/J2/J3` 的 public
  `mcp/pip/dip` labels 不构成 CMC/MCP/IP 解剖认证，该候选也不是论文作者原始 mapping；最终共同世界的
  Sharpa runtime 模型/FK 和局部坐标/镜像约定仍须核验。

#### 3.3.1 候选左 Sharpa 22-joint FK/限位接口

隔离接口 `proposed_sharpa_left_22joint_fk_v1` 为 **PROPOSED**。它直接读取已登记交接包归档成员
`pour17/world/world_manifest.json`（SHA-256 `2339f7a9…dcd5`）中上述 22 名称与软限位，并绑定候选
URDF（SHA-256 `968b41f8…76db`）；没有使用同名但哈希为 `44519e35…ec3` 的 `_inspection` 副本。
输入严格为按 manifest/document 顺序的 finite `q_rad (22,)`，以 `1e-6 rad` 仅作限位比较容差，错误
shape、NaN 或明确超限会报错，绝不 clip。输出为以 `left_hand_C_MC` 为根的 33 个可达 link transforms
（含所有中间 fixed joints）和固定顺序的 10 个测试 landmark `(10,3)`：每指 tip 后接 middle，顺序为
thumb、index、middle、ring、pinky；middle 使用 thumb PP `[0.0195,0,0] m` 与四个长指 MP
`[0.01575,0,0] m`。

内部一致性诊断 **PASS**：22 名称唯一、均在 URDF 中且为无 mimic 的 revolute joints；父子 link、axis、
origin、类型及逐关节 manifest/URDF 限位均已记录，最大限位文本/float 差为
`1.609802247060088e-7 rad`，在既定 `1e-6 rad` 内。`q=0` 与既有 q=0-only zero-frame 的 33 个
transform 差为 `0`；`q_mid` shape/finite/刚体变换检查通过。22/22 次从 `q_mid` 增加各自限位跨度
10% 的单关节检查全部通过：非后代最大变化 `0`，由基准世界轴/支点和独立 Rodrigues 公式得到的后代
位置最大误差 `2.7755575615628914e-17 m`、旋转最大误差 `3.3306690738754696e-16`，landmark 运动关系
最大误差 `2.7755575615628914e-17 m`；最大正交残差 `1.1102230246251565e-15`，最大
`|det-1|=9.992007221626409e-16`。错误 shape、NaN、明确超限三条拒绝路径均 PASS。

有效配置为 [`m1a_sharpa_left_fk_candidate_v1.json`](configs/m1a_sharpa_left_fk_candidate_v1.json)
（SHA-256 `c6ff31d3961fe28714d4556f5fc503f29d745805c6666dcd11af386d0800312e`）；接口为
[`m1a_sharpa_left_fk_candidate_v1.py`](diagnostics/m1a_sharpa_left_fk_candidate_v1.py)
（`1824dd4eaeef280c6d19008e6707f25935889faa9f219960778617ebcbc82c64`）；诊断脚本为
[`m1a_sharpa_left_fk_audit_v1.py`](README.md#archive-access)（归档：`diagnostics/m1a_sharpa_left_fk_audit_v1.py`）
（`6618a661638b0b30318620313f2cc5849685dcf2f2b6b82b9be11eaaade6763d`）；完整逐关节结果见
[`m1a_sharpa_left_fk_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_left_fk_candidate_v1/m1a_sharpa_left_fk_audit_v1.json`）
（`fa1cfba3fb3ea5f307791c0a60b3648ec918f5e5790dc981db28e4e44f676aff`）。该 PASS 仅验证候选 URDF
树和 manifest 限位下的 FK 内部一致性；没有对拍权威 USD/Isaac articulation，不批准十点正式方法合同，
也不是 M1-A retargeting PASS。

#### 3.3.2 隔离单帧合成求解器候选

首次授权的隔离候选 `proposed_sharpa_single_frame_synthetic_solver_v1` 只消费上述 Sharpa FK 自己生成的
`y (10,3)`、`q_prev (22,)`、`q_init (22,)`，与 MANO、固定 A、真实 perception 和机器人 reference
完全分离。目标固定为对十点及 xyz **求和**的
`E(q)=sum_j ||FK_j(q)-y_j||^2 + 4e-3 ||q-q_prev||^2`，约束使用已登记 manifest lower/upper；没有
接触、碰撞、affordance、腕姿态或其它损失。求解封装固定为 NLopt `LD_SLSQP`、`maxeval=500`、
`ftol_abs=1e-12`、`xtol_abs=1e-8 rad`，并只向 NLopt 提供解析梯度。该算法是本项目合成检查的
**PROPOSED** 工程选择，不声称为作者采用的具体 NLopt 算法。

manifest 差异已收口：已登记归档成员 `2339f7a9…dcd5` 与 `_inspection` 副本 `44519e35…ec3` 的左手
22 名称、顺序和限位逐值一致；全 JSON 的其它不同路径仅为 `robot.self_collision_note` 与
`robot.self_collision_source`，未展开其值。本轮继续只用归档成员，不合并或修改两文件。既有候选 FK
报告 `fa1cfba3…6aff` 实际绑定归档成员；更早的 q=0 zero-frame、wrist-zero A 和 thumb 结构证据直接
读取候选 URDF，并未消费任一 manifest 的顺序/限位，因此同名 manifest 差异不推翻这些窄结论。

解析 landmark Jacobian 使用每个祖先 revolute joint 的运动前 root-frame axis/pivot，列为
`axis × (point-pivot)`，非祖先列为零；目标梯度为
`2 sum_j J_j^T(FK_j-y_j)+2 alpha(q-q_prev)`。在 `m` 对 S1、中心差分 `h=1e-6 rad` 的全部正负扰动
均在限位内。解析 Jacobian 对独立中心差分的最大误差为 `2.285326788920017e-11 m/rad`，目标梯度
最大误差为 `1.1095837790242587e-12`，分别通过预登记 `1e-7` 阈值；数值差分只用于此检查，求解器
接口未采用它。

三个固定合成案例均已构造并记录 solver 输入。S0/S1/S2 的 q_star oracle 总目标均为 `0`；q_init=m
的初始 total 分别为 `0`、`0.006143438831401291`、`0.005113563450043941`。测试驱动生成
`y=FK(q_star)`，但求解函数签名不接受 q_star，也没有直接返回 q_prev 的路径。

当前执行环境为项目内专用
`D:/UCBP/RL/Baselines/ViViDex/.venvs/m1a_nlopt_v1`：由 UCBP Python 3.10.19 64-bit 建立，
`include-system-site-packages=false`。官方 PyPI binary wheels 固定为
`numpy-2.2.6-cp310-cp310-win_amd64.whl`（SHA-256 `f0fd6321…03f3`）和
`nlopt-2.10.0-cp310-cp310-win_amd64.whl`（`49aa0439…8e1`），通过哈希锁文件
[`m1a_nlopt_v1_requirements.lock.txt`](README.md#archive-access)（归档：`configs/m1a_nlopt_v1_requirements.lock.txt`）（`09f05ea3…3b9f`）
从本地 wheel 安装。实际导入路径均位于该 venv，版本为 NumPy 2.2.6/NLopt 2.10.0；`pip check` PASS。
没有修改 UCBP、base、远端或训练环境，没有源码编译、自动换版或 SciPy fallback。

在该环境中以完全相同的数学参数和案例重新运行已有审计，梯度数值不变且继续 PASS。S0/S1/S2 均
**实际调用 NLopt 并通过质量验收**：三例 return code 都是 `3 / FTOL_REACHED`，objective call count
分别为 `1/47/50`，无异常、负码或 maxeval/maxtime 耗尽；final q/landmarks 全部 finite、无 clipping、
限位违例均为 `0 rad`。

| case | initial total | final tracking / smoothness / total | landmark max-abs | reported↔recomputed | 状态 |
|---|---:|---|---:|---:|---|
| S0 | `0` | `0 / 0 / 0` | `0 m` | `0` | PASS |
| S1 | `0.006143438831401291` | `6.53406504563831e-12 / 4.59955553854043e-12 / 1.11336205841787e-11` | `1.62973684625367e-6 m` | `0` | PASS |
| S2 | `0.005113563450043941` | `5.65508233141535e-12 / 4.65730341254054e-12 / 1.03123857439559e-11` | `9.07009166245532e-7 m` | `0` | PASS |

当前执行配置为
[`m1a_sharpa_single_frame_solver_synthetic_candidate_v2.json`](README.md#archive-access)（归档：`configs/m1a_sharpa_single_frame_solver_synthetic_candidate_v2.json`）
（SHA-256 `a7d118a3d5926de753f9a28feb64843778c07345367a933828fcac56b64a2c20`）；原求解器
[`m1a_sharpa_single_frame_solver_candidate_v1.py`](diagnostics/m1a_sharpa_single_frame_solver_candidate_v1.py)
保持 `fd8c753a…d89d`，原审计脚本保持 `f8852842…ea62`。当前原始报告为
[`m1a_sharpa_single_frame_solver_synthetic_audit_v2.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_single_frame_solver_synthetic_candidate_v2/m1a_sharpa_single_frame_solver_synthetic_audit_v2.json`）
（`02dd747e…001f`）；环境与逐例显式质量绑定为
[`m1a_nlopt_environment_and_execution_audit_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_sharpa_single_frame_solver_synthetic_candidate_v2/m1a_nlopt_environment_and_execution_audit_v1.json`）
（`918bcd09…4aae`）。旧 v1 缺依赖报告 `96204f79…0951` 保留为历史证据，不作为当前状态来源。

该阶段结论只限于 q_prev=q_star 的三例 Sharpa-generated synthetic diagnostics，不证明十点唯一决定 22
关节、真实 MANO retargeting、实际时间平滑、mean 选择、权威 runtime 或 M0；其 q/y 只存在于明确命名的
synthetic solver diagnostic 报告中，不是正式 reference。当前完整序列候选诊断状态见 3.3.7。

#### 3.3.3 六例真实 MANO 单帧候选集成诊断

首次真实输入集成只读取已登记 142 帧 perception NPZ（SHA-256 `28e13c24…34d7`）的
`frame_ids/left_mano_pose45/left_mano_betas/left_mano_valid`，限定帧 0/71/141；三帧 valid 均为 true，
所用数值均 finite，未读取 world/root offset/object 字段。`proposed_raw_pose` 与
`proposed_plus_hands_mean` 分别按既有候选 decoder 语义运行。每例先以同次解码 J0 做 wrist-zero，再用
同一个 wrist-zero calibration v2 的固定 A 和获准诊断用的固定十点对应生成目标；没有逐帧、逐 beta 或
逐分支重标定。

几何输入复核 **6/6 PASS**：六例的 pose45、betas、vertices_H 与 joints21_H 相对既有 hand-local
diagnostic samples 的 max-absolute 差均为 `0 m`，低于 `1e-10 m`。由于专用 NLopt venv 有意只含
NumPy/NLopt，解码在既有 UCBP 环境中产生哈希绑定的纯数值几何 artifact；求解仍在
`.venvs/m1a_nlopt_v1` 中读取该 artifact，两个阶段没有修改已验证 decoder 或 solver。

六例都独立使用 `q_init=q_prev=m`，没有跨帧或跨 mean warm start。全部实际调用 NLopt 并返回
`3 / FTOL_REACHED`，execution checks **6/6 PASS**：objective call count 为 `57/50/58/60/54/55`，
所有 q/输出 finite，限位违例 `0 rad`、无 clipping、无异常或预算耗尽，独立重算 objective 与 NLopt
reported value 差均为 `0`，final total 均低于 initial total。几何误差没有套用合成测试阈值，状态统一为
**PENDING REVIEWER ASSESSMENT**。

| frame / variant | initial total | final tracking / smoothness / total | initial→final all-10 RMS (mm) | tip / middle RMS (mm) | max point (mm) | calls |
|---|---:|---|---:|---:|---:|---:|
| 0 / raw | `0.0439713548` | `0.0086276462 / 0.0090860399 / 0.0177136860` | `66.311→29.373` | `33.216 / 24.945` | pinky_tip `43.576` | 57 |
| 71 / raw | `0.0328537338` | `0.0075854050 / 0.0065842385 / 0.0141696435` | `57.318→27.542` | `29.724 / 25.171` | pinky_tip `38.487` | 50 |
| 141 / raw | `0.0442278574` | `0.0085739274 / 0.0092638541 / 0.0178377815` | `66.504→29.281` | `32.775 / 25.310` | pinky_tip `41.137` | 58 |
| 0 / +mean | `0.0140271815` | `0.0053217508 / 0.0023227992 / 0.0076445501` | `37.453→23.069` | `19.543 / 26.123` | thumb_middle `36.432` | 60 |
| 71 / +mean | `0.0168298694` | `0.0060296250 / 0.0031225050 / 0.0091521301` | `41.024→24.555` | `22.549 / 26.410` | thumb_middle `34.297` | 54 |
| 141 / +mean | `0.0159309500` | `0.0057668268 / 0.0027996578 / 0.0085664846` | `39.914→24.014` | `21.341 / 26.418` | thumb_middle `35.424` | 55 |

逐点欧氏距离、最终 q、逐关节距上下限距离及完整执行字段见
[`real_mano_single_frame_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_v1.json`）；
纯数值样本仅保存在明确标注的
[`real_mano_single_frame_candidate_diagnostic_samples_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_samples_v1.npz`），
不是 robot reference 或轨迹。统一视角图见
[`real_mano_single_frame_candidate_diagnostic_v1.png`](README.md#archive-access)（归档：`artifacts/m1a_real_mano_single_frame_candidate_diagnostic_v1/real_mano_single_frame_candidate_diagnostic_v1.png`），
仅用于检查对应、手形和残差分布，不证明碰撞、接触、世界对齐或某 mean 分支更正确。

这六例是候选接口集成诊断，不是正式输入/十点合同批准，不建立时间平滑，也不证明完整 142 帧 M1-A
质量。两个 mean 分支不在此选择或排序；权威 Sharpa runtime 仍未验证。该六例本身不构成完整 M1-A
验收；当前完整序列候选诊断状态见 3.3.7，M0 仍 **BLOCKED**。

#### 3.3.4 中点锚定影响诊断

隔离 probe `m1a_midpoint_anchor_effect_probe_v1` 只回答：从上一轮每例已保存的 `q_final` 出发，把
`tracking+0.004||q-m||²` 临时改为同一十点的 `tracking` 后，局部继续优化能降低多少同定义 tracking
误差。它直接加载上一轮六例 target `y` 与 `q_final`，复核 prior config/report/samples、landmark/joint
顺序、manifest 限位及代码哈希；没有重新解码 MANO、重算 A、修改原 solver/FK、扫描 alpha、restart
或跨案例传值。正式候选 `alpha` 仍为 `0.004`。

在原 `q_final` 处，六例 `||g_tracking||₂` 与 `||g_anchor||₂` 数值接近且方向抵消，未投影
`||g_total||₂` 为 `1.11e-7` 至 `2.40e-4`。边界解释采用 `1e-6 rad`：位于下界时正梯度分量可由约束
抵消，位于上界时负梯度分量可由约束抵消；其它分量保留。按此约定，原解 projected
`||g_total||₂` 为 `1.11e-7` 至 `5.20e-7`。其中只有 frame 141/raw 的 `left_pinky_CMC` 在原解处识别为
下界 active；其未投影 `||g_total||₂=2.4013e-4`，投影后为 `1.9572e-7`。这些数值只解释旧目标中
tracking 与 anchor 的平衡，不用新阈值重新判定旧结果。

六例 probe 均从同例原 `q_final` 独立启动，实际调用相同 NLopt `LD_SLSQP` 并返回
`3 / FTOL_REACHED`；calls 为 `139/147/148/161/160/162`。执行检查 **6/6 PASS**：finite、无 clipping、
限位违例 `0 rad`，reported 与独立重算 tracking 差为 `0`，probe tracking 未高于其初始 tracking。

| frame / variant | tracking 原→probe（下降） | all-10 RMS 原→probe (mm) | tip / middle RMS probe (mm) | probe max point (mm) | `||q_probe-q_old||₂` / `||q_probe-m||₂` (rad) |
|---|---:|---:|---:|---:|---:|
| 0 / raw | `0.00862765→0.00380806` (`55.86%`) | `29.373→19.514` | `14.991 / 23.171` | pinky_middle `29.594` | `2.295 / 3.031` |
| 71 / raw | `0.00758541→0.00296106` (`60.96%`) | `27.542→17.208` | `13.103 / 20.507` | pinky_middle `28.126` | `2.136 / 2.780` |
| 141 / raw | `0.00857393→0.00360521` (`57.95%`) | `29.281→18.987` | `14.865 / 22.363` | pinky_middle `28.479` | `2.260 / 3.004` |
| 0 / +mean | `0.00532175→0.00178478` (`66.46%`) | `23.069→13.360` | `7.717 / 17.245` | thumb_middle `25.381` | `2.364 / 2.717` |
| 71 / +mean | `0.00602963→0.00181923` (`69.83%`) | `24.555→13.488` | `7.772 / 17.420` | thumb_middle `24.776` | `1.905 / 2.454` |
| 141 / +mean | `0.00576683→0.00175921` (`69.49%`) | `24.014→13.264` | `7.769 / 17.073` | thumb_middle `24.759` | `2.204 / 2.653` |

聚合 tracking/RMS 降低不表示每个点都降低：0/raw 的 index/middle/ring middle 分别增加
`0.200/1.747/1.977 mm`，141/raw 的 ring_middle 增加 `1.183 mm`；其它案例未出现逐点增加。
完整十点逐点误差、tracking 下降量、每关节变化、到 `m` 与上下限的距离、三项梯度及其 projected
版本、返回码和独立重算检查见
[`m1a_midpoint_anchor_effect_probe_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_midpoint_anchor_effect_probe_v1/m1a_midpoint_anchor_effect_probe_v1.json`）。
probe 配置 SHA-256 为 `a4fb88d8…c65e`，脚本为 `5050533e…6c11`，报告为 `15c5db6f…b08a`；原 solver
仍保持 `fd8c753a…d89d`。
数值表已充分表达本轮唯一干预，因此没有生成新图。允许结论仅为：移除中点锚定并从原解继续局部优化后，
六例 tracking 分别降低上述比例。不能称为全局最小值、不可达下界或剩余误差来源证明；不能据此把
`alpha=0` 纳入正式 baseline，也不能选择 mean 分支。该 probe 不构成 M1-A 验收；当前状态见 3.3.7，
M0 仍 **BLOCKED**。

#### 3.3.5 原始帧 0..15 的 I/T 短序列候选诊断

隔离诊断 `m1a_short_sequence_candidate_diagnostic_v1` 使用原始 perception NPZ 的连续 source frame
`0..15`，分别处理 `proposed_raw_pose` 与 `proposed_plus_hands_mean`。两分支复用同次 J0 归零、固定
wrist-zero calibration v2、固定十点顺序、候选 22-joint FK/限位和既有单帧 solver；全部输入帧
`left_mano_valid=true` 且 pose45/betas finite。没有读取 world/root offset、Ours reference、认证 q、
confidence 或 GraspPose，也没有重采样或生成 timestamps。

两条件均保持 `alpha=0.004` 和同一冻结 NLopt 合同。I 每帧独立使用 `q_init=q_prev=m`；T 的 frame 0
与 I 共用同一实际求解，随后只在本分支内使用上一帧成功解作为 `q_init=q_prev`。因此实际调用 62 次，
对应 64 条条件/帧记录；两分支 frame 0 各节省一次重复调用。执行检查 **64/64 PASS**，62 次实际调用
均返回 `3 / FTOL_REACHED`，无异常、预算耗尽、clipping 或限位违例；reported 与独立重算 objective
最大差为 `0`，每次 final total 均低于各自 initial total。没有失败或未执行帧。

| variant | mean tracking I → T | mean all-10 RMS I → T (mm) | mean tip RMS I → T (mm) | mean middle RMS I → T (mm) | mean `||Δq||₂` I → T (rad) | max `||Δq||₂` I → T (rad) |
|---|---:|---:|---:|---:|---:|---:|
| raw | `0.0087303 → 0.00516863` | `29.547 → 22.657` | `33.299 → 20.743` | `25.242 → 24.295` | `0.01306 → 0.12209` | `0.03542 → 0.40511` |
| +mean | `0.00544726 → 0.00277692` | `23.339 → 16.547` | `19.647 → 9.246` | `26.521 → 21.406` | `0.01024 → 0.11454` | `0.03105 → 0.37674` |

所有 16 帧均可配对；frame 0 因共享求解而相同，T 在其余 15/15 帧的 tracking 和 all-10 RMS 均低于
I。与此同时，T 的平均和最大帧间关节增量在两个分支中都高于 I；相邻目标十点 RMS 位移均值仅为
raw `0.4569 mm`、+mean `0.3921 mm`。这些是同口径的误差与连续性观测，不是方法优劣或质量 PASS。
T 相对 I 同时改变初始化与正则参照，差异不能只归因于其中一项；也不能由较低残差选择 mean 分支。

完整逐帧 tracking/regularization/total、十点误差、q 来源、限位距离、返回码和时序指标见
[`short_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_v1.json`）；
诊断 q/目标数组见
[`short_sequence_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_arrays_v1.npz`）。
两张曲线见 [raw](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_raw_pose_v1.png`）
与 [+mean](README.md#archive-access)（归档：`artifacts/m1a_short_sequence_candidate_diagnostic_v1/short_sequence_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）。
配置/求解脚本/报告/数组 SHA-256 分别为 `5c7c0aa2…3ac4`、`596cf3f9…f81d`、
`9b9c683d…a3ec`、`b8bc35d1…e2b3`。这些数组仅为 `short_sequence_candidate_diagnostic`，不是正式
reference 或 simulator 轨迹。几何质量仍待 reviewer 评估；正式 mean、输入/十点合同及 M1-A 验收均未
批准或通过。当前完整序列候选诊断状态见 3.3.7，M0 仍 **BLOCKED**。

#### 3.3.6 固定目标递推归因对照

隔离诊断 `m1a_fixed_target_recursion_candidate_diagnostic_v1` 直接绑定上一轮配置、报告、目标/q 数组及
FK/solver/manifest 哈希，没有重新解码 MANO 或计算 A。每个 mean 分支复用其 I/T 完全相同的共享
frame-0 解作为 `q_F[0]`（逐元素 max-absolute 差均为 `0 rad`），并把上一轮同分支 frame-0 目标 `y0`
固定重复为 `y_F[k]`。`k=1..15` 每步仅令 `q_init=q_prev=q_F[k-1]`，保持 `alpha=0.004` 和既有
NLopt 合同。`k` 只是诊断迭代索引，不是 source frame、时间戳或获批 warmup。

两个分支新增 30 次实际求解并全部通过执行检查：均返回 `3 / FTOL_REACHED`，calls 范围 `14–54`；
无异常、预算耗尽、clipping 或限位违例；reported 与独立重算 objective 最大差为 `0`。每步初始正则
精确为零，final total 均低于该步 initial total；final tracking 相对上一迭代的最大“上升”仍为下降
`-2.7015e-5`，故 30/30 均满足固定目标单调检查。

| variant | F tracking `k0→k15` | F all-10 RMS `k0→k15` (mm) | F mean/max `||Δq||₂` (rad) | F/T `k15` 相对 q[0] 净位移 (rad) | `||q_T-q_F||₂` at k15 (rad) | FK(T)/FK(F) RMS at k15 (mm) |
|---|---:|---:|---:|---:|---:|---:|
| raw | `0.00862765→0.00429513` (`-50.22%`) | `29.373→20.725` | `0.12021 / 0.40516` | `1.41378 / 1.37473` | `0.22209` | `4.333` |
| +mean | `0.00532175→0.00218649` (`-58.91%`) | `23.069→14.787` | `0.11163 / 0.37688` | `1.42921 / 1.46698` | `0.14006` | `3.042` |

因此，在目标完全不动的 F 中，当前递推机制仍产生约 `1.41–1.43 rad` 的相对 q[0] 净位移，并持续降低
tracking。允许的窄结论是：既有 T 中的运动不能全部归因于视频目标变化。F/T 的步长差不是可相加的
“输入运动贡献”；非线性优化与路径依赖不允许这种严格分解。F tracking 始终针对固定 `y0`，T tracking
针对各 source frame 的 `y[k]`，两者 total 也不用于方法优劣判断。

完整逐迭代目标、误差、关节步长、相对 q[0] 净位移、限位/返回码和 F/T 对照见
[`fixed_target_recursion_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_v1.json`）；
诊断数组见
[`fixed_target_recursion_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_arrays_v1.npz`），
曲线见 [raw](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_raw_pose_v1.png`）
与 [+mean](README.md#archive-access)（归档：`artifacts/m1a_fixed_target_recursion_candidate_diagnostic_v1/fixed_target_recursion_candidate_diagnostic_proposed_plus_hands_mean_v1.png`）。
旧 F 报告字段 `cumulative_q_displacement_rad_k15` 与数组字段 `cumulative_q_F_from_0_l2_rad`
保留为已哈希历史证据；其实际计算定义均为 `||q_F[k]-q_F[0]||₂`，即相对初始解的净位移，而不是
逐步长求和的累计路径长度。旧 artifact 不改写。
配置/脚本/报告/数组 SHA-256 分别为 `0b6658a6…0b3a`、`5479c2f2…7a12`、
`803554f8…369d`、`e0cbb400…0d0f`。本对照不证明 F 达到全局最优，不批准固定目标 warmup、正式
初始化/时间策略或 mean/十点合同。该归因对照不构成 M1-A 验收；当前状态见 3.3.7，M0 仍 **BLOCKED**。

#### 3.3.7 完整 142 帧 T 候选几何诊断

`m1a_full_sequence_candidate_diagnostic_v1` 严格延续上一轮 T 流程。先核验 0..15 前缀的输入/资产/
solver/环境哈希、连续 source frame、32 条 T 执行记录、目标值及 `q_init=q_prev=q_T[t-1]` 链均一致，
然后逐值复用该前缀；旧 solver 调用重跑数为 0。从每个分支各自 `q_T[15]` 继续 frame 16..141，新增
252 次求解。输入完整 `frame_ids=0..141`，全部 `left_mano_valid=true` 且 pose45/betas finite；没有填补、
插值、删帧、重采样、假 timestamps、warmup、重复首帧、alpha=0/F 初始化或禁止输入。

后缀 252/252 次求解均返回 `3 / FTOL_REACHED`，执行检查全部通过：无异常、预算耗尽、clipping 或限位
违例，objective reported/独立重算最大差 `0`，final total 相对各次 initial total 的最大变化仍为下降
`-4.1780e-7`。加上 32 条复用前缀，完整报告含 284/284 条通过的分支/帧记录；这不等于 284 次本轮
新执行。

| variant / segment | mean tracking | mean all-10 RMS (mm) | mean tip / middle RMS (mm) | mean/max `||Δq||₂` (rad) | mean target step (mm) |
|---|---:|---:|---:|---:|---:|
| raw / 0..15 | `0.00516863` | `22.657` | `20.743 / 24.295` | `0.12209 / 0.40511` | `0.4569` |
| raw / 16..141 | `0.00335876` | `18.289` | `13.100 / 22.235` | `0.04727 / 0.15244` | `1.0655` |
| +mean / 0..15 | `0.00277692` | `16.547` | `9.246 / 21.406` | `0.11454 / 0.37674` | `0.3921` |
| +mean / 16..141 | `0.00188959` | `13.742` | `7.959 / 17.726` | `0.03610 / 0.11573` | `0.8105` |

分段只是预先指定的描述范围，不命名为“过渡期/稳态”。后缀仍持续变化：raw 与 +mean 的最大后缀
`||Δq||₂` 均发生于 frame 27，分别为 `0.15244 rad`（最大分量 `left_index_MCP_AA=0.09123 rad`）和
`0.11573 rad`（最大分量 `left_index_MCP_FE=0.07085 rad`）。后缀最大单点误差分别是 frame 21
`pinky_middle=30.358 mm` 和 frame 16 `thumb_middle=28.163 mm`。完整序列最大误差与最大步长仍来自
frame 0/1，因此首段没有从统计中删除。

位移定义现已明确分开：到 frame 141 的 `||q[141]-q[0]||₂` 是**净位移**，raw/+mean 为
`2.14479/2.18557 rad`；逐帧 `||Δq||₂` 的**累计路径长度**为 `7.78712/6.26707 rad`。候选限位附近
停留频繁：raw 的 `left_pinky_CMC`、`left_pinky_MCP_FE` 各为 `141/142` 帧，
`left_index_MCP_FE=139/142`；+mean 的 `left_thumb_MCP_AA=141/142`、
`left_thumb_MCP_FE=123/142`。这是候选限位描述，不证明真实 runtime 或物理可执行性。
后段 `16..141` 的 middle RMS 仍为 raw `22.235 mm`、+mean `17.726 mm`，说明完整候选序列运行后
仍存在持续的 middle 残差；这只是描述性结果，不能据此归因为 landmark、mean、限位或机构差异。

完整逐帧结果与分段统计见
[`full_sequence_candidate_diagnostic_v1.json`](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_v1.json`），
诊断数组见
[`full_sequence_candidate_diagnostic_arrays_v1.npz`](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_arrays_v1.npz`）。
完整曲线：[raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_curves_v1.png`）、
[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_curves_v1.png`）；
固定 frame 0/15/71/141 对照：[raw](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_raw_pose_static_v1.png`）、
[+mean](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_proposed_plus_hands_mean_static_v1.png`）。
配置/几何准备脚本/求解脚本/报告/数组 SHA-256 为 `a61c47e9…e974`、`82ef836c…5250`、
`8f8fef60…10ea`、`1f75cb81…60f3`、`5d9f5947…852c`。这是完整序列候选**几何诊断**，不是正式
robot reference、姿态/物理验收或 mean 选择；正式 M1-A 验收仍未通过，M0 仍 **BLOCKED**。

#### 3.3.8 已完成：既有完整序列的限位与残差归因审计

审计严格复用 3.3.7 保存的 target、`q_final`、实际 `q_prev`、配置和哈希链，从注册归档 manifest 读取
同一 22 项软限位，并只调用已验证候选 FK/Jacobian。没有调用优化器、解码 MANO、重算 A、修改旧结果
或生成新轨迹。284 组 FK 与保存数组逐值一致；分解的 `g_tracking+g_smooth` 与冻结 objective gradient
逐值一致。贴边仍定义为距相应边界 `≤1e-6 rad`，`16..141` 仍只是描述性分段。

| variant / segment | 最常贴边关节（边界，次数） | middle/tip tracking² share | 主要残差点（tracking² share） |
|---|---|---:|---|
| raw / 0..141 | pinky CMC lower `141`；pinky MCP-FE lower `141`；index MCP-FE lower `139` | `70.96% / 29.04%` | pinky_middle `23.17%`；thumb_middle `15.61%`；index_middle `14.30%` |
| raw / 16..141 | pinky CMC lower `126`；pinky MCP-FE lower `126`；index DIP upper `126` | `73.67% / 26.33%` | pinky_middle `24.26%`；index_middle `15.26%`；thumb_middle `15.10%` |
| +mean / 0..141 | thumb MCP-AA lower `141`；thumb MCP-FE upper `123`；pinky PIP upper `117` | `83.17% / 16.83%` | thumb_middle `34.22%`；pinky_middle `23.98%`；ring_middle `9.99%` |
| +mean / 16..141 | thumb MCP-AA lower `126`；thumb MCP-FE upper `123`；pinky PIP upper `117` | `83.21% / 16.79%` | thumb_middle `34.39%`；pinky_middle `24.36%`；ring_middle `9.97%` |

十点各自 mean/median/P95/max@frame、mean xyz、xyz 标准差、方向一致度，以及所有 22 关节的上下界
次数/占比、最长连续 source-frame 区间和最小边距均见[紧凑表](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/M1A_LIMIT_RESIDUAL_AUDIT_V1.md`）。主要 middle 点残差的 root-frame 方向高度一致，但方向相关性
不能直接认作标定错误。

固定 frame `0/15/16/27/71/141` 的两个分支共 12 例；frame 0 无贴边，其余 10 例共 58 个贴边关节
分量。每一项的 tracking 单独与 total 负梯度下降方向均指向相应边界外；投影后 total L2 为
`1.329e-7..7.887e-7`，未投影为 `7.151e-4..4.164e-3`。因此固定帧证据支持“当前候选边界限制这些
关节的局部一阶改善”，而不是只观察到贴边；但它不证明全局最优、机构不可达、边界是全部残差的唯一
原因或应该放宽限位。

[完整 JSON](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_v1.json`）保存 22 维完整/投影
梯度，[NPZ](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_arrays_v1.npz`）保存诊断数组。
报告/数组/配置/FK/Jacobian/注册 manifest member 的实际 SHA-256 为 `990d8916…907c`、
`4a02dc8b…8a78`、`9bf06f7a…2ced`、`1824dd4e…2c64`、`fd8c753a…d89d`、`2339f7a9…dcd5`，完整值见
[哈希清单](README.md#archive-access)（归档：`artifacts/m1a_limit_residual_audit_v1/m1a_limit_residual_audit_hashes_v1.json`）。机构尺寸、点定义、
标定、mean 语义、局部极值/求解路径及权威 runtime 等价性仍不能区分。两个 mean 不按残差选择；限位、
`alpha`、`A` 与十点不变。正式 M1-A 仍 **NOT PASSED**。该任务当时停止；reviewer 随后接受结果并
单独授权 3.3.9 的只读点对距离检查。

#### 3.3.9 已完成：五指 tip–middle 点对距离相容性

审计按名称配对五组点，复用 3.3.7 的完整 target/q、候选 FK/点定义、实际 URDF 与注册 manifest 限位；
scale=`1`。未调用优化器、重新解码 MANO、重算 A 或修改任何旧产物。实际 middle-link→tip-link 链均
由一个无 mimic revolute 与两个 fixed 构成：thumb 的相对关节为 IP，四个长指为各自 DIP，共同上游
运动不改变点对距离，五指均满足解析前提。

| finger | relative joint / manifest interval (rad) | analytic robot distance interval (mm) |
|---|---:|---:|
| thumb | `left_thumb_IP / [0,1.7452999353]` | `30.913684..47.110935` |
| index/middle/ring/pinky | respective DIP / `[0,1.3962998390]` | `32.661645..41.759257` |

区间只由 `d²=A+B cos(q)+C sin(q)` 在闭区间端点及区间内 `atan2(C,B)+kπ` 驻点解析求得，没有稠密
采样、无约束角度或新搜索。下界/中点/上界/驻点的解析值与 FK 最大差 `2.776e-17 m`；284 个实际 q
的解析/FK 最大差同为 `2.776e-17 m`，且实际机器人距离均在解析区间内（最大数值偏差
`2.776e-17 m`）。

明确 `delta>1e-8 m` 的超界为：raw thumb 目标过长 `117/142`、最大 `1.0183 mm @ frame 10`；raw
pinky 目标过短 `142/142`、最大 `3.4103 mm @ 93`；+mean index 目标过短 `64/142`、最大
`0.1843 mm @ 38`；+mean pinky 目标过短 `142/142`、最大 `2.8835 mm @ 93`。其余六个
分支/手指组合无明确超界，只表示该点对必要条件未排除可达性。

按五对互不重复点的 `delta²/2` 求和，raw/+mean 的 tracking 必要下界 median 为
`4.51199e-6/2.72492e-6 m²`，对应实际 tracking median 为 `3.32239e-3/1.87847e-3 m²`；逐帧
actual≥bound 检查均通过，0 帧超过 `1e-10 m²` 容差。下界通常不紧，不能称为全局最优；下界与实际
tracking 的比值不能解释为已说明的因果比例，也不能将差值归因于单一原因。

证据见[紧凑表](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/M1A_TIP_MIDDLE_DISTANCE_COMPATIBILITY_AUDIT_V1.md`）、[完整 JSON](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_v1.json`）、[逐帧数组](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_arrays_v1.npz`）和[实际哈希清单](README.md#archive-access)（归档：`artifacts/m1a_tip_middle_distance_compatibility_audit_v1/m1a_tip_middle_distance_compatibility_audit_hashes_v1.json`）。
任一点对超界足以排除该帧十点同时零误差；两分支 pinky 均为 `142/142` 超界，故当前 284 个分支/帧
样本全部不可能精确零误差匹配，但这不表示近似 retargeting 不可用。超界证明当前候选点定义、模型尺度
和关节范围组合存在特定点对距离不相容，且固定 A 的旋转/平移不能消除；不能据此选择应改模型、尺度、
点定义或 mean。两个 mean 不按超界量选择，限位、`alpha`、A、点定义与 scale 均不变。reviewer 随后
接受该审计并授权下述固定 q 后验诊断。

#### 3.3.10 已完成：固定 q 后验共同平移/刚体配准诊断

以完整序列已有 q 的 FK 十点 `X` 拟合同帧目标 `Y`，分别计算原始、142 帧共享平移、共享正旋转刚体、
逐帧平移和逐帧正旋转刚体误差。所有变换均标记
`POSTHOC_DIAGNOSTIC_ONLY_NOT_CALIBRATION`，未写回 q、Y、正式 A、目标或轨迹。

raw 的全序列 tracking/RMS 从 `0.5059016 m² / 18.8751 mm` 降至共享平移
`0.3257862 / 15.1468`、共享刚体 `0.2652390 / 13.6670`、逐帧平移 `0.3140612 / 14.8718`、逐帧刚体
`0.2553757 / 13.4105`；+mean 从 `0.2825186 / 14.1052` 降至 `0.2176387 / 12.3801`、
`0.2113559 / 12.2001`、`0.2132440 / 12.2545`、`0.2039478 / 11.9844`。RMS 均按全部 `142×10`
点统一计算。

共享刚体的 raw/+mean 平移范数为 `27.2662/7.7337 mm`、旋转角为 `11.0606°/5.1868°`；SVD 与全部
逐帧拟合均为秩 3，无退化/可能非唯一标记。共享平移和共享刚体各自的变差帧均为 0。逐帧刚体的平移
范数范围为 raw `20.7667..32.0482 mm`、+mean `6.6249..26.8990 mm`，旋转角范围为
`7.6729..14.0991°` 和 `3.8309..12.1611°`。原 tracking 对拍差为 0；SSE 顺序、正交性、`det=+1`、
tip–middle 距离不变核验全部通过。

证据见[紧凑表](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/M1A_FIXED_Q_POSTHOC_RIGID_ALIGNMENT_AUDIT_V1.md`）、[完整 JSON](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.json`）、[逐帧数组](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_arrays_v1.npz`）、[紧凑图](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_v1.png`）和[实际哈希清单](README.md#archive-access)（归档：`artifacts/m1a_fixed_q_posthoc_rigid_alignment_audit_v1/m1a_fixed_q_posthoc_rigid_alignment_audit_hashes_v1.json`）。共享刚体降低只说明固定 q 残差含可由共同刚体变换减少的成分；不能直接判定正式 A 错误，逐帧优势也不授权逐帧标定或腕部动作。刚体不能消除 3.3.9 的点对距离冲突，不按结果选择 mean。正式 M1-A 仍 **NOT PASSED**；本轮停止，不授权新优化、replay 或训练，是否进一步检查 A 的构造依据由 reviewer 决定。

#### 3.3.11 已完成：固定 A 构造依据审计

审计复用 v2 登记的 MANO/selector、既有 wrist-zero canonical 数组、Sharpa URDF 与注册 manifest，只做
静态坐标复算。`J0/J5/J9/J13/J17` 的 raw joint 来源、四个 Sharpa MCP_VL 的 joint/link/origin/axis、
四指 MCP_FE/AA 共点及 pinky CMC 上游均已核实。R/t 与 v2 配置最大差为 `0`；A 对齐四 MCP 均值及
构造基而非 wrist/root，`A(J0)-left_hand_C_MC` 范数为 `9.578618 mm`。

未发现坐标系、frame、索引、变换方向或配置一致性的明确实现矛盾；但没有证据认证
`left_hand_C_MC↔J0` 的解剖/作者对应，它仍是用于构造纵向轴的工程假设。四 MCP 均值、纵横构轴参考、
单一 raw canonical 与 scale=1 也未升级为批准合同。证据见[紧凑报告](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/M1A_FIXED_A_CONSTRUCTION_BASIS_AUDIT_V1.md`）、[机读记录](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.json`）、[示意图](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_v1.png`）及[哈希清单](README.md#archive-access)（归档：`artifacts/m1a_fixed_A_construction_basis_audit_v1/m1a_fixed_A_construction_basis_audit_hashes_v1.json`）。不实施替代标定或参数搜索；M1-A 保持 **NOT PASSED**，M0 保持 **BLOCKED**。

#### 3.3.12 已完成：既有完整序列候选几何可视化交付

两分支 source frame `0..141` 已以固定根、共同相机/比例/范围同步展示；固定
`0/15/27/71/141` 含掌面和侧向视角。FK 十点与旧数组差为 `0 m`，GIF 编码 142 帧，无插值或 warmup
复制。登记 visual STL 均为 LFS 指针，故交付明确为骨架/landmark fallback，不提供表面穿透判断。
入口见[动画](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/m1a_full_sequence_candidate_synchronized_skeleton_landmarks_v1.gif`）、[固定帧图](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/m1a_full_sequence_candidate_fixed_frames_skeleton_landmarks_v1.png`）及[简要说明](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_geometry_visualization_delivery_v1/M1A_FULL_SEQUENCE_CANDIDATE_GEOMETRY_VISUALIZATION_DELIVERY_V1.md`）。不据图选择 mean 或批准正式质量；M1-A 仍 **NOT PASSED**，M0 仍 **BLOCKED**。

#### 3.3.13 已完成：登记 Sharpa visual mesh 精确恢复及既有候选表面展示

28 个登记 visual 引用的 pointer 精确对应 13 个去重对象、`5,954,492` 字节；已从登记仓库的现有 LFS
负载恢复至隔离目录并逐项验证 oid/字节数，原 pointer、URDF 和骨架交付不变。表面按登记 visual origin、
mesh scale 和 link FK 顺序接线；两分支 142 帧、固定五帧双视角、十点与残差线均已交付，FK/旧数组差为
`0`。入口见[资产表](README.md#archive-access)（归档：`artifacts/m1a_sharpa_visual_mesh_exact_recovery_v1/M1A_SHARPA_VISUAL_MESH_EXACT_RECOVERY_V1.md`）、[表面动画](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_synchronized_surface_v1.gif`）、[固定帧图](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_fixed_frames_surface_v1.png`）和[说明](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/M1A_FULL_SEQUENCE_CANDIDATE_SURFACE_VISUALIZATION_DELIVERY_V1.md`）。该图只支持视觉检查，不认证碰撞安全、物理执行、任务成功或 mean；M1-A 仍 **NOT PASSED**，M0 仍 **BLOCKED**。

`alpha=4e-3` 只是在论文同样的损失单位和**对 J 求和**时可原样作为初始候选。若实现改为对关键点
求平均，为保持同一相对权重应相应把平滑系数除以 `|J|`；若 MANO/Sharpa 点单位、尺度或 q 单位不同，
则必须先推导并记录换算，不能机械照抄。论文没有提供本项目 Sharpa 的这些适配事实，故它们是实施
前必须核验的接口/几何事项，而不是许可加入新抓取、接触或 affordance 方法。

| 事实缺口 | 对候选 M1-A 左手局部规格的影响 | 主要阻塞的后续范围 |
|---|---|---|
| 权威左 Sharpa runtime 模型/FK、22-joint 限位，以及历史 MANO layer/mean-pose 语义 | 候选 FK/限位接口已内部一致性 PASS，但未证明等价于权威 runtime；候选十点和局部标定仍不可当作冻结正式接口。 | M1-A |
| reconstruction-world↔Isaac、MANO root↔robot wrist 外参、双臂 IK/控制 | 不阻塞手局部 FK 规格；`A` 不得偷用这些未定 world 变换。 | M1-C、世界回放、M1-D/E |
| object valid 63/142、59/142 的填补/剔除政策 | 不阻塞只读的左 MANO 局部 FK 规格；不得为方便而推断物体位姿。 | object reference、完整训练/比较 |
| warmup/hold、G2 与真实 Ours runtime | 不决定左手几何对应；不能据此声称共同世界验收或公平结果。 | M0、正式比较、M1-D/E 认证 |

### 3.4 候选输入合同草案（仅供 reviewer 审核，未批准）

本节完成的是评审材料整理，不是历史合同复原、合同批准或 retargeting 实施授权。下文的
**VERIFIED** 仅表示已有证据直接支持相应窄结论；**PROPOSED** 表示可复现但尚未选择的工程候选；
**UNKNOWN** 不得用默认值补齐。v4 decoder 验证和证据绑定已经结束，不因本草案扩大测试范围。

#### 3.4.1 A. 可直接引用的既有事实

| 项 | 已有事实 | 结论边界与证据 |
|---|---|---|
| 交接包 perception artifact 身份 | `D:/UCBP/_inspection/pour17_baseline_bundle_20260829/pour17/perception/pour17_perception.npz`；SHA-256 `28e13c2454ec010945119e9611498bc72e733a519f8f9b6c6ad7a5a2f98234d7`。 | 字节身份 **VERIFIED**；它是当前候选工程输入，不等于已绑定的历史 HaWoR run 或权威 Ours runtime。见[静态 schema/valid 证据](README.md#archive-access)（归档：`artifacts/m1a_static_geometry_audit/m1a_static_geometry_audit.json`）。 |
| M1-A 可读 schema | `left_mano_trans (142,3) float32`、`left_mano_rot (142,3) float32`、`left_mano_pose45 (142,45) float32`、`left_mano_betas (142,10) float32`、`left_mano_valid (142,) bool`。 | 字段/shape/dtype **VERIFIED**；只允许读取左手同名字段，不能用 `left_hand_q`、Ours reference、认证 `q` 或 confidence 补齐。 |
| valid 与索引 | `left_mano_valid` 为 142/142 true；`frame_ids=0..141`，`timestamps` 全 NaN。 | 数组事实 **VERIFIED**；valid 必须原样保留。绝对时间 **UNKNOWN**，不得把派生相对时间写成观测时间戳。 |
| 候选 MANO 模型 | `MANO_LEFT.pkl` SHA-256 `c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30`。v4 指定官方 `ManoLayer` 源码 SHA-256 `e9a38f5df9b54ec82e631dbab678ebb9e7276210a76ee88de6df771bf6d77927`。 | 模型/对拍层身份和指定条件下的数值一致性 **VERIFIED**；历史 HaWoR 是否使用该模型/层仍为 **UNKNOWN**。详细结果只引用 [M1-A audit §2](M1A_INPUT_GEOMETRY_AUDIT.md#2-当前有效证据)，不在此重抄。 |

原始 perception NPZ 与 v4 七案例测试 NPZ 必须严格区分：

| artifact | 身份与内容 | 允许用途 | 禁止解释 |
|---|---|---|---|
| 交接包原始 perception NPZ | 上表 `pour17_perception.npz`，SHA-256 `28e13c…34d7`；142 帧序列，带 `left_mano_valid`、`frame_ids` 和 `timestamps`。 | 候选 M1-A 序列输入及 provenance 审核。 | 不能因被 v4 抽取了三帧就称为历史 HaWoR 语义已恢复或 M0 已绑定。 |
| v4 七案例测试 NPZ | `m1a_mano_inputs_v4.npz`，SHA-256 `91b10f1a9e32135b8e9b6b4d670c000af4df9787a7ca443ef4fbf602e8df41a4`；四个构造案例加原始序列帧 0/71/141，接口为 `case_names`、`global_aa (7,3)`、`pose45 (7,45)`、`translation (7,3)`、`betas (7,10)`。 | 仅用于已完成的 v4 decoder 等价性和证据绑定。见[绑定记录](README.md#archive-access)（归档：`artifacts/m1a_mano_crosscheck_v4/m1a_v4_evidence_binding.json`）。 | 它不是 142 帧 perception 序列、没有序列 valid 合同，不能替代或回写原始 perception NPZ，也不能作为 M1-A 正式输入。 |

#### 3.4.2 B. 两个解码候选的精确差别

以下内容直接来自[冻结 v4 配置](README.md#archive-access)（归档：`configs/m1a_mano_crosscheck_v4.json`）（SHA-256
`2cbd797ae58dfdd57cad9fb58051939033e51c18f702e41d63f539843e185db0`）和[候选配置](README.md#archive-access)（归档：`configs/m1a_left_landmark_calibration_candidate_v1.json`）（审计记录 SHA-256
`9f71748a78e895fdce4d574210ac6291fcca729d8f8d7e764af71cab5ac0b518`）。它们是两个且仅两个可复现
**PROPOSED** 设置，不是历史事实；本轮不产生第三种设置，也不按图像、residual 或任务结果排序推荐。

| 候选 id | NumPy mean 行为 | 指定官方层对拍行为 | 历史结论 |
|---|---|---|---|
| `proposed_raw_pose` | `local_aa = pose45`；`add_hands_mean=false`。 | 原始 `pose45` 不预加 mean；`flat_hand_mean=true`。 | 历史是否使用该分支 **UNKNOWN**。 |
| `proposed_plus_hands_mean` | `local_aa = pose45 + MANO_LEFT.pkl:hands_mean`；`add_hands_mean=true`。 | 原始 `pose45` 不预加 mean；`flat_hand_mean=false`，由层内部采用 mean。 | 历史是否使用该分支 **UNKNOWN**。 |

两者共同设置如下：

- 同一个 `MANO_LEFT.pkl`（SHA-256 `c4022f…6a30`），`side=left`、`use_pca=false`；
- `left_shapedirs_x_flip=true`，即构造层后仅一次执行左 `shapedirs[:,0,:] *= -1`；历史 HaWoR
  是否采用同一 shapedirs 预处理仍为 **UNKNOWN**；
- 模型/输入单位按米处理，不乘除 1000；`betas` 仅通过 shapedirs 使用一次；
- `root_rot_mode=axisang`、`joint_rot_mode=axisang`、`center_idx=null`、`root_palm=false`、
  `share_betas=false`；`left_mano_rot` 作为 global/root axis-angle，`left_mano_trans` 在 LBS 后加一次；
- v4 对拍的 `device=cpu`、`dtype=torch.float32` 是验证环境绑定，不自动成为未来实现/runtime 合同。

候选配置中的 `calibration_variant=proposed_raw_pose` 只记录既有静态标定图采用的诊断分支；它不构成
mean 分支选择或合同批准。

#### 3.4.3 C. 手局部输入边界

| perception 字段 | 现有候选 decoder 中的用途 | M1-A 合同边界 |
|---|---|---|
| `left_mano_betas` | 形状参数，经候选左 shapedirs 生成 shaped mesh/rest joints。 | 可用于手形；模型身份、mean 分支和 shapedirs 处理必须一起记录。 |
| `left_mano_pose45` | 15 个局部关节 axis-angle/rotvec；是否叠加 `hands_mean` 由两个候选分支决定。 | 可用于局部姿态；mean 未选择前不能生成获批目标。 |
| `left_mano_rot` | 当前候选代码把它作为 MANO global/root axis-angle。 | 只可按候选 decoder 语义记录/解码；不能据此声称 reconstruction world 或 Isaac world 对齐。 |
| `left_mano_trans` | 当前候选代码在 LBS 后作为 global/root translation 加一次。 | 同上；历史完整 root/translation 语义仍为 **UNKNOWN**。 |
| `left_mano_valid` | 选择有效输入帧。 | 只控制序列有效性；不得替代物体 valid、confidence 或世界对齐证据。 |

历史 HaWoR/world/root 解释与新的工程候选必须分开：

- 历史模型、mean、shapedirs、完整 root/translation、reconstruction-world→Isaac、MANO root→robot
  wrist、wrist→EE 和 `hand_root_offset_mano` 解释仍为 **UNKNOWN**；本轮结果不改变这些状态。
- 新的 hand-local 工程候选 `proposed_wrist_zero_global_hand_local_v1` 为 **PROPOSED**。对每个样本和
  每个既有 variant，令 `P0=D(beta,pose45,0,0;variant)`、`o0=P0.joints21[0]`，并定义
  `vertices_H=P0.vertices-o0`、`joints21_H=P0.joints21-o0`。H 原点是该样本/variant 解码的 wrist，
  H 轴是 global rotation 为零时的候选模型轴；保留原始 `betas/pose45`、variant mean 行为和米制手尺寸。
- 该候选不额外减 `hands_mean`，不做尺寸归一化，不减掌心均值，不读取机器人/物体坐标、
  `hand_root_offset_mano` 或 Sharpa `A`。

已验证 decoder 的根旋转/平移源码结构支持 reviewer 指定关系。固定七案例、两个 variant、
`global_aa=[0.2,-0.1,0.15] rad`、`trans=[0.02,-0.03,0.04] m` 下，vertices 与 joints21 的
`R^T(Pglobal-oglobal)=PH` 坐标不变性检查 **PASS（14/14）**；总体最大 max-abs
`8.326672684688674e-17 m`，最大 RMS `1.3524489597802808e-17 m`，且 shape/finite/wrist-zero 全通过。
证据见[诊断报告](README.md#archive-access)（归档：`artifacts/m1a_hand_local_candidate_v1/m1a_hand_local_candidate_v1.json`）；该 14 项是新的
hand-local 坐标检查，不计入原 v4 42 项，也没有重跑官方 layer。

该 PASS 只证明上述候选定义在测试范围内消除整体旋转和平移并保留传入的 `betas/pose45`。

与该 H 定义配套的新固定标定候选当前记录在
[`m1a_wrist_zero_hand_local_calibration_candidate_v2.json`](configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json)，状态仍为 **PROPOSED**。v2 只修正 v1 的公式适用范围，运行逻辑和全部数值不变：
一般样本必须用 `D(beta,pose45,0,0;variant)` 减去**同次解码**的 `J0`；只有下面的固定 canonical 才写作
`Pcanonical_H=Pcanonical-w_ref`。已哈希 v1 配置与代数证据保持不变。标定只用一次固定
canonical：MANO `beta=0,pose45=0,global_aa=0,trans=0`、`proposed_raw_pose`、既有 left shapedirs；
Sharpa 候选 URDF `q=0`，目标坐标根保持 `left_hand_C_MC`，`scale=1`。raw canonical 只是保留旧标定
canonical，不选择真实输入 mean 分支。

旧脚本/产物已核实直接用未减 wrist 的 canonical MANO 点构造 `A_old`。令其 full-precision wrist 为
`w_ref=[-0.09566993092407175,0.006383428857461437,0.006186305280135195] m`，则新旧关系为：

`p_old=p_H+w_ref`，`R_new=R_old`，`t_new=t_old+R_old w_ref`。

重算结果 `R_new` 与 `R_old` max-abs 差为 `0`；`t_new` 为
`[-0.0020081801195304307,-0.001578327634321078,0.009231794169036217] m`，与解析转换 max-abs 差
`4.336808689942018e-19 m`；canonical 21 点两种表示的映射 max-abs 差
`2.7755575615628914e-17 m`。B/R 正交、`det(R)`、掌原点和掌基方向检查也均在预登记 `1e-10`
容差内，详见[新旧坐标诊断](README.md#archive-access)（归档：`artifacts/m1a_wrist_zero_hand_local_calibration_candidate_v1/m1a_wrist_zero_hand_local_calibration_audit_v1.json`）。旧 A 在旧输入坐标下保留为有效证据；只是不能直接与 wrist-zero 点混用。

同一个 `A_new` 已应用于上一轮七样本、两个 mean 分支的 hand-local diagnostic arrays，14/14 的 shape/
finite 检查通过；这不比较或选择分支。非零 beta 样本仍先按各自同次解码 wrist 局部化，再使用这一个
canonical 固定 A，不逐 beta/逐帧/逐分支重标定。九项既有 landmark 和指定 thumb 候选、最终 Sharpa
runtime FK/22 关节顺序与限位仍需独立批准；结构/代数 PASS 不代表解剖匹配、正式输入合同或运行授权。

因此，候选手局部工程验证与正式共同 benchmark 验证分开：前者不能自动解除 M0，后者也不能反向替代
MANO/Sharpa 局部输入合同。仅冻结 decoder/mean 合同不足以启动 M1-A 优化器。

#### 3.4.4 D. 三个相互独立的待决合同

| 待决合同 | 必须冻结的内容 | 主要阻塞的后续工作 |
|---|---|---|
| 1. MANO 解码输入合同 | 交接包 perception 的正式输入身份；历史语义处置路径；两个 mean 分支之一；模型、left shapedirs、单位、root 参数及字段使用。 | 阻塞生成有确定语义的 MANO 21 点/landmark 序列；单独获批仍不能启动优化器、批准十点/A 或解除 M0。 |
| 2. Sharpa 十点与固定局部坐标合同 | 五 tip、四个长指 middle、已结构核验但未批准的 `proposed_thumb_internal_segment_midpoint_v1`；最终 Sharpa runtime 模型/FK；22 名称顺序/限位；hand-local 构造、固定 `A` 和 `s=1`。 | 阻塞定义论文式（1）的获批完整 `J`、同帧误差和可执行 M1-A 目标；单独获批不选择 MANO mean，也不认证世界回放/共同 benchmark。 |
| 3. M0 共同世界与 evaluator 合同 | 权威 runtime 或新 benchmark 身份、USD/资产/物理/reset、world transform、共同时间/物体无效帧政策、warmup/hold、G2 callback 和 evaluator。 | 阻塞 M1-C、世界 replay、训练、正式公平评测及 M1-D/E 认证；不能因候选手局部验证通过而自动解除。 |

任一合同获批都不得解释为其余两项同时获批；三个 gate 的证据与批准记录必须分别留档。

### 3.5 验收检查与建议阈值

完整 142 帧候选序列已经运行，且其中的 provenance、finite、候选限位和逐帧诊断已有记录；但这些仍是
未批准合同下的候选证据。M1-A 只能在下列项目按正式合同重新绑定并留档后评估，当前正式验收状态为
**NOT PASSED**，不得把候选执行检查 PASS 直接迁移为正式验收 PASS。

| 类别 | 自动检查 | 建议阈值与依据 |
|---|---|---|
| 输入/provenance | source hash、字段 shape/dtype、T、`frame_ids` 单调、timestamps 全 NaN 事实、valid mask 原样保存 | 完全一致；这是 schema 完整性，非性能阈值。 |
| 数值/限位 | finite、q 名称精确匹配、每帧 q 位于读取的 `[lower,upper]` 内 | 0 个 NaN/Inf、0 个超限；仅允许 `1e-6 rad` 浮点比较容差，不允许后处理静默 clip 掩盖违例。 |
| 左右/帧 | 输入只读左 MANO；输出只含 `left_*` 名称；输入/输出 `valid`、`source_frame`、`t_s` 一一对应 | 0 个字段/索引不一致；这是防镜像和错序的硬门。 |
| 几何 | 报告每帧论文定义的 tip + middle-phalanx 点的绝对误差、RMS、95/99 percentile、每关节帧间增量和 objective residual | 数值通过阈值暂不设定：MANO landmark map、尺度与 `A` 未冻结。先固定视觉/尺度定义，再在看结果前登记阈值，避免事后调参。 |
| 可视化 | 同一坐标下展示 open/approach、最大闭合、交互三帧：左 MANO keypoint、左 Sharpa FK、cup mesh；另做五指命名颜色图 | 人工检查必须确认五指无镜像/交叉、tip 朝向 cup 互动区。没有冻结 world transform 前不声称 absolute-world 对齐。 |
| 可重复性 | 同 input/config 在相同 CPU/runtime 重算 q 和 diagnostics | q 输出逐元素一致或预先登记的确定性容差；不涉及训练随机性。 |

“0 超限”和“0 schema 错配”来自数据接口安全要求；几何毫米阈值未给出，是因为当前坐标/
尺度/landmark 事实不足。给一个看似合理的 mm 数会伪造证据，不符合本 M0 规则。

## 4. 最少决策清单（均待 reviewer / 项目决定）

以下每项都是独立决定；本文件不替项目选择。

1. **历史 MANO 语义的处置路径**
   - 需要决定什么：只保留两条路径——获得新的历史源码/运行材料后恢复；或项目明确采用新的候选输入合同并持续标注“历史语义 UNKNOWN”。
   - 已有证据：已完成限定范围内的交接包、当前包装/平滑/导出/消费源码溯源；详见 [M1-A audit §5](M1A_INPUT_GEOMETRY_AUDIT.md#5-pour17-hawor-输入语义溯源)。
   - 缺少什么：历史 source/config/environment/model/run manifest、实际 producer 实现与可绑定运行材料。
   - 选择后能够推进什么：有新材料时可恢复历史核验；采用新合同时可继续审核项目定义的工程合同。
   - 哪些结论仍不能声称成立：采用新合同不能称为历史复原；仅获得新材料也不能在完成绑定核验前称历史语义 VERIFIED。

2. **MANO mean 分支**
   - 需要决定什么：在 `proposed_raw_pose` 与 `proposed_plus_hands_mean` 中选择一个；不允许隐式默认或第三分支。
   - 已有证据：两个分支在冻结 v4 条件下均通过 decoder 等价性与证据绑定。
   - 缺少什么：能把任一分支绑定到历史 Pour17 HaWoR producer 的材料，或项目对新工程语义的明确批准。
   - 选择后能够推进什么：冻结候选 MANO decoder 的局部 pose 解释并生成一致的候选 MANO 几何。
   - 哪些结论仍不能声称成立：不能声称所选分支是历史事实、Sharpa 十点/A 已批准、M1-A 优化器可启动或 M0 已解除。

3. **Sharpa 十点与固定局部坐标**
   - 需要决定什么：批准或拒绝五 tip + 四长指 middle 及已结构核验的 `proposed_thumb_internal_segment_midpoint_v1`，并冻结 hand-local 构造、固定 `A`、`s=1`、最终 Sharpa runtime FK、22 名称顺序和限位。
   - 已有证据：官方 MANO 21 点顺序已核验；九项既有点、指定 thumb 候选的直接父子链/URDF 端点/有限扰动 FK，以及 wrist-zero 固定标定坐标换算均已有记录。
   - 缺少什么：thumb 候选的项目批准及解剖/作者语义证据、正式 hand-local 输入合同、权威 Sharpa runtime 模型/FK 绑定及局部坐标/镜像确认。
   - 选择后能够推进什么：与已批准 MANO 解码合同共同定义完整 `J` 和 M1-A 优化器输入/目标，之后才可进入实现评审。
   - 哪些结论仍不能声称成立：不能声称是作者原始 mapping、absolute-world 对齐、共同 benchmark 通过或 M0 自动解除。

4. **M0 共同世界与 evaluator**
   - 需要决定什么：采用历史 runtime 复原，还是冻结交接包为新的共同 benchmark 并让双方重评；同时冻结 world transform、时间/物体无效帧政策、warmup/hold、G2 callback 与 evaluator 版本。
   - 已有证据：交接包本地/远端文件哈希内部一致，候选 world/reset/evaluator 参数已审计。
   - 缺少什么：实际 Ours train/eval manifest 与权威 runtime 对拍，以及双方共同采用各项 evaluator 细节的证据。
   - 选择后能够推进什么：在完成对应绑定和共同评测准备后推进 M1-C、replay、训练及正式比较。
   - 哪些结论仍不能声称成立：任何 M1-A 手局部候选结果都不能单独证明 M0、公平 benchmark、M1-D/E 或任务性能成立。

输出 NPZ 路径/版本/hash manifest 以及几何误差、平滑度、可视化阈值，只能在上述接口事实获批后作为普通
实现细节处理；它们不是当前输入合同决策的替代物。

## 5. Gate 状态

| Gate | 状态 | 说明 |
|---|---|---|
| M0-A common world/evaluator | **BLOCKED** | 已核验交接包内部一致，但无法核验它等于当前 Ours runtime。 |
| M0-B common upstream input/fairness | **BLOCKED** | 输入来源/污染项已分类；有效帧、world transform、DexPilot 与权威 runtime 的关系尚未冻结。 |
| M1-A left Sharpa retargeting | **NOT PASSED** | 已完成完整 142 帧 T 候选几何诊断、限位/残差归因、tip–middle 点对距离、固定 q 后验刚体及固定 A 构造依据审计；这些只提供候选模型的局部、点对、反事实配准与实现可复现性证据。跨模型锚点对应仍未认证，正式 reference、权威 runtime 对拍、输入/十点/时间策略合同及姿态/物理验收均未通过。 |
| M1-B right Sharpa retargeting | **NOT TESTED** | 不在本 session 范围。 |
| M1-C wrist/arm mapping | **NOT TESTED** | 不在本 session 范围。 |
| M1-D unilateral replay | **NOT TESTED** | 不在本 session 范围。 |
| M1-E synchronized bilateral replay | **NOT TESTED** | 不在本 session 范围。 |

