# ViViDex baseline：Pour17 交接入口

**截止：2026-09-13 01:38:28 Asia/Shanghai（UTC 2026-09-12 17:38:28；PDT 2026-09-12 10:38:28）。文档更新时还有一个ViViDex训练正在进行。接手同事或AI agent同步后，请先找Dennis索取训练结束后的报告、最终checkpoint/state、预算账本与SHA256，再判断后续工作。不要把这个快照当作训练完成，也不要重启已有作业。**

本次只同步ViViDex代码身份、文档和上传集合；没有启动实验、改功能代码、push或更改其他track。建议GitHub目标为`RL/Baselines/ViViDex/`，保持本目录结构；共享GPU规则只上传到仓库根`GPU_POLICY.md`，由team leader统一协调一次，不覆盖其他track结论。目标仓库为[DennisXu626/Baseline-for-RL_correction](https://github.com/DennisXu626/Baseline-for-RL_correction)，只读核实其存在、公开且默认分支main；这不是本适配已提交/已上传的证明。

## 阅读顺序与权威入口

1. 本README → [AGENTS.md](AGENTS.md) → [当前状态](REPRODUCTION_STATUS.md)。
2. [BASELINE_PLAN.md](BASELINE_PLAN.md)的H顶部：唯一当前工作包；F保留W1/S1等方法合同。A–G、旧WP指令和已耗用额度是历史，不能自动执行。
3. [交接实时快照](verification/handoff_20260913_snapshot.json)及[上传清单/文件身份](verification/handoff_upload_manifest.json)。
4. 原版MuJoCo研究仅按[已结束的原版协议](POUR_TWO_SEED_PROTOCOL.md)和[固定300回合结果](verification/seed_0_fixed_eval.json)理解；需要几何来源时再读[M1-A审计](M1A_INPUT_GEOMETRY_AUDIT.md)，不默认重跑。
5. GPU运营以[根目录GPU_POLICY.md](../../../GPU_POLICY.md)为准。共享许可不增加科学预算。

## 当前结论与此次增量

任务是建立human-video → RL correction的ViViDex双臂/双Sharpa baseline，当前只做Pour17；sweep、take off bottle cup、clean plate尚未推进。自身MANO/物体视频参考、W1/S1、十点手指retarget和双臂IK进入Isaac，policy输出58维关节位置目标。比较范围是状态RL能否学会任务，视觉student不在本实验中。不能引入Ours/Dexonomy或H2S2R成功关节轨迹作为隐藏输入。

| 状态层级 | 截止时的事实 |
|---|---|
| 已实现 | 58D PPO、544维观测、增强/curriculum、物理接触与诊断、保存/恢复、新根在训练/内部验证一致接线 |
| 已部署 | 本次清单中的71个部署Python/JSON/YAML与当前A6000 runtime逐项SHA一致；本地和运行副本没有可用适配Git HEAD，不能编造commit |
| 已运行/封存 | seed1701新world模型57,344→262,144；262,144确定性右/左边界误差中位96.85/163.68mm，stochastic195.76/218.96mm；两模式均联合预抓取0/25，训练累计0/7,343 |
| 已验证通过 | 根/重置身份、固定躯干与桌47.725mm间隔；PPO更新、模型保存/加载动作一致；源MANO world→W1→reference→reward抽查无接线分歧。均不是任务成功 |
| 运行中 | DEV_TO_FIRST_CURRICULUM_V1已获批，从262,144恢复，快照577,536、stage0；524,288双模式诊断已执行。目标1,003,520或本包4h训练墙钟；最终结果尚未封存 |
| 未解决/未放行 | 联合门槛未通过；源MANO+W1有桌内点；随机探索时臂桌接触；操作阶段学习和最终共同物理合同未认证。M0 BLOCKED、M1-A NOT PASSED。正式seed0/3M及最终512未启动/未放行 |

上次同步基线是已封存262,144结果与reviewer待批建议；随后实施方记录了用户批准、部署和启动。此次补齐：当前调度入口、独立百万步额度、已完成中点诊断、按GPU_POLICY共享准入/监督、恢复绑定，以及当前进程/源码/依赖的实时只读证据。没有改变方法或新定下一路线。

原版MuJoCo已独立训练60,002,304步，固定300回合颗粒入杯比例均值0.846944；seed1后来被用户取消。这不是双手Pour17二值成功率，也不是当前policy。原版campaign已关闭，禁止照历史脚本重新启动。

## 代码、依赖与运行边界

当前本地根`D:/UCBP/RL/Baselines/ViViDex`；服务器适配工作目录不是Git checkout。训练方法身份采用reference/overlay/URDF/源码SHA和checkpoint绑定；当前上传集合保留完整部署Python文件，包括历史WP脚本，因为`development_binding.py`枚举同目录全部Python计算身份。删除旧脚本或改目录内代码会改变绑定。保留不代表它们仍有执行授权。

主要入口：`adaptation/pour17_v1/train.py`（PPO）、`internal_eval.py`（原定curriculum验证）、`policy_diagnostic_eval.py`（固定双模式开发评测）、`development_curriculum_job.py`（当前已启动作业的监督器）、`wp4_world.py`（根overlay）。`env.py/scene.py/reward.py/augmentation.py/curriculum.py`为方法实现。`reference.py`、`wp2_prepare.py`和`wp4_reference.py`承担自身参考准备/合成/臂重建；读取既有hash绑定输入，不是取得一个成功policy的快捷入口。

**运行环境必须分开：** 当前Isaac Python为`/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/venv/bin/python`。本次实时metadata：Torch2.7.0+cu128、SB3 2.7.0、Gymnasium1.2.1、NumPy1.26.0、SciPy1.15.3、Isaac Sim5.1.0.0、IsaacLab0.54.2、NLopt2.7.1；Python版本在快照。IsaacLab源`/home/kailang/opt/IsaacLab-5.1`，HEAD `37ddf626871758333d6ed89cf64ad702aef127d0`。这些是现成环境版本，不是经过空白机器重装验证的requirements。

原版MuJoCo使用独立`.../vividex_mujoco/env/bin/python`、SB3 1.1.0/Gym0.23.1等旧依赖，详情在状态文档；不可把它的依赖安装到当前Isaac环境。原版source HEAD `9790140170d8be49b828ab214ce3537a2475fcea`不代表无本地适配改动。

CPU参考重建另需配置中指定的MANO左右模型、原始perception、URDF、已选左手候选数组、decoder/FK/NLopt代码和官方ManoLayer文本。配置仍是实际使用过的D盘/服务器路径，未为公开仓库伪造“开箱即用”路径。迁移前准备下节外部资源，并在新的配置副本里显式映射路径和校验hash；不得改已封存配置/绕过hash。此次没有安装依赖或验证跨机器重建。

## A6000路径与已用命令

以下别名只用于阅读；完整展开路径和真实进程命令也在快照中：

```text
R=/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco
A=/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1
W=/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01
C=/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/curriculum_runtime_20260913_01
O=/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/curriculum_runtime_20260913_01/artifacts/pour17_v1/development_curriculum_20260913_01
```

本次只读确认`C`存在、调度PID2527603及训练PID2527650运行；物理GPU0 UUID `GPU-1d992e9e-0787-f30e-3b2b-b5ea38a7897a`，子进程cuda:0。这个映射是当时事实，下一次准入不固定GPU0/1。资源规则要求按初始化/训练/并存评测峰值计算余量、约10秒采样与持续监督；设置`RL_ISAAC_NO_GUARD=1`只作用本任务子环境，不删除别人锁。

**已实际使用的当前启动命令，仅供追溯，禁止因看到命令重复启动：** 在`C`中由`A/venv/bin/python -m adaptation.pour17_v1.development_curriculum_job`调度；实际train完整argv在快照`processes[].command`，其中包含`--root-overlay`、`--resume`、`--total-timesteps 1003520`和`--development-curriculum-approved`。原输出目录已存在；该脚本本来就是一次性实验入口，不是通用任意路径启动器。

当前验证：中点`O/eval_midpoint_stochastic/policy_diagnostic.json`、`O/eval_midpoint_deterministic/policy_diagnostic.json`各25回合已产生；调度器实际调用`policy_diagnostic_eval.py`，完整参数在其源码和训练固定诊断事件中。包末固定诊断以及1,003,520内部curriculum验证尚未在本快照确认。`batch_eval.py`/`eval_entry.py`是登记外部评测接口，旧WP2有开发执行证据，但当前world的正式任务评测尚未认证，不能把它们写成已经跑通最终512的命令。

已验证的262,144模型加载方式见本地封存报告（归档：`artifacts/pour17_v1/development_extension_20260912_01/DEVELOPMENT_EXTENSION_REPORT.md`；只加载模型，不创建环境、不训练）：

```bash
cd /media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/extension_runtime_20260912_01
CUDA_VISIBLE_DEVICES=1 /media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/venv/bin/python -c "from stable_baselines3 import PPO; m=PPO.load('artifacts/pour17_v1/development_extension_20260912_01/training/final_model.zip',device='cuda:0'); print(m.num_timesteps,m.development_binding)"
```

上面是封存报告实际验证过的历史命令，本次未重跑；物理GPU1仅是当时记录，下一次操作须重新按GPU_POLICY准入。预期计步262144、绑定存在；当前运行的新模型不能用这个旧计步标签替代。

## Archive access：不上传的大资源与访问前提

<a id="archive-access"></a>

访问前提：Dennis为同事分配A6000 SSH/项目文件权限；历史端点`kailang@128.32.164.89`仅用于定位，本仓库不分发账号密码、私钥或token。资源可在服务器直接使用；需要下载时先核对封存manifest，复制到自有目录。示例`scp <AUTHORIZED_SSH_TARGET>:<ABSOLUTE_RESOURCE_PATH> <LOCAL_DESTINATION>`是未执行模板，不能把占位符当有效凭据。

| 资源 | 绝对路径／来源 | 作用与校验 |
|---|---|---|
| 当前运行代码 | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/curriculum_runtime_20260913_01` | 上传71文件相对路径与服务器相同；SHA在快照，部署包7f3c83bc…a82eb7 |
| 当前训练／日志／中点诊断 | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/curriculum_runtime_20260913_01/artifacts/pour17_v1/development_curriculum_20260913_01` | `curriculum_job.json`、`development_training_ledger.json`、`training/progress.json`、`training/rollout_windows.json`、中点eval目录；运行中可变，不提前计算最终checkpoint hash |
| 最近封存checkpoint/state与完整证据 | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/extension_runtime_20260912_01/artifacts/pour17_v1/development_extension_20260912_01` | `training/final_model.zip` SHA92868d65…b19483、`training/training_state.pt` SHA6de5b407…c5；之前/之后双模式eval及账本。完整最终hash见轻量delivery_manifest |
| WP4自身参考 | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/wp4_20260912_01/artifacts/pour17_v1/wp4_20260912_01/reference_build_v1/reference_wp4.npz` | SHA34c8f30eaf37a674d5eabf3943218cdba5dd1612014d87b8f9055c45978fe649；自身目标/手指/臂参考，训练必需 |
| 登记bundle | `/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17` | `world/`USD/canonical、`perception/`、`evaluator/run_eval.py`和`progress.py`、`reference/`登记评测输入；既有访问权限，不能只拷贝公开代码就运行 |
| bundle原始归档 | `D:/UCBP/pour17_baseline_bundle_20260829.tar.gz` | SHA77edc03ca5be028fcbd8b6d978fd1a4072582b5c9904d603f5c86b29a70210f6；向Dennis获取，不上传 |
| URDF及关联mesh | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/adaptation_pour17_v1/assets/vega_1p_sharpa_fix.urdf`；本地`D:/UCBP/RL/Worktrees/RL-Correction-Step4-Publish/datasets/vega_urdf/vega_1p_sharpa/` | URDF SHA968b41f8…db；mesh需关联目录，许可和资产访问由项目提供 |
| MANO模型及官方层 | `D:/UCBP/RL/Baselines/ViViDex/upstream/vividex_sapien/mano/assets/MANO_LEFT.pkl`、同目录`MANO_RIGHT.pkl`；源码`D:/UCBP/RL/Baselines/ViViDex/upstream/vividex_sapien/mano/manolayer.py` | 模型许可独立，不公开上传；精确模型/层hash见wp1_config。官方源码从既有checkout或官方仓库取得；不能假设MIT覆盖模型资产 |
| 左手候选与CPU历史来源 | `D:/UCBP/RL/Baselines/ViViDex/artifacts/m1a_full_sequence_candidate_diagnostic_v1/full_sequence_candidate_diagnostic_arrays_v1.npz`及`D:/UCBP/RL/Baselines/ViViDex/diagnostics/` | 重建参考所需，不是policy训练必需；向Dennis取得指定hash版本，未声称整套历史CPU输入已部署当前runtime |
| 历史WP视频/数值/报告 | `D:/UCBP/RL/Baselines/ViViDex/artifacts/pour17_v1/` | 每包自带交付manifest；WP4对应服务器`W`，其他包远端精确位置由各包报告给出。历史文档中未入上传集合的相对路径，均以此本地track根定位，不能当GitHub已包含 |
| native MuJoCo代码/依赖/结果 | `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/pour_multiseed_source`、`/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/env/bin/python`、`/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco/outputs/pour_multiseed_60m_20260906_01` | 原版60M已关闭；原版模型hashd988b1b1…120见状态及已上传300回合JSON。不是当前Isaac policy |
| 原环境总表 | `D:/UCBP/RL/Reports/环境报告_ENVIRONMENT_INVENTORY.md` | 共享历史索引，不由ViViDex覆盖上传；当前运营规则用GPU_POLICY，现场版本以本次snapshot为准 |

历史MD中的“归档”链接会导向本节，原路径保留在旁边，用上述本地根／报告索引查找；它不表示该资源不存在，也不表示已经复制到GitHub。必须只取自己的track和明确共享资产，不读/改他方训练状态。

## 第三方来源与许可

官方源码：[MuJoCo实现](https://github.com/zerchen/vividex_mujoco)、[SAPIEN实现](https://github.com/zerchen/vividex_sapien)。本集合不复制完整upstream或资产；保留本地upstream的[MuJoCo LICENSE](upstream/vividex_mujoco/LICENSE)和[SAPIEN LICENSE](upstream/vividex_sapien/LICENSE)及来源说明。适配文件中的原来源注释保留，未新授予机器人/MANO/数据的分发许可，也未替team leader决定项目整体许可证。`mano_layer_source`用于既有decoder读取官方输出约定，完整第三方包按原checkout恢复，不能把一个层文件当独立可执行MANO发行版。

## 接手后首先做什么

先向Dennis索取运行结束的交付包，核对job是否COMPLETED、实际停止步数/墙钟、固定评测、curriculum事件、最终checkpoint/state及hash。若仍运行，仅按既有职责查看，不另起job。当前批准包结束后必须reviewer决定下一步；正式3M、其他task、改变目标/PD/reset/时间门槛或采用桌约束IK都没有因这次文档同步获得新授权。

已知缺口：当前百万步包尚未最终验收；没有从干净机器复现安装；登记资产/evaluator和CPU参考输入需权限获取；公开代码不是自包含资产分发包；共享正式runtime尤其PD/材料尚未冻结；外部最终任务评测尚未证明就绪。下方清单的代码import完整性是静态核对，不等于新运行测试。

## GitHub逐文件上传集合

本清单是完整维护集合，不仅是本次修改。不得上传整个artifacts目录或整个upstream目录。代码/配置保持服务器字节；MD仅整理状态与可移植索引。新建本README因为原来没有仓库入口；新建实时snapshot因为运行状态与远端hash不能由旧封存报告承载；新建upload_manifest因为需要确定、可验证的逐文件集合。所有新文件均用于本次交接，不是重复实验计划。

| 本地文件绝对路径 | 建议仓库相对路径 | 用途／为什么需要 | 本次是否更新 |
|---|---|---|---|
| [D:/UCBP/RL/Baselines/ViViDex/README.md](README.md) | `RL/Baselines/ViViDex/README.md` | 交接入口、资源/命令、逐文件上传索引 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/AGENTS.md](AGENTS.md) | `RL/Baselines/ViViDex/AGENTS.md` | 本track约束及向Dennis索取运行结果要求 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) | `RL/Baselines/ViViDex/REPRODUCTION_STATUS.md` | 当前/历史状态与结果边界 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/BASELINE_PLAN.md](BASELINE_PLAN.md) | `RL/Baselines/ViViDex/BASELINE_PLAN.md` | 已批准工作包与历史决策；仅H顶部当前有效 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/M1A_INPUT_GEOMETRY_AUDIT.md](M1A_INPUT_GEOMETRY_AUDIT.md) | `RL/Baselines/ViViDex/M1A_INPUT_GEOMETRY_AUDIT.md` | 几何/输入来源历史证据 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/POUR_TWO_SEED_PROTOCOL.md](POUR_TWO_SEED_PROTOCOL.md) | `RL/Baselines/ViViDex/POUR_TWO_SEED_PROTOCOL.md` | 已关闭native MuJoCo实验合同 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/diagnostics/m1a_full_mano_decode_audit.py](diagnostics/m1a_full_mano_decode_audit.py) | `RL/Baselines/ViViDex/diagnostics/m1a_full_mano_decode_audit.py` | 自身MANO解码/手指retarget必要CPU组件 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/diagnostics/m1a_sharpa_left_fk_candidate_v1.py](diagnostics/m1a_sharpa_left_fk_candidate_v1.py) | `RL/Baselines/ViViDex/diagnostics/m1a_sharpa_left_fk_candidate_v1.py` | 自身MANO解码/手指retarget必要CPU组件 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/diagnostics/m1a_sharpa_single_frame_solver_candidate_v1.py](diagnostics/m1a_sharpa_single_frame_solver_candidate_v1.py) | `RL/Baselines/ViViDex/diagnostics/m1a_sharpa_single_frame_solver_candidate_v1.py` | 自身MANO解码/手指retarget必要CPU组件 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/m1a_sharpa_left_fk_candidate_v1.json](configs/m1a_sharpa_left_fk_candidate_v1.json) | `RL/Baselines/ViViDex/configs/m1a_sharpa_left_fk_candidate_v1.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json](configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json) | `RL/Baselines/ViViDex/configs/m1a_wrist_zero_hand_local_calibration_candidate_v2.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/verification/handoff_20260913_snapshot.json](verification/handoff_20260913_snapshot.json) | `RL/Baselines/ViViDex/verification/handoff_20260913_snapshot.json` | 交接快照/上传集合、SHA与自检 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/verification/handoff_upload_manifest.json](verification/handoff_upload_manifest.json) | `RL/Baselines/ViViDex/verification/handoff_upload_manifest.json` | 交接快照/上传集合、SHA与自检 | 是 |
| [D:/UCBP/RL/Baselines/ViViDex/verification/seed_0_fixed_eval.json](verification/seed_0_fixed_eval.json) | `RL/Baselines/ViViDex/verification/seed_0_fixed_eval.json` | native固定300回合结果/已关闭campaign | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/verification/campaign_stopped_after_seed0.json](verification/campaign_stopped_after_seed0.json) | `RL/Baselines/ViViDex/verification/campaign_stopped_after_seed0.json` | native固定300回合结果/已关闭campaign | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/upstream/vividex_mujoco/LICENSE](upstream/vividex_mujoco/LICENSE) | `RL/Baselines/ViViDex/upstream/vividex_mujoco/LICENSE` | 保留第三方许可证 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/upstream/vividex_sapien/LICENSE](upstream/vividex_sapien/LICENSE) | `RL/Baselines/ViViDex/upstream/vividex_sapien/LICENSE` | 保留第三方许可证 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/fixed_endpoint_summary.json](artifacts/pour17_v1/development_extension_20260912_01/fixed_endpoint_summary.json) | `RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/fixed_endpoint_summary.json` | 262144封存结果轻量统计或完整产物索引 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/all_learning_windows.json](artifacts/pour17_v1/development_extension_20260912_01/all_learning_windows.json) | `RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/all_learning_windows.json` | 262144封存结果轻量统计或完整产物索引 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/delivery_manifest.json](artifacts/pour17_v1/development_extension_20260912_01/delivery_manifest.json) | `RL/Baselines/ViViDex/artifacts/pour17_v1/development_extension_20260912_01/delivery_manifest.json` | 262144封存结果轻量统计或完整产物索引 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/__init__.py](adaptation/__init__.py) | `RL/Baselines/ViViDex/adaptation/__init__.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/__init__.py](adaptation/pour17_v1/__init__.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/__init__.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/augmentation.py](adaptation/pour17_v1/augmentation.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/augmentation.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/batch_eval.py](adaptation/pour17_v1/batch_eval.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/batch_eval.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/bind_wp2_inputs.py](adaptation/pour17_v1/bind_wp2_inputs.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/bind_wp2_inputs.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/black_table_record.py](adaptation/pour17_v1/black_table_record.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/black_table_record.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/contact_events.py](adaptation/pour17_v1/contact_events.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/contact_events.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/curriculum.py](adaptation/pour17_v1/curriculum.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/curriculum.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/development_binding.py](adaptation/pour17_v1/development_binding.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/development_binding.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/development_curriculum_job.py](adaptation/pour17_v1/development_curriculum_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/development_curriculum_job.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/development_extension_job.py](adaptation/pour17_v1/development_extension_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/development_extension_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/development_job.py](adaptation/pour17_v1/development_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/development_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/env.py](adaptation/pour17_v1/env.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/env.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/eval_entry.py](adaptation/pour17_v1/eval_entry.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/eval_entry.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/execution_evidence.py](adaptation/pour17_v1/execution_evidence.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/execution_evidence.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/followup_approach_probe.py](adaptation/pour17_v1/followup_approach_probe.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/followup_approach_probe.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/followup_blocker.py](adaptation/pour17_v1/followup_blocker.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/followup_blocker.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/followup_saved_control.py](adaptation/pour17_v1/followup_saved_control.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/followup_saved_control.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/followup_source_check.py](adaptation/pour17_v1/followup_source_check.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/followup_source_check.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/geometry.py](adaptation/pour17_v1/geometry.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/geometry.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/internal_eval.py](adaptation/pour17_v1/internal_eval.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/internal_eval.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/policy_diagnostic_eval.py](adaptation/pour17_v1/policy_diagnostic_eval.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/policy_diagnostic_eval.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/reference.py](adaptation/pour17_v1/reference.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/reference.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/render_capture.py](adaptation/pour17_v1/render_capture.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/render_capture.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/reward.py](adaptation/pour17_v1/reward.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/reward.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/robot_table_contact.py](adaptation/pour17_v1/robot_table_contact.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/robot_table_contact.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/runtime_checks.py](adaptation/pour17_v1/runtime_checks.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/runtime_checks.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/sb3_vecenv.py](adaptation/pour17_v1/sb3_vecenv.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/sb3_vecenv.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/scene.py](adaptation/pour17_v1/scene.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/scene.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/schema.py](adaptation/pour17_v1/schema.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/schema.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/smoke.py](adaptation/pour17_v1/smoke.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/smoke.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/tensor_contact_probe.py](adaptation/pour17_v1/tensor_contact_probe.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/tensor_contact_probe.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/tests.py](adaptation/pour17_v1/tests.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/tests.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/train.py](adaptation/pour17_v1/train.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/train.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/train_state.py](adaptation/pour17_v1/train_state.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/train_state.py` | 当前部署运行/评测/参考组件；保持源码身份 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/__init__.py](adaptation/pour17_v1/wp1_compat/__init__.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/__init__.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/augmentation.py](adaptation/pour17_v1/wp1_compat/augmentation.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/augmentation.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/curriculum.py](adaptation/pour17_v1/wp1_compat/curriculum.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/curriculum.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/env.py](adaptation/pour17_v1/wp1_compat/env.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/env.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/eval_entry.py](adaptation/pour17_v1/wp1_compat/eval_entry.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/eval_entry.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/geometry.py](adaptation/pour17_v1/wp1_compat/geometry.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/geometry.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/reference.py](adaptation/pour17_v1/wp1_compat/reference.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/reference.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/reward.py](adaptation/pour17_v1/wp1_compat/reward.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/reward.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/sb3_vecenv.py](adaptation/pour17_v1/wp1_compat/sb3_vecenv.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/sb3_vecenv.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/scene.py](adaptation/pour17_v1/wp1_compat/scene.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/scene.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/smoke.py](adaptation/pour17_v1/wp1_compat/smoke.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/smoke.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/tests.py](adaptation/pour17_v1/wp1_compat/tests.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/tests.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/train.py](adaptation/pour17_v1/wp1_compat/train.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp1_compat/train.py` | 历史WP1兼容源，随部署保留；不启动 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_cpu_tests.py](adaptation/pour17_v1/wp2_cpu_tests.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_cpu_tests.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_delivery.py](adaptation/pour17_v1/wp2_delivery.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_delivery.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_job.py](adaptation/pour17_v1/wp2_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_prepare.py](adaptation/pour17_v1/wp2_prepare.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp2_prepare.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_finalize.py](adaptation/pour17_v1/wp3_finalize.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_finalize.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_job.py](adaptation/pour17_v1/wp3_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_reference_interpretation.py](adaptation/pour17_v1/wp3_reference_interpretation.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_reference_interpretation.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_scene_probe.py](adaptation/pour17_v1/wp3_scene_probe.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp3_scene_probe.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_delivery.py](adaptation/pour17_v1/wp4_delivery.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_delivery.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_job.py](adaptation/pour17_v1/wp4_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_reference.py](adaptation/pour17_v1/wp4_reference.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_reference.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_rollout.py](adaptation/pour17_v1/wp4_rollout.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_rollout.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_rollout_job.py](adaptation/pour17_v1/wp4_rollout_job.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_rollout_job.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_scene_probe.py](adaptation/pour17_v1/wp4_scene_probe.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_scene_probe.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_static.py](adaptation/pour17_v1/wp4_static.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_static.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_world.py](adaptation/pour17_v1/wp4_world.py) | `RL/Baselines/ViViDex/adaptation/pour17_v1/wp4_world.py` | 随部署保留的历史入口/组件；非新启动授权 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/dev_to_first_curriculum_v1.json](configs/pour17_v1/dev_to_first_curriculum_v1.json) | `RL/Baselines/ViViDex/configs/pour17_v1/dev_to_first_curriculum_v1.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/development_schema.json](configs/pour17_v1/development_schema.json) | `RL/Baselines/ViViDex/configs/pour17_v1/development_schema.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/observation_action_schema.json](configs/pour17_v1/observation_action_schema.json) | `RL/Baselines/ViViDex/configs/pour17_v1/observation_action_schema.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/wp1_config.json](configs/pour17_v1/wp1_config.json) | `RL/Baselines/ViViDex/configs/pour17_v1/wp1_config.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/wp1_train.yaml](configs/pour17_v1/wp1_train.yaml) | `RL/Baselines/ViViDex/configs/pour17_v1/wp1_train.yaml` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/wp2_train.yaml](configs/pour17_v1/wp2_train.yaml) | `RL/Baselines/ViViDex/configs/pour17_v1/wp2_train.yaml` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/RL/Baselines/ViViDex/configs/pour17_v1/wp4_root_overlay.json](configs/pour17_v1/wp4_root_overlay.json) | `RL/Baselines/ViViDex/configs/pour17_v1/wp4_root_overlay.json` | 实际配置/输入合同；历史路径需外部资源 | 否 |
| [D:/UCBP/GPU_POLICY.md](../../../GPU_POLICY.md) | `GPU_POLICY.md` | 共享GPU唯一规则；team leader只上传一次 | 否 |
