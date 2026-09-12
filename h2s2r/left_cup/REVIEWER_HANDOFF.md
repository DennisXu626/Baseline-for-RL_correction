# H2S2R LEFT/CUP — 团队接手入口

本次文档同步截止：**2026-09-13 01:39:34 Asia/Shanghai (UTC+08:00)**（UTC 2026-09-12 17:39:34）。仅左手抓杯穿模；不负责右瓶训练或Clean3。

## 当前权威结论

**未修复、无LEFT_PASS；D诊断数据有效，进程退出失败。没有已经批准待执行的新物理包。**

项目目标是保留H2S2R/FABRICS方法的UCBP adapted baseline；本目录仅提供左杯诊断，不是完整训练baseline，也不是ours训练入口。

| 层次 | 本次交接状态 |
|---|---|
| 已实现 | 固定历史目标回放、全左手接触记录、真实支持函数CPU测试、首12子步原子保存、三视角录像；移除运行中SCP批准依赖 |
| 已部署 | 下述隔离D运行树；6个支持脚本本次远端SHA与本机/执行清单一致。未集成V12/共享资产 |
| 已运行 | 新包2次App启动；第1次传感器注册支持失败，0control；第2次100controls/1200substeps。旧包另有3次前control支持失败 |
| 已验证 | D全部目标/drive路由、1200子步、3×101帧及raw完整；所有100步末29维左q与杯XYZ逐值等于历史warmup0记录 |
| 未验证/失败 | app.close超时；actual cooked整体几何、穿模修复、A碰撞移植均未验证；没有LEFT_PASS或学习成功率 |
| 运行状态快照 | 2026-09-13 01:33:41 Asia/Shanghai，只读ps未发现该隔离track replay/supervisor进程。这不是长期空闲声明 |

首接触在4.167ms，小指中节7.056977N，separation +0.0566mm；随后其他指节承力。261636条有效接触间距全正，control阶段0状态回写。正的contact-point separation不证明全表面不相交；输入网格交叉仍成立。D复现的是v7 hold0，不是v6 hold15的大范围掌部案例。不要把“张手抬杯”自动解释为无接触漂浮，也不把当前数据写成物理solver穿透已证实。

## 阅读顺序

1. 本文件：当前结论、环境、外部资源、停止边界。
2. [RESULTS.md](RESULTS.md)、[REVIEWER_REVIEW.md](REVIEWER_REVIEW.md)：实际结果与独立审阅。
3. [validation.json](validation.json)、[decision.json](decision.json)、[costs.json](costs.json)：验收与已用预算。
4. [A_D_DIFFERENCES.md](A_D_DIFFERENCES.md)、[A_SOURCE_INVENTORY.json](A_SOURCE_INVENTORY.json)：A的已知差异和缺件，不是移植授权。
5. [execution_versions.json](execution_versions.json)、[frozen_manifest.json](frozen_manifest.json)、[SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)：运行版本与只读取回源码对应。
6. [analysis/FIRST_12_SUBSTEPS.csv](analysis/FIRST_12_SUBSTEPS.csv)、[analysis/D_ANALYSIS.json](analysis/D_ANALYSIS.json)、[raw_evidence_index.json](raw_evidence_index.json)：首接触与完整raw获取。
7. [UPLOAD_MANIFEST.csv](UPLOAD_MANIFEST.csv)：唯一逐文件上传映射；共享GPU政策上传到仓库根GPU_POLICY.md，其他track不要复制另一份。

## 上次同步基线与增量

上次统一审阅为2026-09-13的REVIEWER_REVIEW.md（本机mtime 01:04:22 UTC+8，仅作文件更新时间，不伪称实验时刻）；之前executor交接文件mtime 00:40:37。上次审阅已经接受D数据、独立复算逐位复现，并关闭A移植。**从该基线到本次没有新实现/新实验/新放行。** 本次新增的是服务器身份与进程快照核对、最小源码依赖及许可证只读取回、上传映射、便携阅读入口和缺口说明。

此前一系列几何/接触包的历史顺序：初步网格差异→A/B10/历史姿态几何定位→3次前control记录支持失败→当前D有效记录。A=PMIN_P17s51；B10=成功源58目标的旧v10回放；D=v7 warmup0的100条29目标。三者不可混名。

已有旧计划H2S2R_LEFT_CUP_CONTACT_CAUSE_PACKAGE_20260912.md和H2S2R_LEFT_CUP_A_ENV_TRANSFER_PACKAGE_20260912.md均关闭，不能凭剩余数字恢复。它们留在本机reports作历史预算审计，本次不重复上传整套历史计划。A移植0、PPO/训练0。旧“必须空闲/无compute进程”运营规则已被[GPU_POLICY.md](../../GPU_POLICY.md)替代；不是增加实验次数。

## 代码身份与上传布局

实际隔离目录（不是Git checkout）：
`/home/kailang/experiments/baselines/h2s2r_left_cup_a_env_transfer_20260912`

源v7目录：`/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v7_uniformphysics`，只读Git HEAD `ead03335336857cfb0585e6e4ee1ebcb36e98cb1`，branch `baseline/h2s2r-dexmate-single-side`。该HEAD是底层旧仓库提交，**不能标成所有未提交适配代码的执行commit**；执行身份以逐文件SHA为准。隔离树、official及FABRICS副本本次无法从Git读取HEAD；第三方来源版本按THIRD_PARTY_NOTICES记录，不冒充本次Git核验。

本机 `D:/UCBP/RL/Baselines/RL-Correction-H2S2R` 的旧README介绍ours，不能用它作当前左杯入口。本次没有更改该共享克隆。

建议仓库路径统一为`h2s2r/left_cup/`。6个支持脚本和分析脚本在该目录根；本机`runtime_core/`内文件上传时去掉此中间目录，合并成同根`tasks/`、`rl_rebuild/`和`third_party_licenses/`。不要上传成同根下再套runtime_core，否则replay的相对synergy路径会错误。SOURCE_MANIFEST记录每个原服务器文件与本机相对路径、SHA；以UPLOAD_MANIFEST为实际目标路径权威。

新取得源码为实际D执行依赖的静态Python导入闭包，非完整训练仓库。未取回训练入口/旧诊断脚本/右瓶入口，不代表能从本目录启动训练。功能代码原字节保存，没有为上传重写硬编码路径。

## 环境与复现边界

2026-09-13 01:33:41 UTC+8在实际Python环境读取安装元数据：Isaac Sim5.1.0.0、torch2.7.0+cu128、numpy1.26.0、scipy1.15.3、imageio2.37.0、imageio-ffmpeg0.6.0、warp-lang1.16.0。**这是当前安装元数据，不是历史loaded-module完整锁文件**；有Isaac扩展路径覆盖的可能。原始实际启动命令见launch记录，既有PhysX ContactView版本107.3.26提供6数组。未制作/测试新机器pip安装方案。

Python `/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python`；IsaacLab `/home/kailang/opt/IsaacLab-5.1`；FABRICS `/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src`；official runtime子目录`third_party/h2s2r_official`需保留对应副本及许可证。

已验证实际启动环境：cwd=隔离运行根；PYTHONPATH=运行根:FABRICS src:运行根/third_party/h2s2r_official；TMPDIR=运行根/tmp。物理GPU0当时映射进程cuda:0，RL_ISAAC_NO_GUARD=1。以后必须先按仓库根GPU_POLICY测量准入，不固化GPU0。

以下为**历史真实命令存档，不是现在的启动授权**，原输出已存在不得复用：

```bash
/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python /home/kailang/experiments/baselines/h2s2r_left_cup_a_env_transfer_20260912/replay_contact.py --bundle_root /home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17 --commands /home/kailang/experiments/baselines/h2s2r_left_cup_a_env_transfer_20260912/command_inputs.npz --out /media/msc-auto/HDD/users/kailang/h2s2r_left_cup_a_env_transfer_20260912/D_replay02 --device cuda:0 --headless
```

该命令完成物理采样但在app.close卡住；不要把它标为全生命周期PASS。现有run_supervised.py固定服务器Python/bundle路径，只适用于该服务器部署。新位置运行须补齐下表依赖、恢复完整PYTHONPATH，再由reviewer批准新输出/预算；本次未验证GitHub全新clone可直接执行。

CPU/离线入口：`test_runtime_support.py`（真正函数行为测试，历史PASS，执行会写验证文件）；`analyze_d.py RUN --commands COMMAND_NPZ --out NEW_ANALYSIS_DIR`（真实已用解析形式，可从保存raw复算）。将RUN设为已保存D_replay02，commands使用同SHA的command_inputs.npz；使用新分析输出以保留历史。不得以CPU通过认证物理修复。

关键配置：1env，seed20260829，reference32，hold0，dt1/240，12substeps/control，100targets；只有左目标变化，右保持。cfg.py默认warmup15/质量是覆盖前值；本D最终hold0、物体质量/摩擦由实际构造与ours_physics覆盖，不能单读默认值宣称不同。全层请求/drive exact equality、0controller calls、1200行raw、3×101帧是本次诊断检查。

## external-resources

以下资源不默认上传GitHub。服务器绝对路径是定位信息，不含登录凭据；需要项目方授权的SSH账户/主机和个人私钥。自行配置`<SSH_HOST>`、`<SSH_USER>`、`<SSH_KEY_PATH>`，不要上传私钥或含token的环境文件。使用scp/sftp只读取回需要的文件，并按索引SHA验证。

| 资源 | 服务器/存储绝对路径 | 作用与准备 |
|---|---|---|
| D完整raw/视频/首12子步 | `/media/msc-auto/HDD/users/kailang/h2s2r_left_cup_a_env_transfer_20260912/D_replay02` | substeps.npz、contacts.npz、state_writes.jsonl、early_substeps、media/front.mp4/top.mp4/side.mp4；用raw_evidence_index与delivery_validation校验 |
| D启动/超时/资源日志 | `/media/msc-auto/HDD/users/kailang/h2s2r_left_cup_a_env_transfer_20260912/D_replay02_launch`及上述run/stacks.log | 完成后关闭卡住，不能重复采样掩盖；日志留服务器 |
| 完整冻结运行树 | `/home/kailang/experiments/baselines/h2s2r_left_cup_a_env_transfer_20260912` | 已部署可对照源码；包括本次未随Git上传的依赖/生成资产 |
| 任务bundle | `/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17` | perception/pour17_perception.npz、world及rest/reset/reference配置；不只复制cup单USD，保留依赖闭包 |
| 手与杯资产 | 上述bundle的`world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`、`world/cache/object_0.usd` | robot SHA c2bba25…71e347；cup20478176…909e71，全SHA见execution_versions/旧bundle身份记录；不因机器人MD5相同推断cooked等价 |
| 两侧FABRICS URDF | 运行根`/rl_rebuild/baselines/h2s2r/assets/vega_sharpa_left_fabric.urdf`与`vega_sharpa_right_fabric.urdf` | 按原相对路径复制；含资产生成来源，未默认公开分发机器人资产 |
| PCA两侧数据 | 运行根`/tasks/h2s2r_pour17/artifacts/synergies/left_synergy_pca5.npz`和`right_synergy_pca5.npz` | SHA取随代码上传synergy_manifest；固定回放仍构造controller，不能因不advance就省掉 |
| FABRICS和official依赖 | `/home/kailang/experiments/baselines/fabrics_h2s2r_reference`；运行根`/third_party/h2s2r_official` | 使用项目授权副本/按来源版本获取，许可证随本包保留；安装包不能替代该分支 |
| A原成功包（本机有，A运行环境仍缺） | `D:/UCBP/reports/H2S2R_POUR17_SUCCESS_REFERENCE_20260910/source_extracted/pour17_success_reference_20260910` | 用户/同事提供，含last.pth、world、cmds_trace；checkpoint不用来运行本D。A1杯cache/逐prim/filter/执行source仍待补齐 |
| 历史warmup0完整诊断 | `D:/UCBP/reports/h2s2r_run06_left_warmup_ab_20260905/warmup0_retry3/diagnostic.json` | 此包的command_inputs是它的100条目标提取；来源SHA见frozen_manifest。需要历史全姿态对照时向维护者取回 |

生成URDF和PCA等必要非Git资源本次按既有路径登记，不伪称已包含在上传源码或已在新机器复验。大型视频本机副本仍可从`D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/remote_evidence/D_replay02/media`观看。

## 接手第一步及停止边界

先复核本目录结论及首12子步，必要时用保存raw做离线接触/可见网格对应。当前已批准方向仅为使用已有数据继续判读；**没有已放行的下一物理、训练、A移植或初始化改变工作包**。具体碰撞表示/初态适配待reviewer与用户决策。A身份缺失不能用A2猜补；A定向搜索已经结束，不能默认为继续无限查找。

app.close问题未修；它不否定已有数据，也不授予重新采样权。其他track是否集成任何候选由各自reviewer接收，不能根据本目录给右瓶或Clean3放行。

## 上传、权限与已知缺口

目标仓库`https://github.com/DennisXu626/Baseline-for-RL_correction`本次未上传/push/改权限；网页读取未成功，因此未核对其已有目录/分支。建议路径需由Team leader合并确认。共享GPU_POLICY只使用仓库根一份；根README可以由负责人链接本入口，本track不覆盖其内容。

上传集合刻意不含旧接触失败脚本、其他track入口、整批图/视频/轨迹/ckpt、私钥/账号配置。现有LICENSE、NOTICE、THIRD_PARTY_NOTICES及两份上游许可证保留；外部机器人/物体数据是否可公开分发需其权利方确认，本次不纳入。

本次只读取回新源码快照，没有另写平行README。新增SOURCE_MANIFEST、UPLOAD_MANIFEST、HANDOFF_SYNC_VALIDATION用于文件级身份/映射/自检，旧实验manifest维持历史含义，不能用于校验本次已更新文档。详见HANDOFF_SYNC_VALIDATION中的实际检查与缺口。

## 完整上传文件表

逐文件列出，共69项；本表与UPLOAD_MANIFEST.csv对应。所有代码都是原字节快照，新增不等于功能修复。

| 本地文件绝对路径 | 建议仓库相对路径 | 用途／为什么需要 | 本次是否更新 |
|---|---|---|---|
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/replay_contact.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/replay_contact.py) | `h2s2r/left_cup/replay_contact.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_support.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_support.py) | `h2s2r/left_cup/runtime_support.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/contact_capture.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/contact_capture.py) | `h2s2r/left_cup/contact_capture.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/camera_delivery.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/camera_delivery.py) | `h2s2r/left_cup/camera_delivery.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/run_supervised.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/run_supervised.py) | `h2s2r/left_cup/run_supervised.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/test_runtime_support.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/test_runtime_support.py) | `h2s2r/left_cup/test_runtime_support.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analyze_d.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analyze_d.py) | `h2s2r/left_cup/analyze_d.py` | 实际D运行/真实CPU验证/离线分析入口 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/__init__.py) | `h2s2r/left_cup/rl_rebuild/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/__init__.py) | `h2s2r/left_cup/rl_rebuild/baselines/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/__init__.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/contract.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/contract.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/contract.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/controller.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/controller.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/controller.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/fabric_params.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/fabric_params.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/fabric_params.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/__init__.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/bimanual.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/bimanual.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/bimanual.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/inputs.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/inputs.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/inputs.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/observation.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/observation.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/observation.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/reset.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/reset.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/reset.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/trajectory.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/pour17/trajectory.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/pour17/trajectory.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/reward.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/reward.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/reward.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/synergy.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/synergy.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/synergy.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/tools/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/tools/__init__.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/tools/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/tools/generate_single_side_urdf.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/tools/generate_single_side_urdf.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/tools/generate_single_side_urdf.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/vega_sharpa_fabric.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/baselines/h2s2r/vega_sharpa_fabric.py) | `h2s2r/left_cup/rl_rebuild/baselines/h2s2r/vega_sharpa_fabric.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/__init__.py) | `h2s2r/left_cup/rl_rebuild/correction/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/__init__.py) | `h2s2r/left_cup/rl_rebuild/correction/env/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/correction_env_cfg.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/correction_env_cfg.py) | `h2s2r/left_cup/rl_rebuild/correction/env/correction_env_cfg.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/dexmate_env_cfg.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/env/dexmate_env_cfg.py) | `h2s2r/left_cup/rl_rebuild/correction/env/dexmate_env_cfg.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/frames.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/frames.py) | `h2s2r/left_cup/rl_rebuild/correction/frames.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/load_replay.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/load_replay.py) | `h2s2r/left_cup/rl_rebuild/correction/load_replay.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/paths.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/paths.py) | `h2s2r/left_cup/rl_rebuild/correction/paths.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/schema.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/rl_rebuild/correction/schema.py) | `h2s2r/left_cup/rl_rebuild/correction/schema.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/__init__.py) | `h2s2r/left_cup/tasks/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/__init__.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/__init__.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/__init__.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/artifacts/synergies/synergy_manifest.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/artifacts/synergies/synergy_manifest.json) | `h2s2r/left_cup/tasks/h2s2r_pour17/artifacts/synergies/synergy_manifest.json` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/cfg.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/cfg.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/cfg.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/env.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/env.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/env.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_env.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_env.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/left_env.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_workspace.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_workspace.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/left_workspace.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_workspace_bounds.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/left_workspace_bounds.json) | `h2s2r/left_cup/tasks/h2s2r_pour17/left_workspace_bounds.json` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/ours_physics.py](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/ours_physics.py) | `h2s2r/left_cup/tasks/h2s2r_pour17/ours_physics.py` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/ours_physics_spec.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/tasks/h2s2r_pour17/ours_physics_spec.json) | `h2s2r/left_cup/tasks/h2s2r_pour17/ours_physics_spec.json` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/third_party_licenses/H2S2R_LICENSE](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/third_party_licenses/H2S2R_LICENSE) | `h2s2r/left_cup/third_party_licenses/H2S2R_LICENSE` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/third_party_licenses/FABRICS_LICENSE](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/runtime_core/third_party_licenses/FABRICS_LICENSE) | `h2s2r/left_cup/third_party_licenses/FABRICS_LICENSE` | 实际服务器D最小源码依赖/配置/上游许可证，按SHA保留 | 新取回；未改功能 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/REVIEWER_HANDOFF.md](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/REVIEWER_HANDOFF.md) | `h2s2r/left_cup/REVIEWER_HANDOFF.md` | 当前状态、运行与资源交接、实验及reviewer结论 | 是 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/RESULTS.md](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/RESULTS.md) | `h2s2r/left_cup/RESULTS.md` | 当前状态、运行与资源交接、实验及reviewer结论 | 是 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/REVIEWER_REVIEW.md](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/REVIEWER_REVIEW.md) | `h2s2r/left_cup/REVIEWER_REVIEW.md` | 当前状态、运行与资源交接、实验及reviewer结论 | 是 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/validation.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/validation.json) | `h2s2r/left_cup/validation.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/decision.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/decision.json) | `h2s2r/left_cup/decision.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/costs.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/costs.json) | `h2s2r/left_cup/costs.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/execution_versions.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/execution_versions.json) | `h2s2r/left_cup/execution_versions.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/frozen_manifest.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/frozen_manifest.json) | `h2s2r/left_cup/frozen_manifest.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/raw_evidence_index.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/raw_evidence_index.json) | `h2s2r/left_cup/raw_evidence_index.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/delivery_validation.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/delivery_validation.json) | `h2s2r/left_cup/delivery_validation.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/A_SOURCE_INVENTORY.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/A_SOURCE_INVENTORY.json) | `h2s2r/left_cup/A_SOURCE_INVENTORY.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/A_D_DIFFERENCES.md](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/A_D_DIFFERENCES.md) | `h2s2r/left_cup/A_D_DIFFERENCES.md` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/cpu_behavior_validation.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/cpu_behavior_validation.json) | `h2s2r/left_cup/cpu_behavior_validation.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/actual_execution.diff](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/actual_execution.diff) | `h2s2r/left_cup/actual_execution.diff` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analysis/FIRST_12_SUBSTEPS.csv](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analysis/FIRST_12_SUBSTEPS.csv) | `h2s2r/left_cup/analysis/FIRST_12_SUBSTEPS.csv` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analysis/D_ANALYSIS.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/analysis/D_ANALYSIS.json) | `h2s2r/left_cup/analysis/D_ANALYSIS.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/reviewer_checks/independent_raw_check.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/reviewer_checks/independent_raw_check.json) | `h2s2r/left_cup/reviewer_checks/independent_raw_check.json` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/command_inputs.npz](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/command_inputs.npz) | `h2s2r/left_cup/command_inputs.npz` | 必要轻量证据/身份/固定诊断输入（非可部署policy） | 否 |
| [D:/UCBP/RL/Baselines/RL-Correction-H2S2R/LICENSE](D:/UCBP/RL/Baselines/RL-Correction-H2S2R/LICENSE) | `h2s2r/left_cup/LICENSE` | 保留源项目及第三方许可/来源说明 | 否 |
| [D:/UCBP/RL/Baselines/RL-Correction-H2S2R/NOTICE](D:/UCBP/RL/Baselines/RL-Correction-H2S2R/NOTICE) | `h2s2r/left_cup/NOTICE` | 保留源项目及第三方许可/来源说明 | 否 |
| [D:/UCBP/RL/Baselines/RL-Correction-H2S2R/THIRD_PARTY_NOTICES.md](D:/UCBP/RL/Baselines/RL-Correction-H2S2R/THIRD_PARTY_NOTICES.md) | `h2s2r/left_cup/THIRD_PARTY_NOTICES.md` | 保留源项目及第三方许可/来源说明 | 否 |
| [D:/UCBP/GPU_POLICY.md](D:/UCBP/GPU_POLICY.md) | `GPU_POLICY.md` | 全项目唯一GPU运营权威；共享上传目标需Team leader协调 | 否 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/SOURCE_MANIFEST.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/SOURCE_MANIFEST.json) | `h2s2r/left_cup/SOURCE_MANIFEST.json` | 本次源码对应、逐文件上传映射与交接自检 | 新建元数据 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/UPLOAD_MANIFEST.csv](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/UPLOAD_MANIFEST.csv) | `h2s2r/left_cup/UPLOAD_MANIFEST.csv` | 本次源码对应、逐文件上传映射与交接自检 | 新建元数据 |
| [D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/HANDOFF_SYNC_VALIDATION.json](D:/UCBP/reports/H2S2R_LEFT_CUP_A_ENV_TRANSFER_20260912/HANDOFF_SYNC_VALIDATION.json) | `h2s2r/left_cup/HANDOFF_SYNC_VALIDATION.json` | 本次源码对应、逐文件上传映射与交接自检 | 新建元数据 |
