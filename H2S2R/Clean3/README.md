# H2S2R Clean3 / plate 当前入口

交接证据截止：2026-09-13 03:02:17 Asia/Shanghai（UTC+08:00）。上次维护同步基线：2026-09-13 01:01:27（独立交接文件C3-P2放行状态）；未发现此前Clean3 GitHub上传提交，不能把其他track的提交算作Clean3基线。

## 当前事实

- 实现：RTS面内参考、18cm几何、共享静态初态、双侧PCA5、固定动作校准、左掌z下界-0.03m及ZYX已在本地实现。双手342obs/22action（每侧6掌+5手），不是每手22关节直接控制。
- 已运行：仅旧R2两次Isaac启动，各约15秒，均在环境构造前结束；总0 controls/子步/帧，物理保持/穿模未验证，PPO及外评均0，成功率N/A。
- C3-P2增量：本地cfg已使用正确独立robot资产路径；增加preflight_checks.py/run_with_preflight.py并修改监督器failure-marker检查。CPU语法/监督器4例为已有交付记录；本轮不重跑实验。
- 未部署：远端仍为deployment_v3。cfg.py、launch_guard.py、launch_commands.json、deployment_manifest.json与当前本地不同；两个preflight文件远端不存在。不能把本地修改视为部署完成。
- 受阻：最新本地preflight_v5是FAIL，不是整体PASS；本地pxr及远端路径解析不可用。交接只读核验远端Python find_spec('pxr')也为None（不等于整个Isaac安装没有USD）。USD依赖/58活动关节/碰撞CPU前置尚未通过。SSH在03:02核验可达，此前终端Permission denied是历史环境限制，不能写服务器持续失联。
- 额度：R2 2/2关闭；C3-P2另批准最多2次，目前新增启动0。旧文档“继续attempt02剩余额度”作废。C3-P2 CPU总预算60分钟，但多个增量执行仅有0.5分钟局部成本，累计余额未核实，接手需核账；不默认剩59.5分钟。

## 阅读顺序与范围

1. 本README；2. Clean3独立交接（本地reports/H2S2R_CLEAN3_REVIEWER_HANDOFF_20260912.md）；3. C3-P2工作包；4. 最新C3P1R2_RESULTS.md及preflight_v5；5. deployment_c3p1r2/README.md。
最新方法/授权以C3-P2与R2/R修订链为准；本次仅同步，不新定路线、不加启动额度。22关节控制仅讨论过备选，未获本轮切换放行。

## 本地与服务器对应

本地工作根：D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_c3p1_20260912（不在独立Git工作树，commit未核实；用逐文件SHA）。当前源码在deployment_c3p1r2/。
远端旧overlay：/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/deployment_v3。
共享只读代码：/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912。
FABRICS：/home/kailang/experiments/baselines/fabrics_h2s2r_reference/src。
Python：/home/kailang/.local/miniconda3/envs/rl-correction-pour/bin/python；03:02元数据Python3.11.15、Torch2.7.0+cu128、NumPy1.26.0、SciPy1.15.3、IsaacLab0.54.2、IsaacSim5.1.0.0、Warp1.16.0。依赖锁未完整导出；无需为本次文档同步安装任何包。

03:02:17服务器进程快照未发现Clean3进程，仅是当时快照。GPU规则唯一权威D:/UCBP/GPU_POLICY.md，GPU0/1及满足准入的同卡共享允许，旧idle-only约束无效。

## 复现边界与资源

GitHub集合为当前Clean3运行overlay、支持测试、决策和轻量证据；不是含完整Isaac/共享V12/资产的自包含发行包。上传映射及每个源文件身份见同次GitHub文件清单和SOURCE_IDENTITY.json。目标子树H2S2R/Clean3，GPU_POLICY.md只在仓库根保留一份，AGENTS及其他track主文档交team leader汇总，不覆盖。

准备脚本prepare_c3p1.py、prepare_c3p1r.py、prepare_c3p1r2.py和原CPU测试保留在本地工作根；历史工作区依赖staging解包数据和reports/.../source_after目录，不能搬到任意目录直接运行。归档源与其索引在完整本地目录（见下方），已生成运行输入/PCA/校准可从远端v3/data取回，必须核对该部署manifest的SHA。当前GitHub运行入口使用这些已有输入，不要求重新生成。若需要原始数据重建，先恢复原工作区结构和访问权限。

- 原视频/重建和准备来源：D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_staging_20260912；原包clean3_base_package_20260909(1).tar.gz在其父目录。unpack_manifest.json逐文件索引；prepared/provenance.json记录字段来源。数据来源是设备ARKit/相机+重建对象+尺度先验，非纯估计；不上传视频/大资产。
- 运行输入、PCA、校准与plate/sponge USD/纹理：远端deployment_v3/data和assets；SHA见deployment_v3/deployment_manifest.json。v3原包在同级incoming/c3p1r2_deployment_upload_v3.tar.gz，SHA cc424a2c22e55393022aa9586e5de21e50bea0ba64f16bf4a788e1b268f1cf8c；它不含本地C3-P2支持修复。
- 正确机器人：/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17/world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd；SHA c2bba25aeb7b04b270df51d0c0efbe8eec95b499d8a7aef46834cefcfb71e347。只读复用，不加载Pour17策略或任务。
- 两次失败证据：远端Clean3根physics_gate_01和physics_gate_02；本机reports/H2S2R_CLEAN3_C3P1R_20260912/c3p1r2_evidence，index.json有SHA。attempt01旧state=completed是记录bug，ready为空且0controls，不能解释为成功。不存在可用checkpoint/物理视频/轨迹。
- 共享V12、FABRICS、Human2Sim2Robot及Isaac必须由获准访问服务器/原工作区的同事配置；SSH私钥/token不随仓库上传。内部上传到已确认服务器已有持续授权，不自动扩展成新第三方传输或资产重新授权许可。第三方notice及许可证随代码保留。

接手第一步是核对本地/远端差异及C3-P2累计CPU成本，解决CPU USD可见性并重过完整前置，再按已批准包继续；本次未解决这些技术问题。不得复制旧launch_commands.json就重用physics_gate_02输出目录，模板未验证通过。

---

以下保留历史C3-P1/R说明；“PCA重建>0.25rad即停止”和旧掌范围阻塞均已由R/R2替代，不是当前执行指令。

# Clean3 C3-P1 isolated adapter

This directory implements only the reviewer-frozen CPU-verifiable portion of
`H2S2R-adapted Clean3 / RTS planar reference / shared static initialization`.
It never imports ours policies, checkpoints, dynamic GraspPose commands,
reference rows after row zero, reward, certification, pin/release, or squeeze.

Run:

```powershell
python prepare_c3p1.py
python tests/test_c3p1.py
```

`prepared/manifest.json` is the machine-readable gate result. A PCA initial
target jump over 0.25 rad is a mandatory stop before Isaac, physics, or PPO.

## C3-P1R

The reviewer correction is preserved separately and does not rewrite the old
gate result:

```powershell
python prepare_c3p1r.py
C:\Users\Dennis\miniconda3\envs\UCBP\python.exe tests\test_c3p1r.py
```

`c3p1r_prepared/manifest.json` records the fixed 5% action-range sidecar and the
subsequent unchanged-palm-bound gate.

## GitHub阅读与离线边界

当前入口：[交接](HANDOFF.md)、[结果](RESULTS.md)、[C3-P2](decisions/C3P2.md)、[部署说明](deployment_c3p1r2/README.md)。方法补充在decisions/C3P1.md、C3P1R.md、C3P1R2.md；后者覆盖前者冲突条款。文件与远端对应见SOURCE_IDENTITY.json，证据在evidence/。GPU政策只用仓库根../../GPU_POLICY.md。

preparation/是现有准备器原字节归档，不是可直接执行的便携CLI。须按SOURCE_IDENTITY.source恢复原D:/UCBP布局，并恢复staging数据和source_after共享快照；其clean3_adapter/c3p1r/c3p1r2模块使用本overlay中的对应实现。原工作区已有输入生成记录，GitHub部署可取远端v3/data输入，不要求重生成。builder会生成机器专用旧输出路径，仅保留复现来源，不执行它覆盖当前证据。

本集合保持外部V12依赖，不复制或覆盖H2S2R/Pour17的实现。共享机器人、源视频、重建网格不重新许可；取回需项目权限。官方Human2Sim2Robot/PPO与FABRICS版本和许可证见THIRD_PARTY_NOTICES.md。脚本中的机器路径是环境定位，不含认证信息；SSH配置由接手人自备，仓库不存私钥。
