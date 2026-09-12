# 交接同步核对（2026-09-13 03:02:17 Asia/Shanghai）

当前：R2关闭且2/2启动、0物理样本；C3-P2本地支持代码已继续修改，未部署、未新增启动，USD预检仍FAIL。保持/穿模/PPO/外评未验证，成功率N/A。

更正下方历史增量中的“继续attempt02剩余额度”：C3-P2另批准最多2次，目前0次；R2仍2/2关闭，不新增授权。runtime_decision时间字段2026-09-13T00:00:00+00:00晚于实际核查时刻，未核实，不作为事件时间；0.5分钟runtime_cost仅覆盖局部probe，不能证明全包CPU成本。

本次SSH可达且无Clean3进程，远端Python直接find_spec(pxr)=None，不等于Isaac整体没有USD。本地cfg已修正资产路径，远端仍旧版；两个新preflight文件远端不存在，guard/commands/manifest亦不同。见GITHUB_REMOTE_SNAPSHOT_20260913.json及上传SOURCE_IDENTITY。历史Permission denied不能写成服务器持续失联。

以下保留历史交付及实施者增量；本节核实结论优先，原日志/机器读数不改写。

# Clean3 C3-P1R2 implementation result

## Resume final（2026-09-13）

最终状态：`CLOSED_ATTEMPT02_MISSING_ROBOT_USD`。上传阻塞由 reviewer 解除，v3 在新 `deployment_v3` 目录解包并逐文件 30/30 PASS，冻结 controller/FABRICS/双 URDF 身份 5/5 PASS。按 GPU_POLICY 采样 10 秒后选择物理 GPU1：空闲 48,063 MiB、利用率 2–3%、主机可用内存最低约 253.75 GB；`RL_ISAAC_NO_GUARD=1`。

最后一次 attempt02 启动 15.002870 秒，修复后的入口正确保存了 `FileNotFoundError`：Clean3 cfg 要求的
`/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter_v12_training_pilot_20260912/world/vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`
不存在。进一步只读核验发现 V12 自身声明的 `rl_rebuild/assets/vega_1p_sharpa_fixedtorso.usd` 和回退 `/home/lyh/luhr/MagicSim/Assets/Robots/vega_1p_sharpa.usd` 也不存在。环境构造未开始，0 controls、0 substeps、0 frames、无轨迹。

attempt02 已用完 2/2 启动额度。依停止表没有修路径重启、没有 attempt03、PPO 或外评。物理有效性、零动作稳定、双手真实推进、持物和穿模均为 NOT_TESTED；有效评测分母 0，成功率 N/A。新证据见 `c3p1r2_evidence/physics_gate_02/`。

以下保留上传解除前的历史断点：当时状态为 `DEPLOYMENT_APPROVAL_BLOCKED`。CPU 实现与数值签核通过；真实物理有效性、PPO 和外评均未完成，成功率为 N/A。

## 已完成

- 左掌动作 z 下界只在 Clean3 侧车冻结为 -0.03 m；其余掌边界不变。固定名义 FK 零点，不在线扩界。
- 角度编码改为 SciPy 大写 `ZYX`，对应 `Rz(z) @ Ry(y) @ Rx(x)`。双掌 FK、旋转矩阵和七掌点往返误差均远低于 1e-6。
- 实际 `H2S2RFabricController.set_targets` 零动作误差：右位置 1.42e-8 m、旋转矩阵 3.02e-8；左位置 1.06e-8 m、旋转矩阵 7.5e-8。端点、有界性、reset 固定中心、arm_center/world 变换及双侧每控制步 12 次推进均通过 CPU 契约测试。
- 保持 R 版手 PCA/范围侧车字节不变，SHA256 `f878f87f61f218175292583b266f6150142c95dc3366a328c516383e58507747`。
- Clean3 独立真实 cfg/env、官方 LSTM/PPO 入口、外评和录像/原始轨迹入口已接线并通过源码/语法检查。
- 修复三入口异常证据在应用关闭前保存，并强化监督器：没有终态成功标志时，即使子进程返回 0 也判 incomplete。CPU 替身的正常、异常、直接退出三条路径 3/3 PASS。

## 唯一真实启动 attempt01

GPU0 启动前 47,378 MiB 空闲、利用率 1%；进程墙钟 15.003 秒。Isaac App 完成启动后，日志停在 `build_cfg()` 期间；未出现环境构造 begin/complete，物理控制 0、子步 0、帧 0、轨迹 0。旧监督器将返回码 0 误写为 completed；这是已定位并修正的记录支持故障，不是 Clean3 物理 PASS/FAIL，也不是方法成功率 0%。本机证据在 `c3p1r2_evidence/physics_gate_01/`。

## 停止原因

修复后的 v3 部署包大小 26,448,738 字节，SHA256 `cc424a2c22e55393022aa9586e5de21e50bea0ba64f16bf4a788e1b268f1cf8c`。向明确的独立 HDD 目录上传时，审批在进程创建前拒绝，传输 0 字节；完整原文见 `deployment_approval_rejection_c3p1r2_v3.json`。按 reviewer 的明确停止分支，没有绕过、没有消耗 attempt02，也没有启动 PPO/终评。

因此当前结论只能是：入口的本地实现和数值契约已通过；是否穿模、能否动态持盘/海绵、能否自主擦盘都尚未测试。有效评测分母 0，成功数和成功率均 N/A。

## 本次 C3-P2 实施更新（2026-09-13）

- 已在本地完成剩余支持修复与预检联动：修正了预检命令包装、更新了 `launch_guard` 的失败分类、补齐 `launch_commands.json` 的预检输出目录，并回归了本地 launch/support 语法与异常证据测试（3/3 PASS）。
- 依据 reviewer 停止边界，`physical_starts` 在本条权威记录中仍为 2/2；在本次连续执行中未发起新的物理启动（无新增 4-env 迹象、无视频/轨迹输出），不会与既有预算冲突。
- 远端部署身份与远端解析服务当前不可达（用户侧 SSH 22 端口访问在此终端返回 Permission denied），因此未新增 attempt 启动与 GPU 运行证据；若 reviewer 批准并恢复可达性，可在保留上述支持修复后继续按 attempt02 剩余额度继续。

## C3-P2 第二轮补充（2026-09-13，执行包）

- 按 C3-P2 继续推进本地支持链路：未改核心算法、核心/共享代码或 PPO/Eval 流程，仅修正预检与监控器。
- `deployment_c3p1r2/preflight_checks.py` 收紧：
  - 解析 `tasks/h2s2r_clean3/cfg.py` 中 `build_cfg` 的真实资源表达式（inputs/sidecar/pca/usd/urdf）并输出核验条目；
  - 继续核对常量资产路径与 SHA；
  - USD 阶段新增 joint/固定 torso 候选扫描与网格碰撞覆盖统计（缺失时返回 FAIL）。
- `tests/test_c3p1r2_launch_guard.py` 新增 failure-marker 场景回归，验证出现 `[CLEAN3-ENTRY-FAILURE]` 时无论子进程是否 exit 0 都判 failed。
- 本地回归结果：`py_compile` PASS；`test_c3p1r2_launch_guard.py` 4/4 PASS。
- 本轮 CPU 预检产物：
  - `c3p1r2_preflight_c3p2.json`
  - `c3p1r2_preflight_c3p2_v2.json`
- 本轮停止条件：`pxr` Python 依赖缺失导致 USD 58 关节/碰撞链路无法在本机完成；远端 robot/URDF 仍不可直接本机读取；未开启新 attempt、未启动 PPO/外评。

## 本次 C3-P2 最新执行（本次会话）

- 按 C3-P2 裁定继续在本地补齐启动前支持检查与 attempt02 监控 parser：  
  - `preflight_checks.py` 继续修正了 `cfg.py` 资源表达式解析路径和 `ROBOT_ASSET_PATH` 解析逻辑，新增 `+` 号连接解析和远端路径归一化保护；  
  - `launch_guard.py` 支持在出现 `[CLEAN3-ENTRY-FAILURE]` 时即使子进程返回码为 0 也判定为 failed；  
  - `parser_probe` 运行结果见 `c3p1r2_evidence/parser_probe_status.json`（`state: failed`, `ready_marker_seen: true`, `failure_marker_seen: true`），满足 attempt 监控失败场景验证。
- 本次 CPU 独立前置输出新增：
  - `c3p1r2_preflight_c3p2_v4.json`（历史）
  - `c3p1r2_preflight_c3p2_v5.json`（本次修复后）
- 本次 `c3p1r2_preflight_c3p2_v5.json` 关键结果：
  - `cfg.py` 关键资源表达式可见且已解析（包括 `root / ...` 与 `v12 / ...`）；
  - 本地 `pxr` 不可用：`pxr import failed: ModuleNotFoundError("No module named 'pxr')`；
  - 本机缺失共享资产/URDF（预期 remote-only）：`/home/.../vega_1p_sharpa_fixedtorso__RUNTIME_LOADED.usd`、`v12_root` 下 `vega_sharpa_*_fabric.urdf`；
  - 未启动 attempt02 物理、不产出新视频/轨迹。
- 资源端当前执行环境为笔记本级 GPU（`NVIDIA GeForce RTX 5080 Laptop GPU`），不符合该包 A6000 共享 GPU 运行前提；本地不进行远端 GPU/Isaac 复跑。
- 小结：本轮任务收敛到“C3-P2 起跑前门禁已补齐，但因 CPU 运行时缺依赖 + 本地无远端部署资产可见性，无法进入 attempt02 物理门控”。历史 decision/costs 与已耗尽 attempt 约束保留不变。
