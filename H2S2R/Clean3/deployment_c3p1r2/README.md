# 当前部署状态：C3-P2本地支持修复，未部署/未通过USD预检

截止2026-09-13 03:02:17 Asia/Shanghai。上级README是当前入口。本目录为当前本地源码，不等于远端deployment_v3；SOURCE_IDENTITY.json逐文件标识差异。deployment_manifest.json是既有构建器产物，不是本次上传清单。

已运行的命令仅历史physics_gate_01/02/process_status.json中command数组，均在环境构造前失败。launch_commands.json是未验证模板，含旧physics_gate_02输出路径和GPU0占位，不能直接执行；选择GPU依据仓库根GPU_POLICY.md，新包输出和额度依据C3-P2。train_lstm.py/evaluate.py从未运行。

CPU命令已有实施者记录：原工作根下C:/Users/Dennis/miniconda3/envs/UCBP/python.exe tests/test_c3p1r2_launch_guard.py，4/4 PASS；源码检查不等于真实装配验证。本次未重跑。当前预检命令形式（未在完整服务器资产环境通过）：

```text
<PYTHON> preflight_checks.py --runtime_root <CLEAN3_OVERLAY> --v12_root <V12_CODE_ROOT> --output <NEW_OUTPUT>/preflight.json
```

PYTHONPATH依次需包含overlay、冻结V12根、FABRICS/src，环境须有human2sim2robot PPO。pxr须能在SimulationApp前解析；当前直接find_spec不可见，本次未安装/修复。共享实现保留外部V12根，不能用Pour17最新V13代替。GitHub子树不含大型data/assets，先按原manifest恢复远端v3对应数据；它们不是缺失源码。

以下保留R2历史配置摘要，不是启动授权。

# 当前部署状态：C3-P2本地支持修复，未部署/未通过USD预检

截止2026-09-13 03:02:17 Asia/Shanghai。上级README是当前入口。本目录为准备上传的本地源码，不能与远端deployment_v3等同。SOURCE_IDENTITY.json逐文件标识差异；本目录deployment_manifest.json是既有构建器产物，不能代替本次上传清单或保证代码已部署。

已验证运行过的命令：仅历史physics_gate_01/02的process_status.json内command数组；两次均在环境构造前失败，不是可成功复现实验命令。
launch_commands.json是未验证的配置模板，包含旧physics_gate_02输出路径和物理GPU0占位，不能直接执行；GPU选择须按仓库根GPU_POLICY.md，新包输出命名/额度须按C3-P2，不覆盖历史证据。train_lstm.py/evaluate.py尚未运行，不能宣称训练入口可用性已验证。

CPU支持测试已有记录：在原工作根使用C:/Users/Dennis/miniconda3/envs/UCBP/python.exe tests/test_c3p1r2_launch_guard.py，4/4 PASS（实施者记录）。源码测试test_c3p1r2_deployment.py不等于真实环境装配PASS。当前预检命令形式（未在完整服务器资产环境通过）：

```text
<PYTHON> preflight_checks.py --runtime_root <CLEAN3_OVERLAY> --v12_root <V12_CODE_ROOT> --output <NEW_OUTPUT>/preflight.json
```

必须使用能在SimulationApp启动前解析pxr/Usd依赖的CPU环境；本次仅find_spec检查，未安装或修复该依赖。PYTHONPATH需按顺序包含本overlay、冻结V12代码根、FABRICS/src，并具备human2sim2robot PPO包。它们是外部前置，GitHub本子树不复制其他track的共享实现。只读V12版本由SOURCE_IDENTITY/原部署依赖记录固定，不能随意用Pour17最新V13替换。

---

以下为历史R2配置摘要（不是启动授权）：

# Clean3 C3-P1R2 deployment overlay

Independent target: `/media/msc-auto/HDD/users/kailang/h2s2r_clean3_c3p1r2_20260912/deployment_v3`.
The overlay is prepended to the read-only V12 runtime and FABRICS source paths;
it does not modify Pour17 or shared upstream files.

Frozen runtime:

- 342 observations, 22 actions; both hands learn.
- 20 Hz control, 240 Hz physics, 30 Hz reference; 12 advances per side/control.
- hold/clamp is zero; plate is object 0/left and sponge is object 1/right.
- C3-P1R hand calibration bytes are unchanged. Only left palm z minimum is
  `-0.03 m`; nominal palm centers use FABRICS `Rz*Ry*Rx` (`ZYX`) angles.
- Physics gate is one 4-env/400-control launch. Training is from scratch at
  256 envs, seed 42, rollout 16, minibatch 4096, up to 512 epochs.

Launch commands are recorded in `launch_commands.json`. Each command requires a
fresh resource/identity snapshot and uses `timeout` at the plan boundary.
