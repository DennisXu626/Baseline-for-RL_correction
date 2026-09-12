# Pour 两种子从头训练：首轮协议（已提前结束）

最终状态（2026-09-08）：seed 0 完成 60,002,304 steps，并在固定 300 episodes 上取得 0.8469444444 平均入杯颗粒比例。用户确认该结果足以验证当前训练路线可行，决定停止 seed 1；seed 1 在 13,918,208 steps 主动终止，未产生最终模型或正式评测，不属于失败 seed。调度器同时停止。原先的两种子计划和执行规则保留在下文作为实验记录，不再是待执行队列。

用户先因总耗时将首轮从五个种子缩减为两个种子：seed 0 和 seed 1，每种子计划 6000 万步。seed 2–4 在启动前取消，该次调整未依据实验分数。此轮不应用 Place 评测补丁，不使用 SAPIEN，不迁移至 Sharpa/Pour17。

## 固定设置

- campaign：`pour_multiseed_60m_20260906_01`。
- 远端根目录：`/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco`。
- 源码：`pour_multiseed_source`，基于 `9790140170d8be49b828ab214ce3537a2475fcea`；独立于官方参考 checkout。
- 输入：官方 `ycb-025_mug-20200709-subject-01-20200709_150949.npz` 参考轨迹。该实验验证从参考轨迹到状态策略的训练流程，不等于从原视频完整重建。
- 训练 seeds：0、1。依次运行在 GPU0，保留两个结果，不因分数低重抽种子。总请求预算为 1.2 亿 environment transitions。
- `resume_model=null`：不加载作者 checkpoint。保留官方从参考轨迹设置 action bias 的 warm start，非预训练权重。
- 每种子请求 60,000,000 environment transitions；PPO 每 rollout 为 4096 transitions，SB3 完成整个 rollout 后的实际步数预计为 60,002,304，并单独记录实际数值。
- 官方设置：32 个训练环境、5 个在线验证环境；4096 global rollout、batch 256、5 epochs、learning rate 1e-5、gamma .95、GAE .95、entropy .001、clip .2；网络 `[256, 128]`。保留原 curriculum、reward、物理和随机化范围。
- 该 6000 万预算来自作者 Pour checkpoint 配套配置，但配置带有续训来源，不能据此声称已经知道作者总训练量。此轮为双方确认的统一从头训练预算。

## 种子与保存的必要接线

原 train.py 未将 cfg.seed 传给 PPO；原任务 RNG 自行从系统熵初始化；`_env_maker` 还会重置全局 NumPy RNG。仅换配置中的 seed 数字不能控制实验。独立副本做了以下改动，完整 diff 见 `verification/multiseed_repro.patch`：

1. cfg.seed 传入 PPO，控制网络、探索及其内部随机数。
2. GymWrapper.seed 接通实际用于轨迹随机化的 task RNG；SB3 将训练 worker 编号加在训练 seed 上。不同训练运行中部分 worker seed 数字可能重叠，整体 worker 流与策略随机数不同，不把它宣称为互不重叠的全部随机流。
3. 移除 `_env_maker` 对 NumPy 的无参重新播种。
4. 在线验证使用固定起始 seed 100000（其余验证 worker 顺延），所有训练 seed 相同；在线验证仍用于原版 curriculum。
5. learn 完成后保存 `final_model.zip` 和 `completed_steps.txt`；避免把更新前的周期性 snapshot 误作最终模型。

同种子重复测试的初始网络参数 hash 与三个 stage-2 reset 序列 hash 完全相同；seed 1 与 seed 0 均不同；同一运行的两个 worker 序列不同。见 `verification/multiseed_checks.json`。这是实测播种检查，不承诺跨驱动/硬件逐位确定性。

## 统一最终评测

- 只评测最终更新后的 `final_model.zip`，不从中途 checkpoint 中选最好分数。
- 300 episodes；reset seeds 固定为 200000–200299，区别于训练与在线验证。
- 采用官方 stage=2 和 deterministic=True。每次 reset 保存初始 physics hash；队列核对两个模型是否使用完全相同的初态，若不一致则停止并报错。
- 指标完全沿用原 Pour 评测：最后 12 个颗粒在固定容器边界内的比例，再对 300 episodes 求均值。保留逐 episode 颗粒计数与比例，不能称为二值 episode 成功率。
- 最终报告两个种子各自分数、均值、样本标准差（n=2）；与论文和作者 checkpoint 的结果区分。两个种子只能提供初步重复性证据，不能据此充分判断训练稳定性。没有预先约定的“接近”数值门槛，不事后据结果挑阈值或隐藏低分种子。
- 训练崩溃、缺少最终 checkpoint 或评测异常属于运行错误，队列暂停保留日志，避免对剩余种子重复浪费预算；正常但低分的模型仍进入汇总并继续下一种子。

## 已验证与启动记录

- 全规模 pilot：原 32 训练 / 5 验证环境，8192 transitions，成功保存最终模型。
- Pilot 的三次固定初态评测通过，颗粒比例为 0，仅作为短训练后的衔接检查，不计入两个正式种子。
- Pilot 训练速度约 600 transitions/s。线性估算单个 6000 万种子约 28 小时；正式速度随阶段、仿真和机器占用变化，该数字不是完成时限。
- 2026-09-06 13:13:06 UTC，队列缩减完成：旧调度器 PID 1179295 已退出，新调度器 PID 1182502 只处理 seeds 0、1。seed 0 原训练 PID 1179296 保持运行，核验时步数已继续增长至 577,536；没有重新开始。实际进展以远端 manifest / log 为准。
- 当前接管脚本：`reduce_campaign_to_two.py`；日志：`logs/two_seed_controller.log`。它等待当前 seed 0 正常完成并评测，随后从头训练 seed 1，最后汇总两个结果。
- 输出沿用原 campaign 名以保留进度：`outputs/pour_multiseed_60m_20260906_01/seed_0` 和 `seed_1`。seed 2–4 不会启动。
- 调整前 manifest 存档于 campaign 的 `manifest_before_two_seed_change.json`；当前 manifest 标明 `seeds: [0, 1]`、`cancelled_seeds: [2, 3, 4]`。
- 总状态：`outputs/pour_multiseed_60m_20260906_01/manifest.json`。
- 每种子日志：campaign 根目录的 `seed_N_train.log` / `seed_N_eval.log`。
- 每种子评测：`seed_N/fixed_eval.json`。汇总及 mean/std 在全部完成后写回 manifest。
- campaign 内存档源码补丁、依赖锁定清单与输入轨迹 SHA-256；W&B disabled。
