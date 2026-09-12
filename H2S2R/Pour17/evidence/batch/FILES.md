# 本机交付索引

- RESULTS.md：结论和限制；validation.json：分项验证；decision.json：停止及reviewer入口；costs.json：本轮及历史成本链；attempt_registry.json：最终2/2账本。
- sample/short02/audit/milestones/epoch_0000/env0_{front,top,side}.mp4：三路完整真实视频；同目录frames/为768张原始PNG、三路jsonl为逐帧元数据。
- sample/short02/audit/raw_chunks/：全部16环境控制原始轨迹；sample/short02/audit/env0_substeps.npz：3072子步。
- sample/short02/audit/reset_boundaries/：终止前图像与元数据；delivery/gates/：预览及第1/16步请求、签核和标注。
- sample/short02/gate.json、limit_dispatch.json、audit/runtime_counts.json：原生检查，保留原始PENDING视觉状态，不覆写。
- sample/short02_launch/：完整启动日志、实际import、观测阶段、调用栈、关闭结果、成本manifest。
- sample/short02_offline_qa/：全视频解码QA、覆盖全部256控制的32张审阅图、files.json逐文件SHA清单；local_integrity.json：本机逐文件和压缩包复核。
- attempt02_complete.tar.gz：完整真实样本913文件，SHA256 383e3de4b3fb105d4768ac4940d29c7459d36156472f73d41a43a242675da6c1。
- attempt01_evidence.tar.gz：首轮支持失败原证据，SHA256 cb3a8e2d00f2cf8dfb0630f03a51ffe2bae5ab9b65bbd022f714c659f851bc68；attempt01_failure.md说明恢复资格。
- source_before/、source_after/、support_diff.patch、support_manifest.json、expected_identity_before/after.json：执行差异和身份。
- cpu_camera01–04、cpu_legacy01–02、review02内目标CPU证据：明确为CPU替身，不是物理样本；support_checklist.md为三入口联检。
- preflight01.json、sample/preflight02.json：两次启动前GPU及存储快照；sample/recovery_deployment.json：最终部署结果。
- input_identity.json：实际加载三项顶层USD的运行后只读SHA；不是递归资产依赖清单。PCA/URDF与执行源码身份在expected_identity_after.json。

sample为原始包解压，不改其中的原证据；本目录顶层报告为执行者后验解释。旧physics_resume_01所有失败和成本继续保留。
