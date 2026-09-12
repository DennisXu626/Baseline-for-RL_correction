# V12 batch_delivery_01 — 已交付，等待 reviewer

本包完成共用录像/记录支持及正常随机策略短验证。SUPPORT、RUNTIME、MEDIA 通过；PHYSICS 为执行者有限样本筛查通过、待 reviewer 最终裁定。没有启动 PPO 或终评，不自动进入下一包。

## 实际运行与证据

- attempt01：真实三视角预览清晰，但签核文件 SCP 半写入引发 JSONDecodeError，0 controls，208.857834 秒。保留原 FAIL、日志与预览，不能记为物理失败或成功样本。
- 修复已定位的纯支持发布竞态：临时文件上传、JSON 校验、原子提交 ready 文件；补充反例测试，重新核验身份后使用最后一次恢复。
- attempt02：GPU1，16 环境、seed42、随机初始化官方 LSTM、256 controls；初始化完成，进程 695.910035 秒，无超时。256 次设目标、3072 次积分与提交；4096 环境 transitions、49152 环境子步。环境0逐子步3072条，全部16环境控制轨迹已交付。
- 原42项 CPU 与新增43项 CPU 通过；最终32项部署身份通过。限位读回、关节顺序、左固定命令隔离、hold0、finite、reset、无活动期物体写入通过。
- 真实预览、第1和第16 control 均在同一进程中完成三视角目视和几何签核。三路960×720、20fps、各256帧，12.8秒仿真时间，无黑帧；不是预览拼接或离线重渲染。包 SHA 和逐文件完整性见 local_integrity.json。

## 物理判断及边界

按三视角逐帧接触图审阅全部256 controls，并放大第4与212 control 接触帧。未见明显持续至少3 control的深穿，未发现需改变物理或控制的实质矛盾；瓶子倾斜、贴近杯子以及随机策略不抓起不判无效。接触力与画面联合使用，不以接触力非零证明无穿模。env0 ≥0.1N 的控制采样为第4、212 control，峰值1.038627N；全环境峰值6.239975N。reset98/208明确标记，终止前边界另存，不能拼接成抓举。

本结论只覆盖本次可视环境与有限采样，不是所有16环境每个接触面的定量穿透深度认证；没有新增碰撞深度接口。训练/终评实际运行、长时编码稳定性及策略自主抓举仍未测试。reviewer仍须独立看真实视频，决定是否放行独立训练包。

## 实际差异

仅修改计划批准的6个支持文件并新增 camera_delivery.py。统一指定环境 origin 一次平移、瓶掌中心近景、既有纯渲染刷新、逐帧元数据、PNG日志与增量MP4、reset边界、预览阶段门、关闭前保存及硬截止。短/训练/终评接入同一共用模块；训练和终评仅CPU/静态联检，未真实启动。恢复仅改签核发布协议。

实际源码前后副本见 source_before/source_after，逐行差异 support_diff.patch、SHA support_manifest.json 和 expected_identity_after.json。核心、zero-margin、FABRICS、PCA、奖励、动作、初态和物理未改；黑桌 env.py SHA 8d5cb546d523fd5902f8e6392bdf9763f20eb7a847004bdd8537a35b0ccf9273 保留。实际 import 路径在 sample/short02_launch/observation/actual_imports.json；输入和执行身份见 expected_identity_after.json、sample/preflight02.json 及 launch_manifest.json。

## 成本、状态与交付

两次启动累计904.767869秒（约15.08分钟），额度2/2已用完。详细墙钟、CPU、内存、传输、历史成本链见 costs.json；未知费用标null，不编造零费用。旧campaign及attempt01失败不覆盖。

训练量0、训练时长0、终评有效分母0、成功数与成功率未定义，不是0%方法成功率；RIGHT Gate B未建立。停止原因是本包交付完成、下一包由reviewer裁定，不是抓举失败。

文件入口见 FILES.md；完整本机视频在 sample/short02/audit/milestones/epoch_0000/env0_front.mp4、env0_top.mp4、env0_side.mp4。完整成功运行包 attempt02_complete.tar.gz，原失败包 attempt01_evidence.tar.gz。
