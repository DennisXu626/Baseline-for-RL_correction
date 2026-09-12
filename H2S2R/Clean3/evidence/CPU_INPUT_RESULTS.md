# Clean3 CPU接入包 — 已完成，等待统一审阅

**CPU_DATA_PREP_PASS / TRAINING_BLOCKED_PENDING_REVIEWER。** 已交付能实际运行的数据转换入口，不是仅调查报告。未启动GPU、Isaac、训练或H2S2R评测，未修改Pour17部署。

## 完成项

1. 两个用户原包在独立staging安全解包，保留原包，逐文件SHA落盘；无链接/设备/越界路径。base177,790,191字节、results23,097,430字节。原代码仅作为静态来源阅读，未执行ours环境/策略/训练。
2. 原视频`inputs/source_video/3.mp4`完整解码300帧，1920×1080，30fps，10秒；`3.hdf5`相机/内参/ARKit300帧可读。明确左手盘子object_0、右手海绵object_1，源视频第250帧已目视核对。
3. `prepared/clean3_perception_candidate.npz`已生成并被**当前实际V12原版输入加载器**成功读取；该文件具备全部14个要求字段。手数组300×22、四组位姿300×7、camera300×4×4、bool masks与frame_ids有效。参考构建器30Hz、原首帧identity-anchor CPU测试通过；没有选仿真reset。
4. 原replay/ref_qpos标15Hz，但frame_ids0..299、原视频/HDF5/world相机逐行匹配。转换输出记录实际30Hz（首末采样跨度9.9667秒），保留原15Hz元数据并明确差异；没有改变任何物轨/手轨样本、平滑、填补、裁帧或时间展宽。HDF5→重建world相机矩阵最大绝对误差2.98e-7。
5. 左右22关节名/顺序与当前H2S2R contract完全一致。`finger_qpos`只来自demonstration-derived DexPilot输入，不来自训练结果、IK母带或GraspPose。反例及实际数据共22项CPU测试通过。
6. 当前OBJ、纹理及USD位置与SHA已整理；OBJ材质纹理依赖齐全。盘当前网格约18cm，海绵约7.6×3.8×13.2cm。USD组合、碰撞几何和仿真有效性未测试，不能由文件存在推断通过。

## 没有偷偷迁移的内容

ours checkpoint、训练后命令、58维残差动作、GraspPose/grasp/squeeze、双臂IK、pin/release、重新贴合的clean3_reference_v*、奖励/soft_rel/reanchor/时钟门均未进入候选输入或H2S2R代码。它们只留在原始解包证据中。source `scene_layout.json` 的持物初态也未转换成可运行reset。

## 尚未关闭

- 需要明确Clean3持物初始化、左/右政策装配、PCA来源身份、奖励/外部评测接口。精确函数、字段和最小后续动作见INTERFACE_GAPS.md。
- 输入并非纯模型估计：相机/ARKit来自数据集，物体为模型估计，盘尺度带类别先验。现有loader会笼统标MODEL_ESTIMATE，本包用侧车/NPZ元数据明确标混合来源，不改loader、不称Oracle；schema PASS不等于训练来源获批。
- 两腕几乎静止；上游盘旋转可信度差。available300/300不等于trustworthy300/300；现数组trustworthy为187/126，旧summary为192/153，保留差异不删不改。
- replay旧盘采样点约24cm、当前网格18cm，已分开记录；旧采样点不能用作当前碰撞几何。没有自动缩放/修姿态。
- 原A4“接触”使用signed gap<5mm，不能证明无深穿；历史success还包括ours认证/终止语义。这里只提取定义，未将其奖励注入baseline。

历史ours三组原始评测按16条episode重新求和为1/16、9/16、16/16；仅用于已有证据核对，**不是本次H2S2R结果**。本任务H2S2R训练量0、评测分母0，成功数/成功率未定义。

## 可直接运行

```powershell
& C:/Users/Dennis/miniconda3/envs/UCBP/python.exe D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_staging_20260912/prepare_clean3.py
& C:/Users/Dennis/miniconda3/envs/UCBP/python.exe D:/UCBP/RL/Baselines/task_plate/h2s2r_clean3_staging_20260912/test_prepare_clean3.py
```

无需安装依赖，使用本机已有UCBP Python。首个CPU准备尝试因Python3.10不支持hashlib.file_digest失败，已改为标准分块SHA并通过；原失败证据保留为preparation_failure.json，仅属已修正的本包支持错误。

交付索引：staging/README.md、unpack_manifest.json、prepared/provenance.json；本目录validation.json、cpu_tests.json、assets.json、source_quality.json、hdf5_schema.json、evaluation_definition.json、historical_results_audit.json、INTERFACE_GAPS.md、decision.json、costs.json、delivery_manifest.json。仅在本包独立目录写入；停止并交reviewer，不自行进入训练。
