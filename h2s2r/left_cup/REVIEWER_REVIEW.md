> 当前团队入口：[REVIEWER_HANDOFF.md](REVIEWER_HANDOFF.md)。2026-09-13同步不改变本次审阅的物理结论；无下一物理包放行。

# 左杯D运行统一审阅：运动已精确复现，未见已记录contact负间距

2026-09-13，左杯reviewer。只读审阅并做本机离线复算；未启动App、物理或训练。GPU运营权威已读取D:/UCBP/GPU_POLICY.md；旧计划独占规则不再生效，但不增加额度或恢复已关闭包。

## 接受的数据结果

读取RESULTS、validation、真实replay/contact保存代码、A_D_DIFFERENCES、completion及原始substeps.npz/contacts.npz。独立复算保存在reviewer_checks/independent_raw_check.json。

- 1200子步、261636条contact；separation未经过abs，原始有效值最小+0.000056605786m、最大+0.00599999819m、负值0。14376条force>0。
- 首子步唯一非零力记录为left_pinky_MP，7.056977N。已记录运动中存在非指尖承力，不能用五tips为0解释为漂浮/无接触。
- **本reviewer新增核验：D每个control末全部29维左手实测q，与原warmup0 trace1..100逐值完全相同；100个杯xyz也逐值完全相同。** 这是强运动复现证据，不只是相似视频。腕/杯旋转及qd未在该新增检查中逐位比较，不扩大口径。
- 查看实际control48 top/side提取帧，记录在reviewer_checks/。这些为真实D视频的帧，不是离线网格重建。不要把该hold0路线与v6 hold15大范围掌部交叉混为同一运行。
- 原raw/media/completion均在app.close卡住前完成并保存。接受D诊断轨迹和媒体有效；进程退出依然TIMEOUT/FAIL，不能改成正常退出，也不因关闭失败丢弃已完整数据或重跑。

## 因果与范围

对于这条warmup0运动，后续冻结/reset回写不是必要原因：新执行没有active state write且精确重现。FABRICS运行时积分/策略采样也不是重现该运动的必要条件，新运行仅回放历史目标。仍不能由此证明当初产生这些目标的策略合理或FABRICS不存在其他问题。

真实contact正separation支持接触壳层内的非指尖受力，不证明整个物体pair的所有表面无交叉。contact separation属于求解器所报告的接触点；没有actual cooked完整几何，也要考虑碰撞/求解阶段与post-step测量时序，不能把它当任意网格点的同时刻signed distance。输入网格/可见网格相交不因此被否认，也不能反推PhysX穿透已证实。

当前证据显著削弱“杯未接触手却被抬起”“必定是后续冻结回写”“该小指碰撞体完全漏载”等解释。更具体的剩余工程问题是特定初始化与碰撞表示是否允许不符合期望的视觉交叉/抓法；不是已有证据证明solver穿透导致失败。

## 下一步工程裁定

1. 不继续盲调solver/PD/offset，不为让抓法更好看而改FABRICS或奖励。
2. 不因A身份不齐无限寻找A。当前没有可证明错误的D模块及可验证A替代物；整套移植不能作为bug修复。
3. 若用户仍要求消除这条hold0路线里的特定可见交叉，先用本次已取得的全手pose/contact对照该画面及前后子步，属于离线结果判读，不再申请GPU来“确认有无非指尖接触”。若必须改变初态或碰撞表示，明确提交方法/资产适配决策，不能伪称已经定位到代码错误。
4. v6 hold15历史异常不由本D自动认证修好；当前其他任务/V12也未被此包认证LEFT_PASS。不同轨迹/版本分开记载，不让所有历史图像成为一项无限验收门。
5. app.close仅是后续支持修复项，不应作为再次采集这条已完整轨迹的理由。本包维持关闭，不消费数字上剩余的App/轨迹额度。

最终审阅状态：D_REPLAY_DATA_ACCEPTED；HISTORICAL_HOLD0_MOTION_REPRODUCED_EXACT_Q_AND_CUP_XYZ；NO_NEGATIVE_REPORTED_CONTACT_SEPARATION；VISUAL_GEOMETRY_CONCERN_NOT_CLOSED；NO_JUSTIFIED_A_TRANSFER。不标baseline已修/LEFT_PASS/学习成功率。
