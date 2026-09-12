# ViViDex baseline: durable project state

## Current handoff snapshot — 2026-09-13 01:38:28 Asia/Shanghai

**文档更新时仍有ViViDex训练正在进行。接手同事或AI agent读到同步文档后，请先向Dennis索取训练结束后的最终报告、checkpoint/state、预算账本和SHA256；不得把本快照当最终结果，不得重复启动当前作业。** 本次只读核对及交接更新没有启动、停止或修改任何实验。

上次文档同步基线为2026-09-13已封存DEV_EXTEND_262144_V1结果及随后“百万步包待批准”的reviewer记录；执行方随后已将H改为获批/运行中。当前代码、配置、远端作业账本和用户本次说明相互支持该变化。截止时点为UTC 2026-09-12 17:38:28（北京时间2026-09-13 01:38:28；PDT 2026-09-12 10:38:28），不是训练结束时间。

- 已实现/部署：训练与内部评测显式使用同一WP4 root overlay；安全恢复新模拟器观测；单独扩展预算与固定策略诊断；本次新增百万步调度入口、524,288中点诊断及GPU_POLICY运营接线。本地71个当前部署代码/配置文件与远端SHA逐项一致。工作目录/依赖/命令与上传清单见README；没有可用的适配仓库commit，采用源码hash绑定。
- **已运行但未完成：** DEV_TO_FIRST_CURRICULUM_V1，seed1701，32env/n_steps128，从262,144恢复，目标1,003,520，最多新增741,376或4h。快照progress=577,536、stage0、training_wall_s=4001.974秒；调度PID2527603、训练PID2527650，物理GPU0，UUID GPU-1d992e9e-0787-f30e-3b2b-b5ea38a7897a，内部cuda:0。524,288两组25回合固定诊断已执行；本交接没有对其数值作最终reviewer验收。完整精确进度见verification/handoff_20260913_snapshot.json。本段PID/运行状态仅适用于该时点。
- 最近已封存、已评审模型仍是262,144步checkpoint（SHA92868d65…b19483）；确定性右/左边界中位96.85/163.68mm，随机195.76/218.96mm；两模式联合预抓取均0/25，训练累计0/7,343。不能把native MuJoCo结果、固定命令回放、训练回合统计或中点诊断冒充正式Pour17任务SR。
- 未解决：源MANO+W1中有桌内手点；探索时臂桌接触；双手门槛尚未通过；操作阶段学习未证明；共同正式runtime/物理合同未冻结。正式3M及最终512未放行，桌约束IK未实施，视觉student明确不在本实验范围，其他三个task暂缓。M0 BLOCKED、M1-A NOT PASSED为原验收状态，不能用它们抹去已运行的开发训练事实。
- 当前下一步仅为完成已批准包、封存并review；之后的扩训/方法/正式预算待决策。本次交接没有新定路线。GPU0/1共享均按../../../GPU_POLICY.md执行，旧“空闲/独占/固定GPU”描述仅为历史。

下方完整保留技术和历史证据。与本节冲突的“未启动”“无遗留进程”“待批准”等日期更早描述仅针对当时交付；可执行授权只看本节和H当前入口。GitHub不包含所有历史大产物；归档路径与读取方式由README的archive-access索引解释。

Maintained 2026-09-13 after DEV_EXTEND_262144_V1. Scope: this baseline only. This document owns durable state; [POUR_TWO_SEED_PROTOCOL.md](POUR_TWO_SEED_PROTOCOL.md) owns the unchanged native experiment contract. BASELINE_PLAN.md H top owns current authority; G records WP3 scope, F preserves W1/S1 and training-mechanism rules. The old open-loop feasibility prerequisite was revoked. New-world seed1701 PPO now totals262,144 transitions. Formal3M remains unreleased; older “未启动/可行路径前置” paragraphs below are historical.

## Active reviewer decision — 2026-09-13

**最新执行状态（已批准、运行中）：** 用户已批准DEV_TO_FIRST_CURRICULUM_V1从262,144到1,003,520，最多新增741,376 transitions或4h训练墙钟；保持方法、seed1701、world/reference/物理/reward/curriculum不变。2026-09-13已部署到隔离runtime并从已登记checkpoint/state启动，哈希分别为`92868d6525087ae36da2acb09982432af074d4a6d2c5208a40de7f4039b19483`/`6de5b4073d0b9ea9effb0b7323fc5cb8fda8d7ac8e9fd77479437d3ca38c08c5`。按GPU_POLICY.md约10秒准入后选物理GPU0、进程内cuda:0，首个完整更新到266,240，stage0、finite、无启动错误；524,288固定双模式诊断后不停训，末点执行原curriculum验证。正式3M不放行，不实施桌约束IK/PD或探索参数改变。此段为进行中快照，最终交付后替换。

**当前裁决/授权边界：** 用户批准的DEV_EXTEND_262144_V1已经执行完毕并按204,800新增步上限停止，不是新的启动授权。[追加开发报告](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/DEVELOPMENT_EXTENSION_REPORT.md`）记录seed1701从57,344恢复至262,144、训练墙钟2,581.625秒；全历史开发账本为270,336 transitions / 3,476.891秒。固定同编号评测显示确定性右/左边界误差中位261.46/285.82→96.85/163.68mm，stochastic为299.53/253.51→195.76/218.96mm；联合预抓取仍为0/25，训练累计0/7,343。实际policy臂桌接触：stochastic 1/25→9/25，deterministic 0/25→0/25。证据支持继续训练减少了部分手部误差，但没有达到联合门槛，也不能证明路线失败或保证继续改善。正式3M仍不放行，暂不实施桌约束IK，等待reviewer下一项方法/训练裁决。

**最新执行快照：** 最终checkpoint SHA256 `92868d6525087ae36da2acb09982432af074d4a6d2c5208a40de7f4039b19483`，配套state SHA256 `6de5b4073d0b9ea9effb0b7323fc5cb8fda8d7ac8e9fd77479437d3ca38c08c5`。恢复时显式reset新模拟器并重绑SB3最后观测；保留actor/critic/optimizer、计步和算法/RNG状态，但不声称恢复PhysX bit-state。训练stage保持0且没有新增curriculum事件，原因是262,144尚未到下一登记验证点1,003,520。结果、固定回合原始记录、曲线、接触记录、命令和完整哈希均在[版本化产物目录](README.md#archive-access)（归档：`artifacts/pour17_v1/development_extension_20260912_01/`）。旧checkpoint和产物未覆盖，无遗留ViViDex进程；不得自动续训或启动正式3M。

**前序执行快照（历史）：** [有限开发学习交付](README.md#archive-access)（归档：`artifacts/pour17_v1/development_20260912_01/DEVELOPMENT_REPORT.md`） / [曲线](README.md#archive-access)（归档：`artifacts/pour17_v1/development_20260912_01/learning_curves.png`）。训练及内部评测均接入同一WP4 overlay，32/25槽PhysX根与canonical读回误差0，source/reference/physics身份一致。seed1701从头同run完成首8192后继续至57,344步，墙钟751.205秒；累计开发65,536步/895.266秒，步数预算已尽。1627训练结束回合联合预抓取通过0；1185次40步门槛失败、442次提前接触终止，stage始终0。内部4096更新处25回合0通过，不是最终checkpoint评测。末窗口右/左终止六点误差中位216/218mm，不能宣布任务进步或可用。最终checkpoint SHA256 `8225439c081d09a7c5d83c1852e5dec17c7a51220a216f6e3621f514f5ba9699`，reload动作差0，进程正常退出；该checkpoint现仅为扩展起点证据，不得作为当前最终模型。旧产物保留。

**前序转达入口（历史，已执行）：** BASELINE_PLAN H中的“接收续包，直接进入有限开发学习”已完成，随后由本页顶部的DEV_EXTEND结果取代。该入口当时接收目标链一致及固定动作碰桌证据，未实施OWN_APPROACH_TABLE_CLEARANCE_V1，并完成WP4 overlay训练接线与57,344步开发学习。其预算和启动授权不得再次消费；来源噪声与动态碰撞风险仍未被声明解决。

**最新V10对照结论（2026-09-12，详细证据表见BASELINE_PLAN H）：** 用户提供的是成功源关节回放，不是H2S2R policy成功。只读远端canonical reset内容/hash与V10归档、WP4 base一致；V10 q0与WP4首次reset的58维q逐名字最大差0，qd与canonical一致，不存在已证实的抬肘初态差异。V10 inherited配置根(-.5,0,0)，WP4(-.75,0,0)；从V10实测q+腕pose反算的arm_center在两侧3帧均比WP4前移25cm，支持实际放置差异。桌高/尺寸、hand_C_MC腕link、env-local/wxyz接口一致，但V10入口显式覆盖PD/物体质量/材料，和WP4绑定源码不同；V10 approach约8.95s，WP4为2s，最大相邻臂命令变化约2.594°/47.535°。不能从回放顺畅推断ViViDex坐标bug或PD无问题，也不能认定全程无前臂碰撞。维持当前已分离根及reset；不自动复制旧PD、根、时长或成功q。优先自身目标来源与接近可行性；任何共同world/控制/时长变更单独明确，最终比较需实际物理参数一致。此轮只读与文档更新，无新增仿真/PPO或合同变更。

**WP4回放抽动的只读复核（2026-09-12）：** 本次不是policy评测：每50ms发送下一帧自身IK/手指关节参考，由原PD执行。reviewer从`wp4_20260912_01/output/reference_v1/trace.json`相邻命令复算：39→40左臂j3跳0.829645rad（47.535°），同期左腕位置目标只移动7.726mm；102→103左j7跳42.922°，103→104左j4跳43.358°。这些是命令变化，不是实际关节瞬时旋转。结合index29起前臂—桌接触，支持“参考关节不连续与碰撞/跟随问题均存在”，不证明PD实现或增益一定有bug，也不证明所有抽动同一根因。训练actor直接输出关节位置命令，不强制重放IK q；因此回放平滑度不新增为PPO硬门槛。优先完成H中实际训练目标的来源/变换和当前阶段可行性判断；不能靠增加训练预算修正确定错误的reward目标，也不要求参考回放先完成任务。未改代码、方法、预算，未启动仿真/PPO，正式3M未放行。

**前序执行优先级（历史，已完成）：** 用户曾要求优先修影响训练的根因；WP4静态根分离、目标来源对照、训练入口接线及有限学习均已完成。世界hand target并非优化后q的FK；配套A与world root转换在源帧点重组关系中抵消，不能盲目归因于A或NLopt。H2S2R object trajectory正常仍不是hand reference正确性对照。

上述前序入口最终从头完成57,344步，之后经单独批准恢复到262,144；两项额度都已耗用并封存。保持所有方法/阈值/阶段规则，正式seed0/3M仍未放行。

**User-confirmed deadline:** 2026-09-15 14:00 America/Los_Angeles (PDT), equivalent to 2026-09-16 05:00 Asia/Shanghai. The latest user direction is to work on Pour17 first because it is the proving task for the same pipeline later used on pour, sweep, take off bottle cup, and clean plate. The other three tasks are deferred for now. Reusable reference/control/train/save/eval interfaces are required; a successful PPO update alone does not establish that the task has been learned.

**Active execution direction; approval boundaries below:** selectively reuse existing Isaac scene/assets/sensors, integrate a ViViDex-owned bimanual reference and state-RL adapter, and measure a real update/save/reload/evaluation loop in the first work package. Stop repeated decoder/FK/A/residual audits without new contradictory evidence. Do not wait for old M1-A to become a retrospective PASS. The active plan replaces the old serial audit schedule; the old facts and fairness constraints remain valid.

**Latest user decisions and execution:** D1 (new common benchmark, including the G2 measured-wrist anchor) is approved. D2 mean/hand-geometry/time/validity/prefix-suffix conventions are approved; the user has now additionally approved W1 common world placement and S1 bimanual Pour synthesis as written in plan F. This resolves the method approval hold, not physical reference feasibility. D3's joint training route is approved. The comparison is whether human-video → RL correction methods learn the task; visual-policy training and its dedicated demonstration dataset collection are explicitly excluded. State-RL trajectory augmentation and curriculum must be pursued and verified, not treated as irrelevant or automatically disabled for a main result. Do not repeat approval requests for approved items. WP1 has now executed a development-only dual-hand reference → Isaac control → PPO update/save/reload → registered evaluator loop; its 0/2 task result is not a learned-task claim. Native MuJoCo training remains ended. Both methods need the same agreed evaluation version before a comparative claim; old Ours scores cannot be reused.

**Discussion clarification:** the 47.8998-degree pose discrepancy was a full-SO(3) diagnostic, not a symmetry-aware functional error. The user does not want immaterial cup/bottle rotations to block progress. Reviewer agrees: axial spin may be irrelevant for an axisymmetric object; mouth/long-axis tilt, center placement and hand-object geometry still matter. Check these within the existing scene rollout, not another audit campaign. Historical frame provenance or quaternion mismatch alone is not a runtime failure. Original ViViDex reward nevertheless uses full rotation distance, so deleting orientation reward or adding symmetry-aware reward would be a separate change, not implied by this discussion.

**Policy scope clarification:** 58 is the robot's total controlled dimension, also when combining two 29-dimensional policies. Joint PPO is the approved first route for implementation simplicity and coupled execution, not mandated or validated against independent bimanual policies by the ViViDex paper. Successful-trajectory collection and BC/diffusion train a separate visual student; they do not update the already-trained state teacher. Their exclusion is the user's chosen comparison level, not a deadline compromise. Evaluation records, verification videos and final state-policy assets remain required. Missing state-RL augmentation would be a substantive deviation; an unaugmented short run is development evidence, not an automatically authorized substitute for the main baseline.

**WP1 reviewer verdict:** accept the development loop; do not accept formal-training readiness or faithful completion of state-RL mechanisms. The 16 code/config hashes and the reference, physical trace and 8192-step checkpoint hashes match the supplied manifest. No repeat PPO-plumbing audit is needed. Both task failures remain development results, not evidence against joint PPO or a learned-task claim. The reviewer did not modify functional code, launch Isaac or start training.

**Training-mechanism follow-through:** [the WP1 mechanism report](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/training_mechanism_report.json`） correctly identifies absent live augmentation, Pour synthesis and automatic curriculum, but its reward completion claim is too broad. The current reward uses ten-point means in both phases and fingertip-force contact as the overall gate; the pinned source uses palm plus five tips before pregrasp, five tips afterward and any hand-object contact as the overall gate. Current tests cover only a self-written pregrasp formula, not all source branches. Current state observations omit original 1/5/10-index future object goals; current home-only action bias also differs from the source warm start. WP2 now covers source-semantic fixes together with integration, not just the already-reported missing classes. Revised estimate: 8–10 engineering hours, with concrete 2h/4h failure checks and one final milestone. No-augmentation is not an authorized main result.

**Current method/launch boundaries:** joint state PPO and prior user-approved input/evaluation boundaries remain. The user explicitly approved W1 and S1 according to plan F; implement those exact rules without asking again. Ordinary render/contact/source-port/curriculum/batch-eval repairs can proceed without repeated permission. Formal launch waits for actual WP2 readiness, not another vote on the approved methods. The reviewer has selected a capacity plan of one formal seed0, requested 3M and actual 3,002,368 global transitions, with the last complete update evaluated; no development-checkpoint resume or automatic second seed. At WP1 conservative speed, training alone extrapolates to about 26.82h. This is not new-version throughput evidence or a guarantee of learning.

## Purpose and approved scope

**Black-table correction and pause decision (2026-09-12, latest local delivery):** current `env.py` now uses `(0,0,0)`; reviewer viewed `artifacts/pour17_v1/wp2_20260912_01/black_table_correction_v1/camera_frame0.png` and confirmed the black table. The implementation's `camera_review.json` reports PASS_VISUAL_ONLY; the earlier manifest's CAMERA_PENDING label predates that review. `pause_resume_record.json`, recorded16:51:03+08:00, says only subsequent dispatch was paused: no running job was interrupted, WP2 PPO had0 completed steps and no checkpoint, and formal training had not started. These are delivered records, not a current remote-process check. Color no longer blocks WP2; contact and physical-reference gates remain. Do not migrate unrelated H2S2R physics presets or claim complete physical equivalence. Old videos/checkpoints remain valid within their original evidence scope. Reviewer changed documentation only.

The project goal is a ViViDex method baseline with a reusable pipeline, independently trained task policies and success-rate data. Current development and first task validation are dual-arm, dual-Sharpa Pour17. The other three tasks are not in this execution package. Visual-policy training is explicitly outside this experiment; comparison ends at the trained state policy's task execution.

Native reproduction uses MuJoCo + Adroit; SAPIEN and Place repair remain paused. Independent training starts from supplied preprocessed references, not raw video. It does not reproduce all paper claims. An Adroit checkpoint cannot directly control Sharpa. ViViDex MANO references remain separate from Ours Task 5's Dexonomy-only grasp pipeline.

## Current evidence and unresolved gates

### 当前baseline情况：来源链无接线bug，接近动作受桌阻挡；最小方法提案待决

**续包最新交付：** [来源/短测/提案报告](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_followup_20260912_01/FOLLOWUP_REPORT.md`）。异常源frame0直接MANO world经W1已有桌内点，最终目标/reward时刻一致到浮点精度；正常左frame81用已有数组复核，无额外解码。无需生产目标接线修复。固定自身q40理论六点误差18.5/28.2mm，但40步直接保持及两秒关节插值均持续右l5/左l6—桌接触，实际误差约120/129mm，直接保持已基本稳定。当前检查到的接近路径未满足H启动条件；不证明所有policy不可行。提交唯一自身arm approach桌分离约束提案，未实施，不改共同root/reset/PD/2s/目标。物理累计700/2048槽，新增PPO=0，剩余学习57,344步/3455.939秒；暂停不是0成功或其他方法未上线。下方WP4报告保留为前序证据。

**最新本地交付（非reviewer训练放行）：** [WP4报告/视频](README.md#archive-access)（归档：`artifacts/pour17_v1/wp4_20260912_01/WP4_REPORT.md`）。两槽root=(-.75,0,0)实测只施加一次，固定躯干间隔47.725mm、原接触消失，首次/重复reset一致；桌/物体初态未变。自身271样本参考已重建，操作目标/手指数组保持，IK失败右30/左92。两槽连续PD至返航完成；step40实际六点均值右138/左196mm，step143右294/左11mm；index29起前臂l6—桌有真实接触。PhysX target/command及actual-q-FK/runtime接线核对一致，不能把剩余误差统称为root或MANO错误。累计620/2048槽转移，新增PPO=0，旧checkpoint未续训；另一方共同runtime读回PENDING。H在执行中新增的来源对照/开发学习补充尚未执行，本次依最新直接上传授权“不追加PPO”交付，不将补充写成已完成或永久取消。正式3M仍未放行。

**当前reviewer裁决（取代下方交付时待批状态）：** 用户明确要求reviewer决定；批准 `CW_ROOT_NEGATIVE_X_075_V1` 完整固定机器人env-local位置 `(-0.75,0,0)m`、旋转不变，进入隔离短验证及双方各自的臂参考重建，规则和完整WP4见H。不再请求同一根候选批准。此为approved-for-short-validation，不是正式world验收或3M放行。reviewer本轮未执行候选/修改功能代码。旧bundle保持原样；新world采用版本化overlay/canonical规则，实际固定锚点与clone偏移只施加一次。ViViDex implementation仅改自己的适配，对方由自己的implementation同版验证；无需等待对方才开始ViViDex短测，缺少对方readback不能宣称共同runtime冻结。新增PPO=0，旧world checkpoint不自动续训。此改动不消除仍在桌内的手部目标，也不证明所有腕误差由躯干碰撞引起。

**WP3本地交付（非reviewer放行）：** [WP3报告](README.md#archive-access)（归档：`artifacts/pour17_v1/wp3_20260912_01/WP3_REPORT.md`）确认固定躯干 l2/l3 同时存在visual及enabled cooked collision与桌面相交，两槽目标pair接触持续。根/桌忠实于登记USD，首次与重复公共reset的q/qd/target/root/body均一致，未发现对应生产接线矛盾。41控制步/82槽转移，新增PPO=0；既有WP2产物不变。按G停止依赖world的新参考物理执行，左腕系仅CPU对照，右侧缺少保存21点未重解码。提交单一[字段级提案](README.md#archive-access)（归档：`artifacts/pour17_v1/wp3_20260912_01/common_world_revision_proposal.json`）：共同固定根(-0.75,0,0)m，未执行、未证明臂可达，须双方同版验证及各自臂参考重建。正式3M仍未放行；M0 BLOCKED、M1-A NOT PASSED。以下WP2事实保留为此前证据，不代表WP3后的实时缺项状态。

本次WP2结果已交付。reviewer接收接触/训练机制/录像/批量评测集成，不放行正式训练。用户截图提出躯干与桌面的视觉相交和reset疑点；下一包见G，先核实场景/reset，再结合既有执行数据和同事腕系对照定位，不直接换方法。以下为本地交付证据，不是远端实时进程检查。

**场景疑点的已知边界：** reviewer查看改色前棕桌探针和改色后首帧，前者已有桌板横穿躯干所在区域的视觉关系；颜色补丁不含transform修改。旧seed20260829报告的58个reset q与登记canonical差0，但这不验证根/躯干/桌面几何；登记root由USD固定根决定，历史 `(0,0.3,1.2)` 明确是占位不能照抄。两条用户观看的失败视频是WP1/286D兼容重执行，不是新版WP2/544D策略。现有FK-runtime腕一致不能证明世界放置合理；需区分遮挡、visual重叠和collision交叠。reviewer仅只读检查及更新文档，未修改功能代码/运行Isaac。

**当前版本是ViViDex方法的双臂、双Sharpa Pour17状态RL适配开发版。** 自身raw MANO/物体视频→既定十点手指retarget→腕部/双臂IK及W1/S1参考→Isaac→联合58D PPO→共同G1–G4 evaluator。视觉student排除，增强/curriculum必须保留；其余三个task暂缓。已具备开发训练闭环，尚无通过验收的正式Pour17 policy或正式success rate。

| 层级 | 已有证据 | 当前缺口 |
|---|---|---|
| 原版MuJoCo | 独立训练及固定评测完成一seed | 不能作为Sharpa Pour17成功证据 |
| WP1开发闭环 |8192-step PPO更新/保存/重载、32×903物理smoke；有开发checkpoint | evaluator0/2，均903步timeout；不能作正式SR或方法失败结论 |
| 手指参考 | 双侧参考已生成；左手完整142帧候选已检查，顺序热启动且无同事包的0.9后裁剪 | 贴限位和点对距离冲突仍在；近似解是否足以抓取未证明 |
| 最新S1/双臂参考 | 已有271样本目标；名义倾倒几何检查通过 | `reference_s1_manifest.json`逐行汇总：右146/271、左196/271 IK未通过；至少一臂关节贴限位样本右157、左201；最近秩P95腕位置误差均约24cm。不是实际执行已修复 |
| 参考与桌面 | `source_to_port.md`报告index40左手ring/pinky目标位于桌面cuboid内；小指tip/middle z=0.845870/0.843660m，桌面top0.87m | 目标点在桌实体内是具体几何风险；不是实际机器人碰撞测量，也不单凭这两个点断言整个pregrasp必然不可达。须本次连续执行解释，不能移动桌/目标掩盖 |
| 接触与视觉 | 黑桌、相机、6组物理接触正反例通过；旧策略两录像及新参考open-loop已交付 | 不能以接触传感器对照PASS认证整个机器人—桌面放置；用户指出的相交关系仍待核实 |
| 原版状态RL机制 | 新版从头8192-step更新/重载通过；18个源reward分支对拍、实际非identity reset；真实curriculum0/25，stage0；batch32×903全timeout | step40实际双手误差129.55/176.63mm；原左index141/S1 index143有239.42mm参考IK误差。保存重载/RNG检查不等于完整中断续跑。尚无任务成功或自然课程升级 |
| 正式训练与比较 | 容量计划seed0、3,002,368 transitions、最终512回合；WP2测得56.86 transitions/s，带25%余量训练约18.33h、评测约2.10h | 尚未放行；吞吐仅stage0短失败回合，后续阶段和正G2开销未知。共同runtime/场景合理性仍未闭合；不能复用旧Ours分数 |

**下一次评审只作集中决定：** (1)真实腕/手/接触是否已排除阻断训练的错误；(2)机制是否真实连入新版PPO且可保存恢复；(3)实测预算能否容纳训练、512评测及结果整理。三者成立就放行正式训练；若是具体接线错误则限时修复；若需改变目标/腕系/尺度等方法则结合同事包提交明确备选。不要求训练前任务已经成功，也不等待所有局部几何零误差。现有计划最迟09-13 14:00北京时间启动、09-15 05:00停训、09-15 17:00封存结果，均是计划节点而非保证；延迟不能靠挤掉评测时间隐瞒。

**Colleague retarget package review (2026-09-12):** plan F records the read-only review of `mano2sharpa_retarget_20260911.tar.gz`. Both packaged hands have32 named joints whose checked kinematic/limit fields match the corresponding current robot URDF fields. Its local-frame/world-wrist conventions are useful diagnostic comparisons; its vector/DexPilot objectives, scale1.07, low-pass and0.9 clip are not approved replacements for the current reference. Current retarget already uses sequential warm starts without that post-clip. No example300-frame outputs or task-success evidence were included; production `valid` describes input validity. No package scripts were executed and no existing retarget/physical-readiness gate was marked passed. See plan F for the bounded WP2 use and limitations.

| Area | Verified state | Limit |
|---|---|---|
| Native Pour | Seed 0 completed independent training and fixed evaluation; evidence below | One completed seed, not multi-seed stability or whole-paper reproduction |
| MANO decoder | v4 trusted-layer comparison and evidence binding passed for both explicit mean variants | Historical HaWoR mean/model/runtime identity remains UNKNOWN |
| Candidate left Sharpa | 22-joint FK, Jacobian/gradient and synthetic NLopt solves passed | Internal consistency, not authoritative simulator equivalence |
| Full sequence | Both mean variants cover source frames 0..141; 284/284 execution checks passed | Candidate geometry, not physical/task validation; D2 input/ten-point/time conventions were subsequently approved as engineering choices |
| Fixed A | Registered construction exactly reproducible; no explicit implementation contradiction found | Cross-model anchor choices remain engineering conventions |
| Residuals | Bound occupancy and point-pair incompatibility documented | All 284 samples exclude exact ten-point zero error; approximate retargeting may still be useful. Bounds/posthoc gains do not establish a unique cause |
| Surface display | 28 references recovered as 13 exact LFS objects, 5,954,492 bytes; q/Y unchanged | Reviewer viewed fixed five-frame surfaces; full GIF NOT reviewed frame by frame. No collision/physics/task acceptance |
| Formal M1-A | **NOT PASSED** | Candidate diagnostics completed, formal acceptance not completed |
| M0 | **BLOCKED** | No authoritative common runtime/run binding; archive is only a candidate |
| Pour17 WP1 integration | **Development vertical loop accepted**: 269-step reference, 1/32-env Isaac wiring, 8,192-step PPO save/reload, 2 registered evaluator episodes and 32×903 long-horizon smoke | Reference records 168 right/194 left IK failures; evaluator is 0/2 with both 903-step timeouts; no formal training/task-success claim |
| Reference execution near pregrasp | Current trace index38 has right/left ten-point errors 14.87/22.53cm; current env decides both-side pregrasp <5cm at step39. Wrist IK position-error P95 is about 23.91/23.64cm | A low median over all 58 joint errors does not establish executable wrist/hand references. Six bounded same-solver home retries did not rescue the five failing examples; not proof of global infeasibility |
| Contact semantics | All ten fingertip booleans are false throughout the 269-step open-loop trace | Could be lack of fingertip contact or a sensor problem; object motion alone does not distinguish them. Missing force matrices currently become zeros. Physical positive/negative controls and distinct training/evaluation contact semantics are required |
| Evaluation capacity | Scalar evaluator already reuses Kit; 512-episode extrapolation is 40.20h | Batch per-slot progress/G2 execution and measured cost must be delivered before scheduling final evaluation; do not silently reduce the common 512 episodes |
| Available scene scaffold | H2S2R Pour17 has scene/assets/sensors and a canonical-t0 branch | Its default G2 reference-derived reset, clamp, controller, reward and evaluator presets must not be inherited |
| Fresh frame alignment finding | Direct first-frame pose-to-reset transforms disagree by 47.8998 degrees between the two objects | Full quaternion discrepancy alone is not a blocker; axial symmetry was not separated from functional tilt/frame effects |
| Approved W1 shared world placement | Gravity-preserving yaw/translation from the two centers yields approximately 5.78248 mm center error each. Existing trace/reference, interpreted with registered local +Y mouth axes, gives cup/bottle axis differences about 5.35/8.49 degrees and mouth-position differences 8.37/14.42mm at the initial sample | Supports dropping the full-quaternion alignment blocker, conditional on the local-frame interpretation; does not certify wrist feasibility or historical identity. W1 is a user-approved engineering convention; it is not a physical-execution PASS |

Detailed evidence: [M1-A audit](M1A_INPUT_GEOMETRY_AUDIT.md).
Latest display: [surface GIF](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_synchronized_surface_v1.gif`）, [fixed five-frame surfaces](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/m1a_full_sequence_candidate_fixed_frames_surface_v1.png`）, [delivery and limits](README.md#archive-access)（归档：`artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/M1A_FULL_SEQUENCE_CANDIDATE_SURFACE_VISUALIZATION_DELIVERY_V1.md`）.
The fixed-view review found no immediately evident whole-model wiring error; it did not select a mean or approve geometric/task quality.

WP1 evidence: [component report](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/wp1_report.json`）, [throughput and budget extrapolation](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/throughput.json`）, [training-mechanism gaps](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/training_mechanism_report.json`）, and [development checkpoint](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/final_model.zip`）. The encoded open-loop MP4 and a camera-enabled retry are byte-identical and all-black; see [video validation](README.md#archive-access)（归档：`artifacts/pour17_v1/wp1_20260912_02/video_validation.json`）. They are not visual evidence. The two failed evaluator episodes were headless and have no corresponding videos. Explicit camera/render wiring is included in WP2's existing evidence-delivery scope; subsequent videos must be labeled as re-executed episodes, not recovered footage of the original runs.


## Verified sources and deployment

- Official source: https://github.com/zerchen/vividex_mujoco, commit `9790140170d8be49b828ab214ce3537a2475fcea`.
- Objects submodule: `96a275cab048eec625edfb6814f551fb9c329981`; openpoints: `2bc0bf9cb2aee0fcd61f6cdc3abca1207e5e809e`.
- Archived, inactive SAPIEN checkout: `b22a07ec60ffc29db80e0d11f8477c5ca1fc0fb6`.
- Published model sources/checksums: [checkpoint manifest](README.md#archive-access)（归档：`verification/checkpoint_manifest.json`）, [download index](README.md#archive-access)（归档：`checkpoint_index.json`）.
- Installed versions: [dependency lock](README.md#archive-access)（归档：`verification/requirements.lock.txt`）; setup: [setup_mujoco.sh](README.md#archive-access)（归档：`setup_mujoco.sh`）.

Remote root **R** = `/media/msc-auto/HDD/users/kailang/baselines/vividex_mujoco`.
Host `kailang@128.32.164.89` / `mscauto-Lambda-Vector` is one machine with two RTX A6000 cards. The ended campaign used GPU0. The latest WP1 runs used physical GPU1 only when no other compute process was present; GPU0's other-user process was not touched. This is not a reservation. Keep new source/environment/cache/temp/output on the data disk under R.

**WP2 execution snapshot (2026-09-12 17:54 Asia/Shanghai):** [WP2 milestone report](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/WP2_REPORT.md`）, [machine report](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/wp2_report.json`）, and [176-file manifest](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/wp2_delivery_manifest.json`）. Black-table color-only diff/camera checks passed; the earlier pause was dispatch-only with0 PPO updates, not lost training. Both old WP1 failures now have903-step/904-frame black-table recordings. GPU contact controls passed6/6 using dynamic sensors and the actual static-table-collider filter, without CPU-readback or physical-contract change. Fresh WP2 seed1701 completed8192 transitions/2 rollouts in144.061s including the true25-episode callback; finite losses, parameter change, exact reload actions and next-RNG-draw restoration passed. Callback pregrasp0/25, stage remains0. Last full checkpoint `output/ppo_v1/final_model.zip` SHA256 `602c652549bf8c663ca3eca5d41d9ef2ac43888b908f5a50fe9519ffd2197e90`; no WP1 resume, cumulative budget not reset. Actual batch32×903 completed in401.002s, all32 timeout/G1–G4 false;512 cost with25% margin~2.10h. Reference gate remains **NOT READY**: step40 six-point means right0.12955m/left0.17663m exceed0.05m; original left141 maps to S1 index143 with0.23942m IK residual and three active arm limits. Actual-q FK agrees with runtime at checked states; no global unreachability or sole-cause claim. Formal3M and final512 were not started; [review-only commands](README.md#archive-access)（归档：`artifacts/pour17_v1/wp2_20260912_01/FORMAL_COMMANDS_REVIEW_ONLY.md`） are not release. Stop at reviewer milestone; old artifacts, physics/control/reset/W1/S1 and historical gates preserved.

`R/env/bin/python`: Python 3.10.20, PyTorch 2.4.1+cu121, MuJoCo 2.3.7, dm-control 1.0.14, SB3 1.1.0, Gym 0.23.1, NumPy 1.23.5. Gym is pinned for the old SB3 seed API. Dependency consistency and CUDA availability were verified. W&B is disabled. Infrastructure reference: [environment inventory](README.md#archive-access)（归档：`../../Reports/环境报告_ENVIRONMENT_INVENTORY.md`）, section 16.

Local upstream is a shallow reference checkout; remote upstream has full history and installed submodules. Remote original working tree was clean on 2026-09-07. Do not infer local asset completeness from remote installation.

History audit: recent upstream commits are README changes. Relevant `control.py` commit `24f9688f94d209ccaea6f78d5c2d7ba054d00920` changes qpos/qvel slices from 36 to 42. Earlier matching source is initial commit `1a85392c5b77b1f3791f865325ee5735f52ffba9`. Workspace-root Git history was not readable via `git rev-parse`; local adaptation commit history is not reconstructed. Saved patches and artifacts establish current changes.

## Original code versus adaptations — verified

| Component | Provenance / status |
|---|---|
| Assets, references, dynamics, reward, curriculum, PPO architecture/hyperparameters | Original pinned MuJoCo implementation |
| Dependency deployment | Our isolated compatible runtime |
| Training copy | `R/pour_multiseed_source`, original commit plus three source-file changes |
| Reproducibility wiring | [multiseed_repro.patch](README.md#archive-access)（归档：`verification/multiseed_repro.patch`）: PPO/task seeds, removal of entropy reseeding, shared validation RNG, final updated model saving |
| Fixed evaluator | [eval_pour_fixed.py](README.md#archive-access)（归档：`eval_pour_fixed.py`）: 300 shared reset seeds, original particle geometry, per-episode records and hashes |
| Queue utilities | [run_pour_campaign.py](README.md#archive-access)（归档：`run_pour_campaign.py`）, [reduce_campaign_to_two.py](README.md#archive-access)（归档：`reduce_campaign_to_two.py`）; orchestration, not a new learning method |
| Place candidate patch | [place_metric_compatibility.patch](README.md#archive-access)（归档：`place_metric_compatibility.patch`）, unapplied and deferred |

The training diff modifies only `tools/train.py`, `hand_imitation/env/environments/gym_wrapper.py`, and `hand_imitation/env/utils/util.py`. Original config seed alone was insufficient: PPO did not receive it, and task RNG was not wired through the wrapper. Reward/physics/curriculum are unchanged. Reference-derived action-bias warm start remains enabled; this is not pretrained policy loading.

## Observation and action interfaces — verified

[control.py](README.md#archive-access)（归档：`upstream/vividex_mujoco/hand_imitation/env/models/control.py`） defines a normalized 30-vector action in [-1, 1], clipped and mapped linearly to actuator control ranges. `hand_imitation/env/models/assets/robots/adroit/actuators.xml` defines 30 position controls: 6 overall translation/rotation, 2 wrist and 22 finger actuators. Advertised hand DoF must not replace actual action dimension.

The current policy consumes `state`, not images: sliced joint positions/velocities; hand/object body positions, rotations and velocities; reference goals; relative hand/object/goal positions; and sine/cosine phase features. See `SingleObjectTask`, `ReferenceMotionTask` and `ObjMimicTask.get_observation`. [Pour config](README.md#archive-access)（归档：`upstream/vividex_mujoco/tools/experiments/env/dexycb_pour.yaml`） enables random episodes and appended time.

| Environment / source | State width | Action width |
|---|---:|---:|
| Relocate mustard, current source | 346 | 30 |
| Pour, current source and author model | 358 | 30 |
| Place, current source | 358 | 30 |
| Place author model and matching initial source | 346 | 30 |

Evidence: [smoke report](README.md#archive-access)（归档：`verification/smoke_02/report.json`）, [recording metadata](README.md#archive-access)（归档：`verification/checkpoint_videos_01`）. Sharpa mappings and future observation/action spaces are not established here.

## Reward and evaluation — verified definitions

[rewards.py](README.md#archive-access)（归档：`upstream/vividex_mujoco/hand_imitation/env/models/rewards.py`） implements `ObjectMimic`. Before pregrasp success, reward exponentially tracks retargeted hand positions. After pregrasp, contact-gated reward combines finger-contact bonuses, exponential object translation/orientation tracking, lift bonus, and hand tracking gated by close object tracking. Pour config sets object error scale 50, object reward scale 10, lift threshold 0.02, lift bonus 2.5 and object tracking termination threshold 0.25. Detailed phase/termination conditions remain defined by this source. Online pregrasp/imitation success statistics are not final Pour scores.

Final Pour evaluation counts the last 12 particle bodies strictly inside a 0.15 × 0.15 container centered at x/y (-0.08, -0.1), with 0 < z < 0.08; divide by 12, then average across episodes. Source: official `tools/eval_state_policy.py`, preserved by our fixed evaluator. Its original SR@3/5/10 labels print the same particle fraction for Pour: neither centimeter thresholds nor binary episode success.

**Decision:** stage 2, deterministic final policy, 300 reset seeds 200000–200299, preserve individual particle counts and initial physics hashes. Reward curves cannot substitute for final evaluation.

## Verified evidence and its limits

| Evidence | Result | Interpretation |
|---|---|---|
| [Smoke report](README.md#archive-access)（归档：`verification/smoke_02/report.json`） | Three scenes reset/step/EGL render; 32-step PPO update/save/reload | Runtime plumbing |
| [Training-entry log](README.md#archive-access)（归档：`verification/train_entry_smoke_01.log`） | Short official entry completed | Entry/wrapper/callback compatibility |
| [Mustard checkpoint log](README.md#archive-access)（归档：`verification/checkpoint_eval_01.log`） | Author pose1, 100 episodes, SR@3/5/10 = 1.0 | Published-policy compatibility |
| [Pour checkpoint log](README.md#archive-access)（归档：`verification/checkpoint_eval_pour_01.log`） | Author model, 100 episodes, fraction 0.9875 | Published-policy compatibility |
| [Seed checks](README.md#archive-access)（归档：`verification/multiseed_checks.json`） | Repeated seed 0 hashes match; seed 1 differs | Tested initial weights/reset reproducibility; not cross-hardware determinism |
| `R/outputs/pour_fullsize_pilot_01` | 8192 transitions, final save, three-episode evaluation fraction 0 | Full-size plumbing test, excluded from formal results |
| [Seed 0 fixed evaluation](verification/seed_0_fixed_eval.json) | 60,002,304 training steps; 300 episodes; mean particle fraction 0.846944 | Sole completed independent training result; seed 1 was subsequently cancelled by the user |
| Place playback / [diagnosis](README.md#archive-access)（归档：`verification/place_mesh_diagnosis.json`） | Matching-source rollout works; volume evaluator fails | No valid Place quantitative score |

The original 100-episode checkpoint runs did not record fixed test seeds. They are not the matched 300-episode test. Saved videos are single unselected rollouts, not aggregate evidence. None of these short tests establishes independent training reproduction.

Place uses `R/upstream/vividex_mujoco_initial` for its 346-wide model, not manual tensor truncation. Its banana mesh fails trimesh volume checks. The in-memory seam-vertex merge candidate remains unapplied by decision; this is not an active Pour blocker.

## Completed native campaign — decisions and final record

The original [two-seed protocol](POUR_TWO_SEED_PROTOCOL.md) planned seeds 0 and 1 at 60,000,000 requested transitions from scratch. After seed 0 completed and demonstrated a usable policy, the user explicitly ended the campaign to avoid further compute. Seed 1 was cancelled at 13,918,208 steps without final evaluation. It must not be reported as a failed seed or included in a two-seed mean; this campaign now provides one completed training result and does not support a multi-seed stability claim.

**Final campaign status, 2026-09-08 11:02 UTC:** seed 0 completed exactly 60,002,304 transitions and its final checkpoint passed the fixed 300-episode evaluation with mean particle fraction **0.8469444444**. Its checkpoint SHA-256 is `d988b1b19a9610c8c65b4bda887bda59f708134281c490a69abcdb66484fd120`. All 300 records and reset seeds 200000–200299 are present; the local copy is [seed_0_fixed_eval.json](verification/seed_0_fixed_eval.json). Seed 1 and the controller were then stopped by explicit user decision; seed 1 had reached 13,918,208 partial steps. The manifest state is `stopped_after_seed0`, seed 1 is `cancelled_by_user`, and no matching trainer/controller process remained at that dated check (not a current process claim). The final manifest snapshot is [campaign_stopped_after_seed0.json](verification/campaign_stopped_after_seed0.json).

Campaign **C** = `R/outputs/pour_multiseed_60m_20260906_01` (name retained after reduction).

- `C/manifest.json`: current state, seeds, commands, source/patch/data hashes, final summary.
- `C/seed_N_train.log`, `C/seed_N_eval.log`: execution evidence.
- `C/seed_N/exp_config.yaml`: actual run config.
- `C/seed_N/final_model.zip`, `completed_steps.txt`, `fixed_eval.json`: completion/evaluation artifacts when available.
- `R/logs/two_seed_controller.log`: queue log.
- `C/manifest_before_two_seed_change.json`, `C/archive/`: superseded five-seed records.
- Local `verification/campaign*_snapshot.json`: historical snapshots, not live status.

Input: `ycb-025_mug-20200709-subject-01-20200709_150949.npz`, SHA-256 `1d71e53925f983758c63f40571b2649d8574a8394b3781a565cec72445a21dd2`. Patch SHA-256 `d7aa9de63fb01d4fbca5d2158152239e255c969c3af14ec7a16ec5d0c8253fb4`. Both are in the manifest.

## Decisions not to casually reverse / rejected shortcuts

- Checkpoint playback or short pilots do not demonstrate from-scratch reproduction.
- Do not resume seed 1 or reintroduce additional seeds without a new explicit project decision. Seed 0 is the sole completed result; seed 1 was intentionally cancelled, not score-selected or failed.
- Particle fraction is not binary episode success; current reward logs are not final evaluation.
- Do not restart training to refresh documentation. Inspect live processes before acting on historical PIDs.
- Preserve the original checkout and explicit adaptation diff; avoid untracked algorithm changes.
- SAPIEN and Place repair remain deferred. Sharpa candidate work exists; further integration needs a scoped decision, not blanket prohibition by an old roadmap.
- Config seed alone was experimentally inadequate; retain verified RNG wiring.
- Root-partition capacity and platform DoF are not substitutes for data-disk capacity and actual action width.

## Unknowns, hypotheses and next steps

**Unresolved:** author's total training budget (published 60-million-step config includes continuation); historical raw-video preprocessing chain; reference execution problems and live state-RL mechanism adaptation. W1/S1 approval and implementation are complete; successful physical execution and task learning remain unproven. Historical preprocessing identity is not an invitation to restart audits of approved engineering conventions. The ended native campaign will not produce a second completed seed or multi-seed aggregate. Adapted Pour17 now has a separate one-seed 3,002,368-transition capacity plan, pending actual training readiness; it does not reopen the native campaign. No numerical native “close enough” acceptance threshold was pre-agreed. Seed 0's 0.846944 result is below the paper's reported 0.97 Pour value and does not by itself establish reproduction of that result.

`tools/generate_trajs.py` collects 500 trajectories and uses a relocation distance <0.1 filter. It is not yet validated for Pour collection. The public README points to BC; no MuJoCo diffusion/denoising entry has been established in the inspected checkout. These visual-student downstream stages are now explicitly excluded and are not blockers for the approved state-RL comparison. Do not implement or port them in this package. Future embodiment transfer feasibility remains a working hypothesis, not a verified result.

The WP1 development package and its independent review are complete. Execute [plan F's WP2 integration prompt](BASELINE_PLAN.md), preserving D1, approved D2 portions and joint state-RL scope; visual training remains excluded. Target WP2 delivery by 2026-09-13 00:30 Beijing and conditional formal training by 02:00; hard latest start remains 14:00 that day. Training ends by 09-15 05:00, results freeze by 17:00, submission is 09-16 05:00 Beijing. These are planning limits, not promises. Do not create another component-audit queue, start a 40-hour serial evaluation without room in the schedule, or automatically substitute an unaugmented main result.

## Last-known resource/runtime state — WP1 delivery reviewed 2026-09-12

- Latest user/implementation report: physical A6000 GPU1 was released after WP1; GPU0's other-user process was untouched. The reviewer inspected local delivery evidence, not fresh remote GPU occupancy this turn. This is not a reservation; check live processes before execution and do not disturb other work.
- WP1 actually ran Isaac+SB3 in the data-disk isolation root `R/adaptation_pour17_v1`, using `venv/bin/python`. Its delivered dependency lock records stable_baselines3 2.7.0, Torch 2.7.0+cu128, Gymnasium 1.2.1 and Isaac Sim 5.1.0.0. The pre-existing base env's absence of SB3 is no longer an integration blocker. The wrapped text lock should not be treated as a verified reinstall script.
- Last disk check remains 2026-09-11 17:29–17:33 UTC: system partition backing `/home/kailang` about 6.3 GB free, data disk `/media/msc-auto/HDD` about 2.8 TB free, not an account quota. Recheck before deployment; keep new environments/cache/output on the data disk.
- Existing Isaac base: `/home/kailang/.local/miniconda3/envs/rl-correction-pour`; source `/home/kailang/opt/IsaacLab-5.1`; scene scaffold `/home/kailang/experiments/baselines/rl_correction_h2s2r_adapter`; registered bundle `/home/kailang/experiments/baselines/pour17_baseline_bundle_20260829/pour17`. WP1 runtime use does not establish an authoritative Ours run.
- Native `R/env/bin/python`, campaign manifest and seed0 checkpoint existed at the earlier live check; native manifest was `stopped_after_seed0`, seeds `[0]`. No native replay or campaign restart is authorized by this review.
- Local RTX 5080 **Laptop** has 16303 MiB. Last local disk/GPU check was also the earlier review: D: about70.2 GiB free; no local Isaac runtime was established. Do not count it as a ready second training device.

## Source / operations map

### Fresh reviewer resource index

Local paths below come from the handoff and selective 2026-09-12 checks. Remote paths explicitly listed in the live snapshot were re-probed for existence; other remote provenance remains historical. Root `git status/log` previously returned “not a git repository”; this review did not repair or reconstruct root history. Local upstream HEAD/cleanliness below is the earlier handoff observation, not a new diff audit.

Let **B** = `D:/UCBP/RL/Baselines/ViViDex`. Start with this document, then BASELINE_PLAN.md's replanning notice; open detailed audit sections only as needed.

| Resource | Location / authority |
|---|---|
| Infrastructure and SSH/account instructions | `D:/UCBP/RL/Reports/环境报告_ENVIRONMENT_INVENTORY.md` (actual file). One host, two A6000s, kailang account; use this document's dated live snapshot, then recheck before execution. |
| Native code and adaptation | `B/upstream/vividex_mujoco`; root utilities and `B/verification/multiseed_repro.patch`; CPU geometry adapters/solvers are in `B/diagnostics`; current Isaac integration is in `B/adaptation/pour17_v1` |
| Original perception | `D:/UCBP/_inspection/pour17_baseline_bundle_20260829/pour17/perception/pour17_perception.npz`; schema/hash and valid-frame limits in M1A_INPUT_GEOMETRY_AUDIT.md |
| Registered manifest | `D:/UCBP/pour17_baseline_bundle_20260829.tar.gz`, member `pour17/world/world_manifest.json`, SHA-256 `2339f7a9ca3eda294908e392744c798c06fbbbc0edd79250bd5e43aaa828dcd5`; do not substitute the similarly named inspection copy |
| Registered reset/evaluator | Same tar: canonical reset SHA `d41cd5d6ab19f55d583dc3fcd458db89b48dc52e069e8d1be1d61e60ad64e679`; run_eval SHA `40e6b157e836cb3ea3cd754791c1231ced7e9ef633c184d09a6ebe125f7d6d6b`; hashes checked this review. The inspection folder has an older evaluator without the registered G2 callback/cap changes and lacks canonical_reset_v1.json. |
| MANO candidate asset | `B/upstream/vividex_sapien/mano/assets/MANO_LEFT.pkl`; using this asset does not activate SAPIEN; historical HaWoR asset identity UNKNOWN |
| Candidate Sharpa URDF | `D:/UCBP/RL/Worktrees/RL-Correction-Step4-Publish/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa_fix.urdf`; candidate only, not certified Ours runtime |
| Geometry interfaces/configs | `B/diagnostics/m1a_full_mano_decode_audit.py`, `m1a_sharpa_left_fk_candidate_v1.py`, `m1a_sharpa_single_frame_solver_candidate_v1.py`; `B/configs/m1a_sharpa_left_fk_candidate_v1.json`, `m1a_wrist_zero_hand_local_calibration_candidate_v2.json`, `m1a_full_sequence_candidate_diagnostic_v1.json` |
| Local NLopt runtime | `B/.venvs/m1a_nlopt_v1/Scripts/python.exe`; `B/configs/m1a_nlopt_v1_requirements.lock.txt` and `m1a_nlopt_environment_v1.json`; existing evidence: Python 3.10.19, NumPy 2.2.6, NLopt 2.10.0; not rerun for this handoff |
| Full candidate outputs | `B/artifacts/m1a_full_sequence_candidate_diagnostic_v1/` (JSON, NPZ and curves; both branches) |
| Exact mesh assets | `B/artifacts/m1a_sharpa_visual_mesh_exact_recovery_v1/assets/sharpa_left`; asset recovery report records source `/home/kailang/experiments/rl_correction_step4_lfs`, not proof of authoritative Ours runtime |
| Surface outputs | `B/artifacts/m1a_full_sequence_candidate_surface_visualization_delivery_v1/`; actual renderer config is `B/configs/m1a_sharpa_visual_mesh_surface_delivery_v1.json` |
| Remote native runtime/source | `R/env/bin/python`, `R/upstream/vividex_mujoco`, `R/pour_multiseed_source`; R is defined above |
| Native final checkpoint/evaluation | `R/outputs/pour_multiseed_60m_20260906_01/seed_0/final_model.zip` and `fixed_eval.json`; local evidence `B/verification/seed_0_fixed_eval.json` and `campaign_stopped_after_seed0.json` |

Current unresolved items (2026-09-13): historical HaWoR producer/model/runtime, authoritative shared Ours train/eval physics, stage-1/2 learned execution, current running package final results and task learning. W1/S1 synthesis and automatic curriculum are implemented; the million-step curriculum check has not yet been reached at the handoff cutoff. Formal seed0/3M is selected but not released. Isaac+PPO integration and throughput are now measured development facts. Remote process/GPU state is always a dated snapshot and may change; do not infer runtime authority from candidate directory names.

Paths below official checkout:

- `tools/train.py`, `tools/experiments/config.yaml`, `tools/experiments/agent/ppo.yaml`: training/defaults.
- `tools/experiments/env/dexycb_pour.yaml`: active environment and reward parameters.
- `datasets/dexycb_traj/pour.yaml`: author config reference; active resolved run config takes precedence.
- `hand_imitation/env/models/control.py`, `reference.py`, `rewards.py`: interfaces, references, reward/termination.
- `tools/eval_state_policy.py`: original evaluator.
- `tools/generate_trajs.py`, `tools/dist_bc_train.py`, `tools/dist_bc_test.py`: downstream visual route, explicitly excluded from the current experiment.

Local utilities:

- [prepare_multiseed.py](README.md#archive-access)（归档：`prepare_multiseed.py`）: assumes original source text; not an idempotent patch reapplication tool.
- [check_multiseed.py](README.md#archive-access)（归档：`check_multiseed.py`）: seed checks.
- [run_pour_campaign.py](README.md#archive-access)（归档：`run_pour_campaign.py`）: fresh campaign launcher, not resume.
- [reduce_campaign_to_two.py](README.md#archive-access)（归档：`reduce_campaign_to_two.py`）: one-time takeover with historical PIDs/five-seed preconditions; **do not rerun**.
- [smoke_mujoco.py](README.md#archive-access)（归档：`smoke_mujoco.py`）, [record_checkpoint_episode.py](README.md#archive-access)（归档：`record_checkpoint_episode.py`）: runtime and recording checks.
- [download_checkpoints.py](README.md#archive-access)（归档：`download_checkpoints.py`）, [inspect_place_mesh.py](README.md#archive-access)（归档：`inspect_place_mesh.py`）: acquisition and deferred mesh diagnosis.

On infrastructure failure inspect manifest, processes and artifacts before proposing recovery. The queue is not a general resume facility. These docs do not authorize new runs, automatic retries or application-level monitoring.

