# PMIN_P17s51 A 与固定输入 D 的证据差异

本表只比较与左手/杯接触链有关、且有实际版本证据的项目。A1 表示文件或运行输出明确绑定 `PMIN_P17s51`；A2 只是同名 Git 候选，不能提升为执行版本。D 是 v7 `warmup0_retry3` 的 100 条历史目标命令回放，不是 A，也不是 B10。

| 项目 | A：PMIN_P17s51 | D：本包实际链 | 证据等级与结论 | 是否可作单项移植 |
|---|---|---|---|---|
| 物理/控制时钟 | `dt=1/240 s`、decimation 12、control 0.05 s | 同值 | A1 `world.json`；D 冻结输入/实际 completion | 否，已一致 |
| 58 关节顺序 | A1 清单为左右臂+两手共 58 关节 | D 运行时逐名核对；左 29 顺序来自冻结输入 | A1+D 实测；一致 | 否 |
| 机器人 USD | 运行路径 `/home/yanghong/.../vega_1p_sharpa_fixedtorso.usd`，MD5 `143385ad217f10f3fe730338d748246d` | D `RUNTIME_LOADED` 同 MD5 | A1+D；机器人内容一致 | 否 |
| 杯 USD 内容/烘焙碰撞 | 运行路径 `/home/yanghong/.../datasets/pour17/cache/object_0.usd`；未保存 bytes/hash、依赖闭包或 cooked identity | D source object SHA256 `6ba6ae859a273d15106767855ec5fcfef3f51cf47f5e7449fd230ad9f5b23e35`；D cache SHA256 `20478176…`，运行保存实际 collision prim 元数据，但 cooked geometry 不可读 | A1 缺失；A2 同名 source LFS oid 与 D source 相同，也不能证明 A1 cache/cooked 相同 | **否：无法证明唯一资产差异** |
| 左手碰撞资产/装配 | A1 仅机器人总文件 MD5与 self-collision=false；无逐 prim 碰撞/装配 manifest | D 运行保存 33 个左手 body 的实际 collision prim、approximation、offset 与 source layer | A1 缺失，不能逐部位比较 | **否** |
| 手杯 pair/filter | A1 world 声明 `replicate_physics=false`，日志可见 Aux；无 pair enabled/filter 运行清单 | D 源调用 `scene.filter_collisions()`；实际 33 手 body→杯 ContactView 成功创建 | 不能从 A2 同名源码推断 A1；没有可验证的 A1 pair 差异 | **否** |
| 质量/摩擦 | A1 杯/瓶 0.5 kg、物体与 pad 摩擦 1.0 | D `ours_physics.py` 在构造时覆盖为同一冻结 spec；`cfg.py` 中 0.15/0.5 等是覆盖前默认值，不是本次运行值 | 既有运行核验支持一致；不能把默认值写成实际差异 | 否 |
| 驱动 | A1 已存 arm stiffness/damping/effort，hand 20/2；未存完整 hand effort/velocity manifest | D 使用冻结 v7 驱动，运行记录 requested/processed/direct drive；未改 PD | 已知项一致，未知项不构成差异 | 否 |
| 初始手杯相对状态 | A 的 q0/物体 nominal pose 叠加 env0 共同 XY 扰动 `[+0.0292968,+0.02255734] m` | D 使用 v7 trace0 历史左手、腕和杯位姿，无扰动 | 明确不同，但属于初态/任务分布 | **计划禁止移植** |
| 前 15 control 状态回写 | A 对物体钳位 15 步 | D `hold=0`，只在 explicit reset 写状态；control 期间由本包记录器核查 | 明确不同，但属于冻结/回写协议 | **计划禁止作为 A 单项碰撞移植** |
| 环境源码身份 | 原运行未保存 commit/tree、`pour_env.py`、`world_fingerprint.py` 或直接父类 SHA | D `env.py` SHA256 `379ee553…`、`left_env.py` `cad387c6…`、`cfg.py` `37e0eea4…`、`ours_physics.py` `8625d002…` | A2 commit `f44e77b…` 只是候选，未与 run manifest 关联 | **否** |

## A 来源结论

在 25 分钟定向搜索内，没有找到能把 A 的具体杯/左手 collision asset、装配变换或 pair-filter 规则绑定到 PMIN_P17s51 的 A1 文件。缺失项逐条列在 [A_SOURCE_INVENTORY.json](A_SOURCE_INVENTORY.json)。因此，即使 D 真实接触复现异常，本包也没有资格自行选择一个 A2 候选做移植；初态扰动和 15 步钳位是已知差异，但权威决策表明确禁止把它们当本次自动移植项。

## 不越界解释

- A2 的同名源码、source USD LFS oid 只能缩小候选，不能证明原运行实际加载了同一 cache/cooked collision。
- D 的输入碰撞 metadata 不是 PhysX 烘焙碰撞体；实际 ContactView 的 separation 也不直接重命名为网格穿透深度。
- 这张表用于决定是否具备单项 A 移植资格，不用于证明 A 与 D 是同一实验。
