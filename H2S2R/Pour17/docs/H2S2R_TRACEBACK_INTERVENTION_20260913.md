> **2026-09-13交接截止：已完成并关闭，Isaac 1/1。** 最新RESULTS与INIT_RETURNED证实初始化返回；下方RELEASED是历史授权，不能重跑。训练尚未集成放行，新训练包待决策。本次只同步文档。

# Reviewer：周期traceback最小干预包

**Implementation final:** `INIT_RETURNED_WITH_PERIODIC_DUMP_DISABLED / IMPLEMENTATION_PASS`。用户解除上传阻塞后，实际完整GDB脚本CPU SIGSEGV验收PASS：full bt、registers、libraries、all-thread、core及单项错误后继续均验证；实际overlay加载并等待31.5秒的周期dump禁用测试PASS、stacks0字节。32/32 V13身份及科学命令字段PASS，共享门选择GPU1。唯一GDB/Isaac启动1/1运行141.872s，环境在观测135.312s返回，实际import为overlay SHA23122a…，真实stacks0字节，无native signal；原_guard在585行前exit86，control/PPO均0。运行GPU总used峰值5383MiB、util峰值13%，无资源阈值/OOM/可访问Xid。结果证明本次最小干预下初始化可返回，不证明历史唯一根因或长期训练稳定；本包停止，不训练。详见 `H2S2R_TRACEBACK_INTERVENTION_20260913/RESULTS.md`。

**Historical implementation checkpoint:** `SUPPORT_LOCAL_READY / BLOCKED_DEPLOY_APPROVAL / ISAAC 0/1`。该上传阻塞随后由用户明确授权解除，未消耗Isaac；保留此记录解释准备成本，不再是当前状态。

状态REVIEWER_RELEASED_FOR_USER_DISPATCH。原native包1/1 CLOSED，不重写其FAIL；新包独立最多一次Isaac，由用户转发prompt后执行。reviewer只做离线审阅。

已直接核对shared_gpu_resume_01的gdb_native.txt：实际SIGSEGV在CPython3.11.15 dump_frame，Thread2；GDB随后因info signals SIGSEGV SIGABRT错误终止采集。核对sitecustomize.py的sys.settrace(_guard)及V13 resume_observation.py的dump_traceback_later(30,repeat=True)。这是故障位置证据，非唯一根因。旧非GDB崩溃没有本次_guard，不能把该护栏当全部历史故障解释。16环境成功也曾有周期dump，故不能宣称所有dump必崩。

工程决定：下一干预仅停用周期Python traceback采样，保留原初始化护栏、普通事件日志、资源计数、GDB和全部科学配置。无需先取得唯一根因才尝试移除非必要诊断功能。不改PCA/22D、FABRICS、奖励、物理或解释器。

输入：V13冻结32身份及实际shared_gpu_resume_01启动命令、最新已批准资源门。原目录不覆盖。隔离支持副本内仅给resume_observation.py的周期dump调度加显式禁用开关，默认保持历史行为；新诊断命令关闭该开关。保留faulthandler故障处理与SIGUSR1注册但不得主动请求dump；不得由其他init_trace路径重启周期定时器。运行前CPU验证实际加载支持路径和唯一差异，不能靠sitecustomize早期cancel后又被构造函数重新注册。保持原_guard不变以减少变量，成功返回仍在策略前退出。

并行完成采集支持修复：GDB删除不必要的双信号info命令，信号捕获后优先bt和寄存器；采集单项报错不能阻止其余项。launcher接受Thread ... received signal及Program received signal，不只靠日志判返回码。CPU真实SIGSEGV使用实际将部署的整份GDB脚本验证输出原生栈/寄存器/库/全线程，并测试单项故障后仍有后续输出；不能仅测另一份简化脚本。

预算：准备≤40min；Isaac最多1次、初始化≤15min、捕获清理≤5min；分析交付≤20min，整包≤80min。准备/资源阻塞不消耗Isaac次数、不重复已通过准备。仅操作本任务进程，不提权、不改系统配置或自行安装；沿用已批准共享资源门，不另猜阈值。不训练或评测，不自动二次运行。

判据：环境返回且前置护栏退出 => INIT_RETURNED_WITH_PERIODIC_DUMP_DISABLED，支持移除该非必要采样作为启动稳定性工程缓解，但不证明唯一根因或长期稳定；reviewer下一包优先恢复既定256训练，不再独立做初始化扫描。仍崩溃 => 保存完整原生栈，交reviewer/原生运行时支持，不继续删诊断/缩规模。超时 => 中断保存原生栈后停止，不判修复。采集不完整 => 如实交付，不追加。任何策略control/优化非零均为越界，立即停止保留证据。

交付D:/UCBP/reports/H2S2R_TRACEBACK_INTERVENTION_20260913/下RESULTS、CPU实际脚本验证、diff/加载身份、初始化计数、native文本、decision/costs、远端core索引SHA。原始转储留远端。无策略或可用渲染样本时无视频如实说明，不新增物理录像探针。保持左杯与Clean3独立。
