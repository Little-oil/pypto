<!-- markdownlint-disable MD013 MD060 -->
# PTOAS Op 添加计划

本文记录尚未完成同名系统测试（system test，ST）验收的 PTOAS op 实施路线及批次验收事实。当前接口与覆盖状态以 [PTOAS Op 状态矩阵](ptoas-op-status.md) 为唯一来源；“代码或 ST 文件已添加”不等于“已完成”。只有生成的 `.pto` 包含最新 PTOAS canonical op 名且至少一个支持平台上的真机 ST 通过后，状态矩阵中的 ST 才能改为 `✅`；尚未真机验证的其他架构必须继续在备注和 G3 中记录。

接口名称以最新 PTOAS `main` `d852dd2dba3e5bf7a69ce8324eb88afc336e8a33` 为准。本计划的原始范围为 67 个无同名真机 ST 的 op；B01 完成 A2/A3 真机验收后，状态矩阵当前剩余 62 个。B01 仍保留在批次表和验收记录中，但不再计入下方“无 ST op 分类”。不适合独立 ST 的内部或 legacy op 不在计划内。每个实现批次仍须先用 PyPTO 当前固定的 PTOAS `v0.48` 做最小组装验证；若接口只存在于更新版本，应先单独升级工具链，不把版本升级与 op 实现混在同一个 PR。

## 无 ST op 分类

分类沿用状态矩阵中的 op 类别；下表 62 项与状态矩阵中 `ST测试` 或 `distributed ST测试` 为 `❌` 的行严格一一对应。

| 类别 | 数量 | op |
|---|---:|---|
| DMA 数据搬运 | 2 | `pto.tprefetch`, `pto.tprefetch_async` |
| 矩阵计算 | 9 | `pto.tmatmul.mx`, `pto.tmatmul.mx.acc`, `pto.tmatmul.mx.bias`, `pto.tgemv`, `pto.tgemv.acc`, `pto.tgemv.bias`, `pto.tgemv.mx`, `pto.tgemv.mx.acc`, `pto.tgemv.mx.bias` |
| 向量算术与数学 | 16 | `pto.trem`, `pto.tpartargmax`, `pto.tpartargmin`, `pto.tprelu`, `pto.taxpy`, `pto.trems`, `pto.taddc`, `pto.tsubc`, `pto.taddsc`, `pto.tsubsc`, `pto.ttri`, `pto.taddrelu`, `pto.tfmod`, `pto.tfmods`, `pto.tpow`, `pto.tpows` |
| 归约 | 1 | `pto.thistogram` |
| 比较与选择 | 1 | `pto.tsels` |
| 位运算 | 10 | `pto.tand`, `pto.tor`, `pto.txor`, `pto.tshl`, `pto.tshr`, `pto.tands`, `pto.tors`, `pto.txors`, `pto.tshls`, `pto.tshrs` |
| 数据重排 | 6 | `pto.tconcatidx`, `pto.tgatherb`, `pto.mgather`, `pto.tfillpad_inplace`, `pto.textract_fp`, `pto.tinsert_fp` |
| MX 量化 | 6 | `pto.tget_scale_addr`, `pto.tmov.fp`, `pto.tquant`, `pto.tquant.mx`, `pto.tstore_fp`, `pto.tdequant` |
| 调试 | 1 | `pto.tprint` |
| 通信（comm） | 10 | `pto.comm.build_async_session`, `pto.comm.tput_async`, `pto.comm.tget_async`, `pto.comm.wait_async_event`, `pto.comm.test_async_event`, `pto.comm.ttest`, `pto.comm.tbroadcast`, `pto.comm.tgather`, `pto.comm.tscatter`, `pto.comm.treduce` |
| **总计** | **62** |  |

## 分期与批次总览

原始计划分为 **7 个阶段、22 个实现批次，共 67 个 op**；B01 的 5 个 op 已完成 A2/A3 真机验收，当前剩余 **21 个待实施批次、62 个 op**。原则上一个批次对应一个可独立评审和回滚的 PR；批次编号表示建议优先级，不表示没有依赖关系的批次必须串行执行。若某批次需要升级 PTOAS，应先提交独立的工具链升级 PR，再继续该批次。

| 阶段 | 批次 | 数量 | 内容 | 主要依赖 |
|---|---|---:|---|---|
| P1：复用历史资产 | B01–B04 | 13 | 已有链路补 ST、选择类、gather/tri、GEMV | 当前 pinned PTOAS；历史实现只手工移植 |
| P2：复验现有链路 | B05–B08 | 18 | remainder、carry、逻辑位运算、shift | 每批先做最新 PTOAS assembly probe |
| P3：通用单卡新增 | B09–B12 | 10 | partial/reduction、math/fused、rearrangement、debug | 完整 IR/API/codegen 链路 |
| P4：FP 与 quant 基础 | B13–B15 | 8 | FP 搬运、普通量化、MX scaling/quant | B15 依赖 B13–B14 |
| P5：A5 MX 矩阵族 | B16–B17 | 6 | MX matmul 与 MX GEMV | 依赖 B15 |
| P6：DMA 预取 | B18–B19 | 2 | 同步预取与异步预取 | B19 依赖 B18 及 async context/session |
| P7：多卡通信 | B20–B22 | 10 | test、exact collective、async comm | 确定性双 rank 测试环境 |
| **总计** | **22 批** | **67** |  |  |

## 详细批次计划

下表的“专项实现与 ST 内容”只列每批特有要求；所有 op 还必须满足本文后面的统一“ST 覆盖要求”。每个 op 在下表中只出现一次。

| 批次 | 数量 | 本批包含的 op | 实现内容 | 专项实现与 ST 内容 |
|---|---:|---|---|---|
| B01：已有链路基础 ST（A2/A3 可执行契约已完成） | 5 | `pto.tdiv`, `pto.tsubs`, `pto.tlog`, `pto.trowmin`, `pto.trowexpandadd` | 补齐 `precisionType`、可选 `tmp`、packed carrier 等直接接口缺口，并新增同名 ST | 覆盖 A2/A3 可执行契约的全部 dtype/签名、`tmp` 规则、边界 shape 和 `valid_shape`；确认最终 `.pto` 为精确同名 op；A5 独立保留待验收项 |
| B02：选择与 PReLU | 2 | `pto.tsels`, `pto.tprelu` | 复核 `tsels` 的选择语义和 `tprelu` 的 `(src, slope, tmp)` 三参数链路，补齐 UT/ST | 覆盖选择条件与 scalar 类型分支、PReLU slope/dtype/`tmp`/alias 分支及非完整 `valid_shape` |
| B03：Tri 与 Gather | 3 | `pto.ttri`, `pto.tgatherb`, `pto.mgather` | 从历史资产移植 IR/API/codegen；把旧 `pto.tmgather` 修正为 `pto.mgather` | 覆盖 `ttri` 上下三角与 diagonal、`tgatherb` byte-offset、`mgather` index/coalesce/memory-space、越界约束及 `valid_shape` |
| B04：普通 GEMV | 3 | `pto.tgemv`, `pto.tgemv.acc`, `pto.tgemv.bias` | 适配当前 memory/layout pass，复验已有 frontend/codegen，分别建立 base/acc/bias ST | 覆盖 1-row Mat、窄 K、BF16/FP16、累加与 bias、支持的 layout/shape/`valid_shape` |
| B05：Remainder 家族 | 4 | `pto.trem`, `pto.trems`, `pto.tfmod`, `pto.tfmods` | 恢复历史 probe，核对 tile-tile/tile-scalar 和整数/浮点 remainder 的精确发射 | 覆盖各 dtype、scalar/tile 形式、符号组合、`tmp` 与 `valid_shape`；除零等未定义输入不得伪造期望值 |
| B06：Carry 家族 | 4 | `pto.taddc`, `pto.tsubc`, `pto.taddsc`, `pto.tsubsc` | 复核 carry 输入输出、算式、operand 顺序以及 scalar 变体，修复 frontend/codegen 不一致 | 覆盖 carry=0/1、add/sub、tile/scalar、溢出/借位、alias 与 `valid_shape` |
| B07：逻辑位运算 | 6 | `pto.tand`, `pto.tor`, `pto.txor`, `pto.tands`, `pto.tors`, `pto.txors` | 复验当前链路与 pinned PTOAS，统一 tile-tile 和 tile-scalar codegen | 覆盖全部支持整数宽度、零/全一/交错 bit pattern、scalar 编码、`tmp` 和 `valid_shape` |
| B08：Shift 家族 | 4 | `pto.tshl`, `pto.tshr`, `pto.tshls`, `pto.tshrs` | 复验 tile/scalar shift operand 与精确 PTOAS 签名，处理 signed/unsigned 差异 | 覆盖左/右移、signed/unsigned、0 和合法边界 shift count、scalar/tile 形式及 `valid_shape` |
| B09：Partial Arg 与 Histogram | 3 | `pto.tpartargmax`, `pto.tpartargmin`, `pto.thistogram` | 新增 C++ IR、类型推导、Python tile API、codegen、UT/ST | 覆盖 value/index 输出、max/min、相等值 tie、partial combine、histogram bin/边界/计数及 `valid_shape` |
| B10：数学与融合算子 | 4 | `pto.taxpy`, `pto.taddrelu`, `pto.tpow`, `pto.tpows` | 新增完整链路；`taddrelu` 精确发射融合指令；`tpow/tpows` 按 dtype 决定是否需要 `tmp` | 覆盖 AXPY scalar、ReLU 正负/零边界、pow tile/scalar exponent、整数/浮点和 high-precision `tmp` 分支及 `valid_shape` |
| B11：Concat 与 In-place Fillpad | 2 | `pto.tconcatidx`, `pto.tfillpad_inplace` | 为 `tconcatidx` 新增完整链路；把现有 `tfillpad` 错误发射修为 `tfillpad_inplace` | 覆盖 concat index/axis/边界、所有 fillpad mode、in-place alias、padding 边界和 `valid_shape` |
| B12：调试输出 | 1 | `pto.tprint` | 在已有 backend hook 上补 IR/Python API、格式属性、codegen UT 与可观察输出 ST | 覆盖全部支持 dtype/format、完整与非完整 `valid_shape`；ST 必须验证输出内容而非只验证不崩溃 |
| B13：FP 提取、插入与搬运 | 4 | `pto.textract_fp`, `pto.tinsert_fp`, `pto.tmov.fp`, `pto.tstore_fp` | 建立 FP tile/type/memory-space 约束及完整链路；把旧 `pto.tstore.fp` 修正为 `pto.tstore_fp` | 覆盖 extract/insert 的 FP 字段分支、move/store 方向与目标空间、支持 dtype/layout、alias 和 `valid_shape` |
| B14：普通量化与反量化 | 2 | `pto.tquant`, `pto.tdequant` | 新增 quant/dequant IR/API/codegen；处理 scale、offset、round/saturate 与平台 `tmp` 差异 | 覆盖 SYM→I8、ASYM→UI8、offset 有无、A2/A3 与 A5 签名、饱和边界、rounding 和 `valid_shape` |
| B15：MX Scaling 与量化 | 2 | `pto.tget_scale_addr`, `pto.tquant.mx` | 在 B13–B14 基础上实现 A5 scaling tile/address 和 MX quant 链路 | 覆盖 FP8/FP4、scaling tile 的 loc/shape/layout、scale address、量化边界及 `valid_shape` |
| B16：MX Matmul | 3 | `pto.tmatmul.mx`, `pto.tmatmul.mx.acc`, `pto.tmatmul.mx.bias` | 基于已有 backend hook 增加 IR/Python API、类型推导、layout/memory pass 和三种精确发射 | 依赖 B15；覆盖 base/acc/bias、FP8/FP4 输入、F32 输出、scaling tile、K/shape/layout 边界及 `valid_shape` |
| B17：MX GEMV | 3 | `pto.tgemv.mx`, `pto.tgemv.mx.acc`, `pto.tgemv.mx.bias` | 在 B15/B16 的 MX 公共基础上新增 GEMV base/acc/bias 完整链路 | 覆盖 MX GEMV 的向量/矩阵 layout、窄 K、acc/bias、scaling tile、FP8/FP4 与 `valid_shape` |
| B18：同步 Prefetch | 1 | `pto.tprefetch` | 新增 tile IR、Python API、memory/layout 约束、codegen 和 ST | 覆盖支持的 GM→目标层级、地址/offset/alignment、shape/layout、边界与适用的 `valid_shape` |
| B19：异步 Prefetch | 1 | `pto.tprefetch_async` | 在 B18 基础上集成 `make_prefetch_async_context` 和 `get_prefetch_async_session`，实现 session 生命周期与精确 codegen | 覆盖发起/完成顺序、多 session、边界地址以及当前 flat-contiguous logical-1D 限制；适用时覆盖 `valid_shape` |
| B20：通信 Test | 1 | `pto.comm.ttest` | 从历史资产移植到当前 distributed IR/backend，保持 PyPTO enum ABI 并显式映射比较模式 | 确定性双 rank 覆盖全部比较 mode、hit/miss、不同数据宽度和边界值 |
| B21：Exact Collectives | 4 | `pto.comm.tbroadcast`, `pto.comm.tgather`, `pto.comm.tscatter`, `pto.comm.treduce` | 增加或修正直接 collective lowering，保证最终生成同名 PTO op，不以 `tput/tget` 分解代替 | 双 rank 覆盖 root/non-root、输入输出角色、支持 dtype/reduce mode、完整与非完整 `valid_shape` |
| B22：Async Comm | 5 | `pto.comm.build_async_session`, `pto.comm.tput_async`, `pto.comm.tget_async`, `pto.comm.wait_async_event`, `pto.comm.test_async_event` | 建立 AsyncSession/Event 类型、生命周期、distributed IR/API 和五个 exact codegen 路径 | 依赖确定性双 rank 环境；覆盖 put/get、wait/test、event 未完成/完成、session/event 匹配和 flat-contiguous logical-1D 限制 |
| **总计** | **67** | **22 个批次** |  |  |

## B01 实施与验收记录

B01 的源码、UT 和五份同名 ST 已落盘。当前矩阵的 A2/A3 真机任务在 device 5 上分三批执行：`task_20260727_012033_6845730792`（32/32）、`task_20260727_012037_7177622613`（32/32）和 `task_20260727_012039_7236615809`（1/1），合计 65/65 通过，因此状态矩阵的五项已改为 `✅`。A5 真机尚未执行，继续作为跨平台 G3 待办；ST 不使用 skip 或 xfail 隐藏接口差异。

| op | 已添加的接口与 ST 覆盖 | 明确排除或待解决项 | ST 文件 |
|---|---|---|---|
| `pto.tdiv` | tile 与 tensor lowering；A2/A3 的 `f16/f32`，A5 的 `i16/i32/f16/f32`；浮点 default/high precision；full、row tail、column tail、row+column tail及小边界 shape；`f16/f32` high precision 均与 combined tail 交叉 | 除零是 target-defined。最新 PTOAS verifier 漏检 A5 整数 high precision，但 ISA 文档、A5 template 和 PTOAS 自有 ST 都只支持 `f16/f32`，PyPTO 因此拒绝该组合。verifier 还只比较 valid shape，而 PTO-ISA `TDIV_IMPL` 要求三块 tile 的完整物理类型一致，PyPTO 按可执行接口要求同 physical shape；对动态 valid shape 则要求两侧可证明相等，避免 verifier 放行的未知关系在运行时越界读取 | `tests/st/runtime/ops/test_div.py` |
| `pto.tsubs` | tile 与 tensor lowering；A2/A3 的 `i16/i32/f16/f32`，A5 另含 `i8/bf16`；可执行路径支持的 `i8/i16/i32/f16/f32/bf16` scalar、负/零/正值及双向 mixed dtype 代表；full 与三种 tail，mixed tile case 含 combined tail | PTOAS TableGen 接受任意 signless integer/float scalar，但 PyPTO 只公开当前 codegen 可表示且 ST 矩阵覆盖的 scalar dtype。PTOAS verifier 虽接受非 row-major，CPU stub 和 A2/A3 device 路径均存在已知风险；PyPTO 因此在 backend layout pass 中统一修复为 row-major，不公开不安全的原始布局执行路径。A5 device 尚待板测 | `tests/st/runtime/ops/test_subs.py` |
| `pto.tlog` | tile 与 tensor lowering；`f16/f32`；default/high precision；full、三种 tail及小边界 shape；`f16/f32` high precision 均与 combined tail 交叉；输入使用正有限值 | 对 `x <= 0`、NaN/Inf，manual 未给出可移植 golden，不写伪期望；A2/A3 接受 high-precision 属性但 pinned 实现不提升精度 | `tests/st/runtime/ops/test_log.py` |
| `pto.trowmin` | tile 与 tensor lowering；`i16/i32/f16/f32`；必选 `tmp` 的 exact/oversized 安全形式；full、三种 tail、valid `1×1` 最小非零边界及 oversized tmp + combined tail；首/中/尾及重复最小值，invalid tail 放置更小 poison | PyPTO 主动保持 DN 输出及 A2/A3 安全的 same-dtype、同 rank、足量 workspace 子集；legacy ND 输出和 A5 relaxed placeholder 暂不作为公共运行时承诺 | `tests/st/runtime/ops/test_row_min.py` |
| `pto.trowexpandadd` | tile 与 tensor lowering；A2/A3 的 `i16/i32/f16/f32`，A5 另含 `i8`；有/无 `tmp`；DN `[M,1]` 与 32-byte packed carrier；full 与三种 tail；packed/no-tmp + combined tail，以及 A5 packed/tmp + combined tail；A5 小型异 dtype `tmp` placeholder | DN carrier 执行 manual 的每行 scalar 语义；row-major packed overload 在 A2/A3 真机与 A5 simulator 上均实际按 32-byte lane block 周期扩展，ST 用非重复 lane 明确验证该 raw overload，避免把它误当成 lane 0 scalar broadcast。A2/A3 的 packed + `tmp` 被 ISA 拒绝，仅 A5 覆盖。PTOAS CPU simulator stub 未实例化 `i8`，因此 5 个 A5 `i8` case 只在 `a5` device 目标保留 | `tests/st/runtime/ops/test_row_expand_add.py` |

| 门槛 | B01 当前结果 |
|---|---|
| G0 接口与版本 | 已按最新 PTOAS canonical op/签名核对；五项均能通过 PyPTO pinned PTOAS 的组装路径 |
| G1 IR/API | 已补齐 precision、可选 `tmp`、packed carrier、按可执行接口收紧的 dtype/physical-shape/layout/valid-shape 约束、mixed scalar/cast lowering、旧位置参数兼容及负向 UT |
| G2 exact codegen | exact-op UT 通过；A2/A3 目标 65/65、A5 目标 81/81 个 codegen-only case 通过 PTOAS 组装 |
| G3 真机 ST | **A2/A3 已完成**：device 5 真机分三批执行，任务 `task_20260727_012033_6845730792`（32/32）、`task_20260727_012037_7177622613`（32/32）、`task_20260727_012039_7236615809`（1/1），合计 65/65 通过；PTOAS `v0.48`，runtime 固定 PTO-ISA `83d01313`。A5 真机尚未执行；`a2a3sim` 65/65、`a5sim` 76/76 通过，A5 `i8` 的 5 个 `trowexpandadd` case 因 CPU simulator stub 限制仅完成 device-target 组装 |
| G5 收口 | A2/A3 同名真机证据已满足状态矩阵 `✅`；A5 真机、CI 与合入主线仍待完成 |

一个 PR 不得同时承担工具链升级、公共 API 设计和多个无关批次。只有存在直接依赖且单独提交无法构建或验证时，才能合并相邻批次，并须在 PR 描述中说明原因。

## 可复用历史资产

| 范围 | 来源 | 移植规则 |
|---|---|---|
| `tdiv/tsubs/tlog/trowmin/trowexpandadd/tsels` | `e6ae95b2` | 只移植这 6 项；`tadds`、`texpands`、`tmrgsort` 已在主线有 ST，不重复带回 |
| `tgatherb`, `ttri` | `feat-add-ptoas-ops-batch5`，核心提交 `64c6b160`, `ee7b686a` | 分成两个 PR，保留 byte-offset 与 diagonal 语义 |
| `mgather` | `issue-1807-loop-local-gather`，核心提交 `711e45c7`, `7ec0e5f2` | 把旧 `pto.tmgather` 映射修为 `pto.mgather`，去掉 skip 后按当前工具链复验 |
| GEMV 三项 | `fix-pr1823-op-isa-gaps` / `5c106339` | 适配当前 memory/layout 实现，覆盖 1-row Mat、窄 K、BF16/FP16 |
| `tprelu` 与 carry | `fix-pr1823-pypto-codegen-ops` 未提交 worktree | `tprelu` 保持 `(src, slope, tmp)` 三参数；验证 carry 的真实算式和 alias 规则 |
| REM/bitwise probes | `dd6b723c`, `6f88b9fe` | 恢复测试 hunk，忽略已有覆盖的 `tnot`；即使当前 ISA 含相关修复也必须重跑 |
| `pto.comm.ttest` | `feat-class-f-async-comm-ops` 未提交 worktree | 手工移植到 `pto_ops_distributed.cpp`；保留现有 PyPTO enum ABI，显式映射比较模式 |

旧分支的 backend 代码基于单体 `pto_ops_common.cpp`，不得整分支 cherry-pick；应把相关 hunk 手工适配到当前 `pto_ops_elementwise.cpp`、`pto_ops_datamove.cpp`、`pto_ops_memory.cpp` 或 `pto_ops_distributed.cpp`。`pto.trandom` 已通过 `ff8028d0` 合入，不属于本计划。

## 受影响代码层检查表

| 层 | 典型文件 | 何时必改 |
|---|---|---|
| PTOAS 兼容性 | `toolchain/versions.env`；最小 `.pto` probe | 每个 op 都先验证；需要 bump 时另开 PR |
| C++ tile IR / type inference | `src/ir/op/tile_ops/*.cpp`, `src/ir/op/type_inference.cpp` | 新 op、签名/shape/dtype/memory-space 变化 |
| Tensor API / lowering | `src/ir/op/tensor_ops/*.cpp`, `src/ir/transforms/op_conversion_registry.cpp` | 只有具备稳定 tensor 语义的 op 才增加 |
| Distributed IR | `src/ir/op/distributed/*.cpp` | comm session/event、collective 或 system op |
| Python IR / DSL / exports | `python/pypto/ir/op/{tile,tensor}_ops.py`, `python/pypto/language/op/{tile,tensor,unified}_ops.py` 及相关 `__init__.py` | 公共 API 或签名变化 |
| PTO codegen | `src/backend/common/pto_ops_{elementwise,datamove,memory,distributed}.cpp`，必要时 `pto_ops_shared.cpp` | 精确 op 名、operand 顺序、属性、enum、in-place outs、tmp 规则 |
| Layout / memory passes | `src/ir/transforms/infer_tile_memory_space_pass.cpp`, `resolve_backend_op_layouts_pass.cpp` 等 | GEMV、MX、FP/scaling、alias/layout 特殊约束 |
| UT | `tests/ut/ir/operators/`, `tests/ut/ir/transforms/`, `tests/ut/language/`, `tests/ut/codegen/` | IR/DSL/type/conversion/codegen 均需对应断言 |
| 真机 ST | `tests/st/runtime/ops/`, `tests/st/distributed/` | 数值或可观察行为验证，并确认最终 `.pto` 含精确同名 op |
| 文档 | `docs/zh-cn/dev/ir/05-operators.md`、中文 distributed 文档、`docs/zh-cn/dev/ptoas-op-status.md` | PR 合入后同步中文 API 文档与状态证据 |

## ST 覆盖要求

每个新增 op 的 ST 必须覆盖最新 PTOAS 文档定义的**全部语义分支**，不能只选一个典型 case。提交时应逐项列出并覆盖该 op 适用的全部 overload/签名、dtype 族、可选 operand、属性或 mode、`tmp` 有无、in-place/alias 规则、memory space、layout 以及架构差异；PTOAS 明确不支持的组合须附 manual 或 verifier 依据，不能以 skip/xpass 代替覆盖。

凡 operand 或 result 使用 tile 且接口允许 `valid_shape`，ST 还必须包含 `valid_shape` case：

- 覆盖 `valid_shape == shape` 的完整 tile，以及 `valid_shape < shape` 的非完整 tile。
- 若行、列均可独立裁剪，分别覆盖 row tail、column tail 和 row+column tail。
- reference 只在 valid region 内比较数值；若 PTOAS 对 invalid region 有明确行为，也必须验证该行为。
- 若某 op 确实不适用或不支持 `valid_shape`，须在测试或 PR 中写明依据，不得直接遗漏。

上述 case matrix 全部通过后，该 op 才能在状态矩阵中标记为有 ST。

## 固定验收门槛

- [ ] **G0 接口与版本**：核对最新 PTOAS canonical name；用 PyPTO pinned PTOAS 组装最小 `.pto`；记录支持架构、dtype、shape、layout、memory space 与 scratch 条件。
- [ ] **G1 IR/API**：C++ IR、Python IR/DSL、导出和类型推导一致；非法 arity/type/layout 有负向 UT；只有存在清晰高层语义时才增加 tensor API。
- [ ] **G2 exact codegen**：codegen UT 断言精确 `pto.*` 名、operand/outs 顺序、属性与 enum；输出能够通过 pinned PTOAS assembler。类似名称或由其他 op 间接实现不算通过。
- [ ] **G3 真机 ST**：至少在一个支持平台执行真机同名 ST；按“ST 覆盖要求”完成该 op 在该平台适用的全部语义分支和 `valid_shape` case，比较可信 reference。其他声称支持但尚未真机验证的平台必须显式保留为待办；任何必需 case 缺失或依赖 skip/xpass 均不算该平台通过。
- [ ] **G4 专项集成**：MX/quant 使用 A5 scaling 约束；async prefetch 使用 context/session；comm 使用确定性双 rank hit/miss、session/event 一致性和 flat-contiguous-1D 限制。
- [ ] **G5 收口**：相关 UT/ST 全绿；至少一个支持平台具备真机同名执行证据后，可在本批变更中把状态矩阵 `❌` 改为 `✅` 并附证据，尚未验证的平台继续写入备注和 G3 待办。若 G0/G2/G3 暴露上游缺陷且没有任何平台通过，保留 `❌`，记录版本、平台和最小复现，不以“已有 frontend”代替完成。

## 高风险接口备注

- `tpow/tpows`：浮点或 high-precision 路径需要 `tmp`；纯整数路径必须省略 `tmp`。
- `tquant`：F32 source；SYM→I8 不带 offset，ASYM→UI8 需要 offset；A2/A3 可用 tmp 合成，A5 不带 tmp。`tquant.mx` 单独按 A5 验证。
- `tget_scale_addr` 与 MX 矩阵族：A5-only；scaling tile 的 loc、shape、valid shape、layout 必须满足 PTOAS 约束。
- `tmov.fp`：legacy 兼容项；新代码优先使用 `tmov` 的 FP operand 形式。
- async comm：session/event 必须成对匹配，put/get 当前只承诺 flat contiguous logical-1D。
- exact collectives：现有高层 collective 若最终只生成 `tput/tget`，不能作为 `tbroadcast/tgather/tscatter/treduce` 的完成证据。
