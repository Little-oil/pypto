# L2 缓存提示（L2 Cache Hints）

`l2_hint` 是前端控制 PTO-ISA L2 缓存的统一接口。load 使用
`pl.LoadL2Hint`，store 使用 `pl.StoreL2Hint`。枚举值保留 ISA 编码；
其中 `NotAllocClean` 在 load 与 store 中的编码不同。
PyPTO 在所有架构上保留调用方选择的精确枚举，不会将某一种 `NotAlloc*` 替换成另一种。

| LoadL2Hint | Encoding | StoreL2Hint | Encoding |
| ---------- | -------- | ----------- | -------- |
| NormalFirstVictim | 0 | NormalFirstVictim | 0 |
| NormalLastVictim | 1 | NormalLastVictim | 1 |
| NormalPersistent | 2 | NormalPersistent | 2 |
| NotAllocKeep | 4 | — | — |
| NotAllocClean | 5 | NotAllocClean | 4 |
| NotAllocDrop | 6 | — | — |

Normal 系列请求正常分配，并指定不同的淘汰优先级；`NormalPersistent` 请求保留。
`NotAlloc*` 请求不分配新缓存行，在支持的架构上进一步区分已有缓存行的处理方式。
hint 不改变张量的数值，也不保证性能收益。

## 单次搬运

```python
x_tile = pl.load(x, [0, 0], [32, 32], l2_hint=pl.LoadL2Hint.NormalPersistent)
out = pl.store(x_tile, [0, 0], out, l2_hint=pl.StoreL2Hint.NotAllocClean)
```

省略 `l2_hint` 或传入 `None` 表示未指定。显式的 `NormalFirstVictim` 即使编码为 0
也会保留，因为它可以覆盖作用域声明。传入方向错误的枚举或不支持的整数编码会报错。

## 作用域声明与隐式搬运

张量运算隐式生成 load/store 时，使用
`pl.set_l2_hint(tensor, load=..., store=...)`。可指定一个方向，也可同时指定两个方向；
两个方向相互独立。单次访问上显式指定的 hint 始终优先。

```python
import pypto.language as pl

@pl.jit
def matmul(
    x: pl.Tensor[[64, 128], pl.FP16],
    w: pl.Tensor[[128, 64], pl.FP16],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.set_l2_hint(w, load=pl.LoadL2Hint.NotAllocClean)
        pl.set_l2_hint(out, store=pl.StoreL2Hint.NormalLastVictim)
        result = pl.matmul(x, w, out_dtype=pl.FP32)
        out = pl.assemble(out, result, [0, 0])
    return out
```

声明必须是 `pl.at` 或内联 `pl.spmd` 主体直接包含的独立语句。它对整个作用域生效，
不取决于在顶层语句中的位置。张量必须在进入作用域前已绑定，且被主体实际捕获。
嵌套在 `if` 或 `for` 内，或者写在作用域外的声明会被拒绝。相同声明去重；同一张量
同一方向的不同 hint 会报错。store 声明要求作用域实际写入该张量。

声明按变量身份（Var identity）解析捕获的绑定。张量降级生成的 load/store 会继承 hint，
包括 GM 写入目标的重新绑定与循环携带变量。若别名可能指向不同参数，仅当所有可能参数
都声明了相同 hint 时才继承。slice 本身不接受 hint：提示应属于实际搬运数据的操作。

作用域对自己会写入的张量声明 load `NotAlloc*` 会被拒绝。对于 A2/A3 地址别名，
调用方还需保证作用域外没有并发的带缓存写入修改同一段字节；编译器无法证明该约束。
非设备层级作用域的声明会作为元数据传递，但不会配置另行调用的设备内核中的搬运。

## 架构行为

| 架构 | Normal load hint | NotAlloc load hint | Store hint |
| ---- | ---------------- | ------------------ | ---------- |
| A5 | 精确 ISA 模板枚举，选择 `l2CacheCtl` | 精确 ISA 模板枚举，选择 `l2CacheCtl` | 精确 ISA 模板枚举，选择 `l2CacheCtl` |
| A2/A3 | 精确 ISA 模板枚举；0/1/2 均为相同的正常读取行为 | 精确 ISA 模板枚举；4/5/6 均使用相同的 ISA nocache 地址别名 | 精确 ISA 模板枚举；无独立硬件缓存控制效果 |

例如，选择 `pl.LoadL2Hint.NotAllocDrop` 后，A2/A3 与 A5 都生成相同的枚举：

```cpp
TLOAD<pto::TLoadL2Hint::NotAllocDrop>(dst, src);
```

在 A2/A3 上，ISA 对三种 `NotAlloc*` load hint 均将
`g_opL2CacheHintCfg.l2Cacheoffset` 加到源地址。运行时必须在执行前为该全局配置提供有效
存储并完成初始化；ISA 读取偏移，不查询驱动。设备偏移来自 `rtGetL2CacheOffset` 暴露的
驱动查询接口，CANN 原生 ELF 加载器在初始化 L2 配置时直接调用相同的驱动查询。
全局配置与显式查询是传递同一个设备相关偏移的不同方式。
PyPTO 不对源地址加偏移，也不传递合成的 offset 参数。A5 则将枚举传给 `l2CacheCtl`，
不修改源地址。

当前 simpler 内核加载器只提取 `.text`，尚未接通 ISA 全局配置的存储、重定位和初始化。
因此 A2/A3 的 `NotAlloc*` load 支持代码生成与编译，但在
[simpler #2548](https://github.com/hw-native-sys/simpler/issues/2548) 解决前不支持板上执行。
全局配置所在的数据段未加载时，源码中的零初始化不能作为安全回退。
CPU 模拟也不能验证设备上的缓存行为。

显式 hint 使用普通的 `pto.tload` / `pto.tstore`，并由 `emitc.verbatim` 标记包围。
PyPTO 的 C++ 最终处理器将标记转换为选定的 PTO-ISA 模板参数，保留 store phase、atomic、
quantization 和 ReLU 参数。PTOAS 仍能分析搬运操作的内存效果，用于同步与内存规划。
两种内存规划器均支持。无需修改 PTOAS 源码，固定版本为 v0.65。

调试重编译使用同一个幂等最终处理器。`skip_ptoas=True` 保存的 PTO 仍包含标记：单独执行
PTOAS 得到的中间 C++ 还需 PyPTO 最终处理。畸形或有歧义的标记搬运会导致编译失败。
Buffer 降级在其搬运配方支持显式 hint 前，会拒绝这些 hint。

## 编译链路

```text
pl.set_l2_hint(t, load=..., store=...)        parser consumes declaration
  -> Scope attrs load_l2_hint_vars / store_l2_hint_vars  (Var, encoding)
  -> Function attrs load_l2_hint / store_l2_hint         (param index, encoding)
  -> tile.load / tile.store kwarg l2_hint                exact encoding
  -> selected PTO-ISA template enum on A2/A3 and A5       exact caller choice
```

[OutlineIncoreScopes](../passes/09-outline_incore_scopes.md) 消费作用域元数据，并按参数索引
排序声明。[ConvertTensorToTileOps](../passes/11-convert_tensor_to_tile_ops.md) 将提示写入
搬运操作的 kwarg，并在后续 pass 改变参数列表前擦除函数属性。解析、打印、结构比较与
序列化保留两个方向以及显式的零值 hint。

## 迁移已删除的接口

`CachePolicy`、`pl.load(cache=...)` 和 `pl.set_cache_policy` 已删除。
显式的 `CachePolicy.DEFAULT` 覆盖改为 `l2_hint=pl.LoadL2Hint.NormalFirstVictim`；
无需覆盖时可省略参数。作用域覆盖使用
`pl.set_l2_hint(t, load=pl.LoadL2Hint.NormalFirstVictim)`。

迁移旧 BYPASS 读取时，应根据目标架构和工作负载选择 `NotAllocKeep`、`NotAllocClean`
或 `NotAllocDrop`，再将用户选择传给 `l2_hint` 或 `pl.set_l2_hint` 的 `load` 参数。
不会自动映射到编码 5。在 A2/A3 上，接通上述运行时依赖后，三者均选择相同的 ISA 地址
别名行为。在 A5 上，三者对应不同的 ISA 模式；旧 BYPASS 没有指定其中一种模式，
因此不能推导出唯一的替代值。store hint 独立选择，使用 `pl.StoreL2Hint`。
