# L2 Cache Hints

`l2_hint` is the single frontend interface for PTO-ISA L2 cache control. Use
`pl.LoadL2Hint` for loads and `pl.StoreL2Hint` for stores. Their integer values
are the ISA encodings; notably, `NotAllocClean` has different load/store values.

| LoadL2Hint | Encoding | StoreL2Hint | Encoding |
| ---------- | -------- | ----------- | -------- |
| NormalFirstVictim | 0 | NormalFirstVictim | 0 |
| NormalLastVictim | 1 | NormalLastVictim | 1 |
| NormalPersistent | 2 | NormalPersistent | 2 |
| NotAllocKeep | 4 | — | — |
| NotAllocClean | 5 | NotAllocClean | 4 |
| NotAllocDrop | 6 | — | — |

Normal hints request normal allocation with different eviction priorities;
`NormalPersistent` requests retention. `NotAlloc*` requests nonallocation, with
different handling of an already cached line where the target supports it.
Hints do not change the tensor's numerical contents or guarantee a speedup.

## Individual transfers

```python
x_tile = pl.load(x, [0, 0], [32, 32], l2_hint=pl.LoadL2Hint.NormalPersistent)
out = pl.store(x_tile, [0, 0], out, l2_hint=pl.StoreL2Hint.NotAllocClean)
```

Omitting `l2_hint` (or passing `None`) leaves the transfer unspecified. An
explicit `NormalFirstVictim` is preserved even though its encoding is zero,
because it overrides a scope declaration. Passing the wrong direction's enum
or an unsupported integer encoding is rejected.

## Scope declarations and implicit transfers

Use `pl.set_l2_hint(tensor, load=..., store=...)` when tensor operations generate
loads/stores implicitly. Supply either direction or both; the hints are
independent. The corresponding explicit access hint always takes precedence.

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

The declaration must be a standalone statement directly inside `pl.at` or an
inline `pl.spmd` body. It applies throughout that scope, irrespective of where
among its top-level statements it appears. The tensor must be bound before the
scope and actually captured by its body. Declarations nested in an `if` or
`for`, or outside a scope, are rejected. Identical repeated declarations are
deduplicated; conflicting hints for the same tensor and direction are errors.
A store declaration requires a tensor the scope writes.

Declarations resolve captured bindings by Var identity. Loads/stores generated
by tensor lowering inherit their hint, including GM destination rebindings and
loop carries. An alias that may refer to different parameters inherits only
when every possible parameter has the same declaration. Slices themselves take
no hint: the hint belongs to the transfer that moves the data.

A scope load `NotAlloc*` declaration on a tensor the scope writes is rejected.
For A2/A3 address aliases, also ensure that no concurrent cached writer outside
the scope modifies the same bytes; the compiler cannot prove that contract.
A declaration on a non-device hierarchy scope is carried as metadata but does
not configure transfers in a separately called device kernel.

## Target behavior

| Target | Normal load hints | NotAlloc load hints | Store hints |
| ------ | ----------------- | ------------------- | ----------- |
| A5 | Exact ISA template enum, selecting `l2CacheCtl` | Exact ISA template enum, selecting `l2CacheCtl` | Exact ISA template enum, selecting `l2CacheCtl` |
| A2/A3 | Accepted; no distinct hardware cache control | All three use the runtime no-cache address alias | Accepted; no distinct hardware cache control |

A2/A3 gets the device-specific offset from `get_l2_cache_offset(args)` and
forwards it as a synthetic kernel argument. Stock PTOAS applies that offset
once using `pto.tload`'s `cache_policy = #pto.load_cache_policy<l2_bypass>` and
`offset` operand. This is an internal assembler lowering detail, not a second
frontend policy. A device reporting offset zero performs an ordinary read.
PyPTO does not also emit an ISA NotAlloc template on that path: the A2/A3 ISA
implementation would add its own globally configured offset, which the PyPTO
runtime does not populate. A2/A3 cannot distinguish Keep/Clean/Drop through the
cache-control field as A5 does.

For other hints, PyPTO emits ordinary `pto.tload` / `pto.tstore` operations
surrounded by `emitc.verbatim` markers. Its C++ finalizer replaces those markers
with the chosen PTO-ISA template parameter, preserving store phase, atomic,
quantization and ReLU arguments. PTOAS still sees the transfers' memory effects
for synchronization and memory planning. Both memory planners are supported.
No PTOAS source changes are required; the pinned release is v0.65.

The debug rebuild path uses the same idempotent finalizer. With
`skip_ptoas=True`, the saved PTO retains markers: invoking PTOAS alone produces
intermediate C++ that still needs PyPTO finalization. Malformed or ambiguous
marked transfers fail compilation. Buffer lowering rejects explicit hints
until its transfer recipes support them.

## Compiler carrier chain

```text
pl.set_l2_hint(t, load=..., store=...)        parser consumes declaration
  -> Scope attrs load_l2_hint_vars / store_l2_hint_vars  (Var, encoding)
  -> Function attrs load_l2_hint / store_l2_hint         (param index, encoding)
  -> tile.load / tile.store kwarg l2_hint                exact encoding
  -> runtime address alias on A2/A3 NotAlloc loads
     or selected PTO-ISA template enum for other transfers
```

[OutlineIncoreScopes](../passes/09-outline_incore_scopes.md) consumes scope
metadata and sorts the parameter-index declarations. Then
[ConvertTensorToTileOps](../passes/11-convert_tensor_to_tile_ops.md) stamps
transfer kwargs and erases the consumed function attributes before later
passes change parameter lists. Parsing, printing, structural comparison and
serialization preserve both directions and explicit zero hints.

## Migrating the removed API

`CachePolicy`, `pl.load(cache=...)` and `pl.set_cache_policy` have been removed.
Replace `cache=CachePolicy.BYPASS` with
`l2_hint=pl.LoadL2Hint.NotAllocClean`; replace a scope bypass declaration with
`pl.set_l2_hint(t, load=pl.LoadL2Hint.NotAllocClean)`. Replace an explicit
`CachePolicy.DEFAULT` override with `pl.LoadL2Hint.NormalFirstVictim`, or omit
the argument when no override is needed. A2/A3 retains the runtime alias path;
on A5 the new NotAlloc hint now selects ISA cache control.
