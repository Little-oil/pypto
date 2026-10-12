# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A declared ``LoadL2Hint.NotAllocClean`` read returns the same bytes on device.

The bypass is an *address* change on a2a3: the load is issued against the
uncached alias of the page, ``addr + get_l2_cache_offset(args)``. That makes the
declaration the one kind of performance hint that can return wrong data — a
wrong offset reads unrelated memory rather than running slower, and on this box
the driver reports a nonzero one (``0x80000000000``), so the aliased path is
what actually executes here.

Only a device run can check that: the simulator has no alias and reports zero,
where ``addr + 0`` is the ordinary address and any offset plumbing would pass.
Both surfaces are covered, because they reach codegen by different routes — the
per-load kwarg annotates one ``tile.load``, while the scope declaration is
resolved to the outlined kernel's parameters and applied by
``ConvertTensorToTileOps``.

Inputs are small integers so the FP32 accumulation is exact and the comparison
runs at ``rtol=atol=0``: a load that read the wrong address would have to
reproduce the product by accident to pass.

Explicit L2 hints are covered separately by exact elementwise addition: every
load/store hint pair runs with both memory planners, including zero-valued hints
and a loop containing an ordinary load beside the hinted load. A2/A3
nonallocating loads use runtime address aliases; A5 uses ISA hints. These cases
check correctness, not cache-hit rates or performance; codegen unit tests
separately verify the emitted ISA arguments.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

M, K, N = 64, 256, 128


@pl.jit.incore
def _bypass_load_kernel(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Tile programming: the weight's one read declares the policy."""
    xt = pl.load(x, [0, 0], [M, K], target_memory=pl.Mem.Mat)
    wt = pl.load(w, [0, 0], [N, K], target_memory=pl.Mem.Mat, l2_hint=pl.LoadL2Hint.NotAllocClean)
    return pl.store(pl.matmul(xt, pl.tile.transpose_view(wt)), [0, 0], out)


@pl.jit
def bypass_load(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return _bypass_load_kernel(x, w, out)


@pl.jit
def bypass_scope(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[K, N], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Tensor programming: every read of ``w`` in the scope declares the policy."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
        pl.set_l2_hint(w, load=pl.LoadL2Hint.NotAllocClean)
        c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(x, w, out_dtype=pl.FP32)
        out = pl.assemble(out, c, [0, 0])
    return out


def _inputs(transposed_weight: bool):
    """Small integers, so the FP32 accumulation is exact at rtol=atol=0."""
    generator = torch.Generator().manual_seed(11)
    x = torch.randint(-4, 5, (M, K), generator=generator).to(torch.float16)
    w_shape = (N, K) if transposed_weight else (K, N)
    w = torch.randint(-4, 5, w_shape, generator=generator).to(torch.float16)
    return x, w, torch.zeros((M, N), dtype=torch.float32)


def _golden_transposed(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(tensors["x"].to(torch.float32), tensors["w"].to(torch.float32).T)


def _golden(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.matmul(tensors["x"].to(torch.float32), tensors["w"].to(torch.float32))


@st.cases(
    st.case(
        bypass_load,
        *_inputs(transposed_weight=True),
        name="cache_bypass_per_load",
        golden=_golden_transposed,
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        bypass_scope,
        *_inputs(transposed_weight=False),
        name="cache_bypass_scope",
        golden=_golden,
        rtol=0.0,
        atol=0.0,
    ),
)
def test_cache_bypass_reads_the_same_bytes(case_run):
    """The aliased read is exact, so the declaration costs cache and nothing else."""
    case_run.assert_passed()


def _l2_hint_case(load_hint, store_hint, planner):
    """Exercise explicit hints beside ordinary transfers inside a tile loop."""

    @pl.jit.incore
    def hinted_kernel(
        x: pl.Tensor[[64, 32], pl.FP32],
        y: pl.Tensor[[64, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        for i, (out_iter,) in pl.range(2, init_values=(out,)):
            a = pl.load(x, [i * 32, 0], [32, 32], l2_hint=load_hint)
            b = pl.load(y, [i * 32, 0], [32, 32])
            updated = pl.store(pl.add(a, b), [i * 32, 0], out_iter, l2_hint=store_hint)
            out = pl.yield_(updated)
        return out

    @pl.jit
    def hinted_add(
        x: pl.Tensor[[64, 32], pl.FP32],
        y: pl.Tensor[[64, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
    ) -> pl.Tensor[[64, 32], pl.FP32]:
        return hinted_kernel(x, y, out)

    x = (torch.arange(64 * 32).reshape(64, 32) % 127 - 63).float()
    y = (torch.arange(64 * 32).reshape(64, 32) % 31 - 15).float()
    return st.case(
        hinted_add,
        x,
        y,
        torch.full_like(x, float("nan")),
        name=f"l2_{load_hint.name}_{store_hint.name}_{planner.name}",
        golden=lambda tensors: tensors["x"] + tensors["y"],
        memory_planner=planner,
        rtol=0.0,
        atol=0.0,
    )


@st.cases(
    *[
        _l2_hint_case(load_hint, store_hint, planner)
        for load_hint in (
            pl.LoadL2Hint.NormalFirstVictim,
            pl.LoadL2Hint.NormalLastVictim,
            pl.LoadL2Hint.NormalPersistent,
        )
        for store_hint in pl.StoreL2Hint
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
def test_l2_hints_preserve_results(case_run):
    """Normal load hints and all store hints preserve exact results on both chips."""
    case_run.assert_passed()


@st.cases(
    *[
        _l2_hint_case(load_hint, store_hint, planner)
        for load_hint in (
            pl.LoadL2Hint.NotAllocKeep,
            pl.LoadL2Hint.NotAllocClean,
            pl.LoadL2Hint.NotAllocDrop,
        )
        for store_hint in pl.StoreL2Hint
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
def test_l2_not_alloc_hints_preserve_results(case_run):
    """Nonallocating loads must read the right bytes for every store hint."""
    case_run.assert_passed()


def _scope_l2_hint_case(store_hint, planner):
    @pl.jit
    def scope_add(
        x: pl.Tensor[[32, 32], pl.FP32],
        y: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_l2_hint(x, load=pl.LoadL2Hint.NotAllocKeep)
            pl.set_l2_hint(out, store=store_hint)
            out = pl.assemble(out, pl.add(x, y), [0, 0])
            out = pl.assemble(out, pl.add(x, 2.0), [0, 0])
        return out

    x = (torch.arange(32 * 32).reshape(32, 32) % 127 - 63).float()
    y = torch.full_like(x, 7.0)
    return st.case(
        scope_add,
        x,
        y,
        torch.full_like(x, float("nan")),
        name=f"scope_l2_{store_hint.name}_{planner.name}",
        golden=lambda tensors: tensors["x"] + 2.0,
        rtol=0.0,
        atol=0.0,
        memory_planner=planner,
    )


@st.cases(
    *[
        _scope_l2_hint_case(store_hint, planner)
        for store_hint in pl.StoreL2Hint
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
def test_scope_l2_hints_preserve_implicit_transfers(case_run):
    """Scope hints on generated transfers survive successive output rebindings."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
