# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Explicit and scope L2 hints preserve exact results on supported runtimes.

All six load hints and all four store hints are generated for A2/A3 and A5,
with both memory planners. User-selected enums are passed to ISA without an
external address offset. Numerical checks cover transfers in loops, ordinary
loads next to hinted loads, and successive scope output rebindings.

A2/A3 ISA NotAlloc loads require simpler to initialize the ISA L2 configuration
global. Until that integration is available, those onboard device cases are
skipped before fixture setup and still run under --codegen-only. A5 and A2/A3
Normal hints remain runnable. CPU simulation ignores L2 hints and provides
numerical coverage only; it cannot validate cache behavior. Codegen unit tests
verify the exact ISA template arguments.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto.pypto_core.passes import MemoryPlanner

M, K, N = 64, 256, 128

# A marker is required: the harness may submit discovered kernels before test
# bodies run, so a pytest.skip inside the body would be too late.
_A2A3_NOT_ALLOC_SKIP = pytest.mark.skipif(
    "not config.getoption('--codegen-only')",
    reason=(
        "A2/A3 ISA NotAlloc loads require simpler to initialize "
        "g_opL2CacheHintCfg.l2Cacheoffset: https://github.com/hw-native-sys/simpler/issues/2548"
    ),
)
_NOT_ALLOC_PLATFORMS = [
    pytest.param("a2a3", marks=_A2A3_NOT_ALLOC_SKIP),
    "a2a3sim",
    "a5",
    "a5sim",
]


@pl.jit.incore
def _not_alloc_load_kernel(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Tile programming: the weight's one read selects its ISA hint."""
    xt = pl.load(x, [0, 0], [M, K], target_memory=pl.Mem.Mat)
    wt = pl.load(w, [0, 0], [N, K], target_memory=pl.Mem.Mat, l2_hint=pl.LoadL2Hint.NotAllocClean)
    return pl.store(pl.matmul(xt, pl.tile.transpose_view(wt)), [0, 0], out)


@pl.jit
def not_alloc_load(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[N, K], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    return _not_alloc_load_kernel(x, w, out)


@pl.jit
def not_alloc_scope(
    x: pl.Tensor[[M, K], pl.FP16],
    w: pl.Tensor[[K, N], pl.FP16],
    out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
) -> pl.Tensor[[M, N], pl.FP32]:
    """Tensor programming: every read of ``w`` in the scope selects its ISA hint."""
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
        not_alloc_load,
        *_inputs(transposed_weight=True),
        name="not_alloc_per_load",
        golden=_golden_transposed,
        rtol=0.0,
        atol=0.0,
    ),
    st.case(
        not_alloc_scope,
        *_inputs(transposed_weight=False),
        name="not_alloc_scope",
        golden=_golden,
        rtol=0.0,
        atol=0.0,
    ),
)
@pytest.mark.parametrize("platform", _NOT_ALLOC_PLATFORMS)
def test_not_alloc_load_and_scope_preserve_results(case_run, platform):
    """Per-access and scope NotAlloc matmul reads preserve exact bytes."""
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
@pytest.mark.parametrize("platform", _NOT_ALLOC_PLATFORMS)
def test_l2_not_alloc_hints_preserve_results(case_run, platform):
    """Nonallocating loads must read the right bytes for every store hint."""
    case_run.assert_passed()


def _scope_l2_hint_case(load_hint, store_hint, planner):
    @pl.jit
    def scope_add(
        x: pl.Tensor[[32, 32], pl.FP32],
        y: pl.Tensor[[32, 32], pl.FP32],
        out: pl.Out[pl.Tensor[[32, 32], pl.FP32]],
    ) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_l2_hint(x, load=load_hint)
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
        name=f"scope_l2_{load_hint.name}_{store_hint.name}_{planner.name}",
        golden=lambda tensors: tensors["x"] + 2.0,
        rtol=0.0,
        atol=0.0,
        memory_planner=planner,
    )


@st.cases(
    *[
        _scope_l2_hint_case(load_hint, store_hint, planner)
        for load_hint in (
            pl.LoadL2Hint.NormalFirstVictim,
            pl.LoadL2Hint.NormalLastVictim,
            pl.LoadL2Hint.NormalPersistent,
        )
        for store_hint in pl.StoreL2Hint
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
def test_scope_l2_hints_preserve_implicit_transfers(case_run):
    """Scope hints on generated transfers survive successive output rebindings."""
    case_run.assert_passed()


@st.cases(
    *[
        _scope_l2_hint_case(load_hint, store_hint, planner)
        for load_hint in (
            pl.LoadL2Hint.NotAllocKeep,
            pl.LoadL2Hint.NotAllocClean,
            pl.LoadL2Hint.NotAllocDrop,
        )
        for store_hint in pl.StoreL2Hint
        for planner in (MemoryPlanner.PYPTO, MemoryPlanner.PTOAS)
    ]
)
@pytest.mark.parametrize("platform", _NOT_ALLOC_PLATFORMS)
def test_scope_not_alloc_hints_preserve_implicit_transfers(case_run, platform):
    """Scope NotAlloc reads retain user choices through implicit transfers."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
