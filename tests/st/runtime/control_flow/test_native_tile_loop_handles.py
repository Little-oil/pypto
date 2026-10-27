# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Observe native Tile handle exchange, zero-trip results and runtime valid extents.

The inputs are preloaded and independently observed after the loop. This tests
reference and metadata transport, not repeated-producer storage protection.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes

_SENTINEL = -701.0


@pl.jit.incore
def _for_handles(
    x: pl.Tensor,
    y: pl.Tensor,
    control: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    history: pl.InOut[pl.Tensor],
    DYNAMIC: pl.constexpr,
):
    if DYNAMIC:
        rows = pl.read(control, [2])
    else:
        rows = pl.const(16, pl.INT32)
    first = pl.load(x, [0, 0], [16, 16], valid_shape=[rows, 16], target_memory=pl.MemorySpace.Mat)
    second = pl.load(x, [16, 0], [16, 16], valid_shape=[rows, 16], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.move(rhs, target_memory=pl.MemorySpace.Right)
    if pl.read(control, [1]) > 0:
        current, spare = pl.yield_(first, second)
    else:
        current, spare = pl.yield_(second, first)
    for block in pl.range(pl.read(control, [0])):
        left = pl.move(current, target_memory=pl.MemorySpace.Left)
        product = pl.tile.matmul(left, right)
        pl.store(product, [4 + block * 16, 8], history)
        previous = current
        current = spare
        spare = previous
    left = pl.move(current, target_memory=pl.MemorySpace.Left)
    product = pl.tile.matmul(left, right)
    pl.store(product, [8, 8], out)
    keep_first = pl.matmul(first, rhs, out_dtype=pl.FP32)
    pl.store(keep_first, [24, 8], out)
    keep_second = pl.matmul(second, rhs, out_dtype=pl.FP32)
    pl.store(keep_second, [40, 8], out)
    return out, history


@pl.jit.incore
def _while_handles(
    x: pl.Tensor,
    y: pl.Tensor,
    control: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    history: pl.InOut[pl.Tensor],
    DYNAMIC: pl.constexpr,
):
    if DYNAMIC:
        rows = pl.read(control, [2])
    else:
        rows = pl.const(16, pl.INT32)
    first = pl.load(x, [0, 0], [16, 16], valid_shape=[rows, 16], target_memory=pl.MemorySpace.Mat)
    second = pl.load(x, [16, 0], [16, 16], valid_shape=[rows, 16], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.move(rhs, target_memory=pl.MemorySpace.Right)
    if pl.read(control, [1]) > 0:
        initial, spare_initial = pl.yield_(first, second)
    else:
        initial, spare_initial = pl.yield_(second, first)
    count = pl.read(control, [0])
    for current, spare, block in pl.while_(init_values=(initial, spare_initial, 0)):
        pl.cond(block < count)
        left = pl.move(current, target_memory=pl.MemorySpace.Left)
        product = pl.tile.matmul(left, right)
        pl.store(product, [4 + block * 16, 8], history)
        result, _spare, _blocks = pl.yield_(spare, current, block + 1)
    left = pl.move(result, target_memory=pl.MemorySpace.Left)
    product = pl.tile.matmul(left, right)
    pl.store(product, [8, 8], out)
    keep_first = pl.matmul(first, rhs, out_dtype=pl.FP32)
    pl.store(keep_first, [24, 8], out)
    keep_second = pl.matmul(second, rhs, out_dtype=pl.FP32)
    pl.store(keep_second, [40, 8], out)
    return out, history


@pl.jit
def _for_entry(
    x: pl.Tensor,
    y: pl.Tensor,
    control: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    history: pl.InOut[pl.Tensor],
    DYNAMIC: pl.constexpr,
):
    with pl.spmd(1):
        out, history = _for_handles(x, y, control, out, history, DYNAMIC)
    return out, history


@pl.jit
def _while_entry(
    x: pl.Tensor,
    y: pl.Tensor,
    control: pl.Tensor,
    out: pl.InOut[pl.Tensor],
    history: pl.InOut[pl.Tensor],
    DYNAMIC: pl.constexpr,
):
    with pl.spmd(1):
        out, history = _while_handles(x, y, control, out, history, DYNAMIC)
    return out, history


def _golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    count, select_first, rows = [int(value) for value in tensors["control"].tolist()]
    products = tensors["x"].float() @ tensors["y"].float()
    out, history = tensors["out"].clone(), tensors["history"].clone()

    def observe(destination: torch.Tensor, offset: int, first: bool) -> None:
        source = products[:rows] if first else products[16 : 16 + rows]
        destination[offset : offset + rows, 8:24] = source

    for block in range(count):
        observe(history, 4 + block * 16, bool(select_first) != bool(block % 2))
    observe(out, 8, bool(select_first) != bool(count % 2))
    observe(out, 24, True)
    observe(out, 40, False)
    return {"out": out, "history": history}


def _compare(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, reference in expected.items():
        result = actual[name]
        guard = reference == _SENTINEL
        assert torch.equal(result[guard], reference[guard]), (
            f"{name}: invalid rows or guards were overwritten"
        )
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def _pattern(rows: int, columns: int, modulus: int) -> torch.Tensor:
    grid = torch.arange(rows * columns).reshape(rows, columns).float()
    return ((grid.remainder(modulus) - modulus // 2) / 8).to(torch.bfloat16)


@st.cases(
    *[
        st.case(
            entry,
            _pattern(32, 16, 23),
            _pattern(16, 16, 11),
            torch.tensor([count, select_first, 8 if dynamic else 16], dtype=torch.int32),
            torch.full((64, 32), _SENTINEL),
            torch.full((64, 32), _SENTINEL),
            dynamic,
            name=f"native_{kind}_{'dynamic' if dynamic else 'static'}_trip{count}_seed{select_first}",
            golden=_golden,
            compare=_compare,
            memory_planner=passes.MemoryPlanner.PYPTO,
        )
        for kind, entry in (("for", _for_entry), ("while", _while_entry))
        for dynamic in (False, True)
        for count in (0, 1, 2)
        for select_first in (0, 1)
    ]
)
@pytest.mark.platforms("a2a3", reason="Native matrix Tile handle and runtime-valid transport on a2/a3.")
def test_native_tile_loop_handles(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
