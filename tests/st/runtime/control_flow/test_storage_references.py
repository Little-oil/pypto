# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Numerical acceptance for the public Mat rotation and live phi bugs.

Input and product dimensions match issues #3039 and #3049. Wider InOut
outputs add sentinel guards; extra observations keep the rotation seed and the
two branch inputs independently live, exposing accidental storage merging.
"""

import pypto.language as pl
import pytest
import torch
from harness import st
from pypto import passes

_SENTINEL = -701.0


@pl.jit.incore
def _rotate_kernel(a: pl.Tensor, b: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    seed = current
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [16 + block * 64, 8], out)
        current = following
    seed_left = pl.move(seed, target_memory=pl.MemorySpace.Left)
    seed_result = pl.tile.matmul(seed_left, right)
    pl.store(seed_result, [16, 8], saved)
    final_left = pl.move(current, target_memory=pl.MemorySpace.Left)
    final_result = pl.tile.matmul(final_left, right)
    pl.store(final_result, [96, 8], saved)
    return out, saved


@pl.jit
def _rotate_entry(a: pl.Tensor, b: pl.Tensor, out: pl.InOut[pl.Tensor], saved: pl.InOut[pl.Tensor]):
    with pl.spmd(1):
        out, saved = _rotate_kernel(a, b, out, saved)
    return out, saved


@pl.jit.incore
def _live_inputs_kernel(x: pl.Tensor, y: pl.Tensor, flag: pl.Tensor, out: pl.InOut[pl.Tensor]):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [8, 8], out)
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [24, 8], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [40, 8], out)
    return out


@pl.jit
def _live_inputs_entry(x: pl.Tensor, y: pl.Tensor, flag: pl.Tensor, out: pl.InOut[pl.Tensor]):
    with pl.spmd(1):
        out = _live_inputs_kernel(x, y, flag, out)
    return out


def _rotation_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    product = tensors["a"].float() @ tensors["b"].float()
    out, saved = tensors["out"].clone(), tensors["saved"].clone()
    out[16:528, 8:24] = product
    saved[16:80, 8:24] = product[:64]
    saved[96:160, 8:24] = product[-64:]
    return {"out": out, "saved": saved}


def _branch_golden(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    products = tensors["x"].float() @ tensors["y"].float()
    out = tensors["out"].clone()
    out[8:24, 8:24] = products[:16] if tensors["flag"].item() > 0 else products[16:]
    out[24:40, 8:24] = products[:16]
    out[40:56, 8:24] = products[16:]
    return {"out": out}


def _compare(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, reference in expected.items():
        result = actual[name]
        guard = reference == _SENTINEL
        assert torch.equal(result[guard], reference[guard]), f"{name}: sentinel guard was overwritten"
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


def _pattern(rows: int, columns: int, modulus: int, denominator: int, dtype: torch.dtype) -> torch.Tensor:
    grid = torch.arange(rows * columns, dtype=torch.float32).reshape(rows, columns)
    return ((grid.remainder(modulus) - modulus // 2) / denominator).to(dtype)


@st.cases(
    st.case(
        _rotate_entry,
        _pattern(512, 64, 31, 16, torch.float16),
        _pattern(64, 16, 19, 16, torch.float16),
        torch.full((544, 32), _SENTINEL),
        torch.full((176, 32), _SENTINEL),
        name="issue3039_rotation_seed_and_guards",
        golden=_rotation_golden,
        compare=_compare,
        memory_planner=passes.MemoryPlanner.PYPTO,
    ),
    *[
        st.case(
            _live_inputs_entry,
            _pattern(32, 16, 23, 8, torch.bfloat16).roll(16 * swapped, dims=0),
            _pattern(16, 16, 11, 8, torch.bfloat16),
            torch.tensor([flag], dtype=torch.int32),
            torch.full((64, 32), _SENTINEL),
            name=f"issue3049_{planner.name.lower()}_flag{flag}_swapped{swapped}",
            golden=_branch_golden,
            compare=_compare,
            memory_planner=planner,
        )
        for planner in (passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS)
        for swapped in (0, 1)
        for flag in (0, 1)
    ],
)
@pytest.mark.platforms("a2a3", reason="These public reproductions target the a2/a3 matrix pipeline.")
def test_storage_reference_numerics(case_run):
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
