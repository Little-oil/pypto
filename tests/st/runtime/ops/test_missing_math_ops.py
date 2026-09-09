# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A2/A3 runtime coverage for the missing math APIs from issue #2665.

Each case executes one public operator and checks a Torch golden, independently
of the other new operators. FP16 goldens accumulate in FP32 and round only the
result, matching the API contract. A5 validation is intentionally out of scope.
"""

import pypto.language as pl
import pytest
import torch
from harness import st

pytestmark = pytest.mark.platforms(
    "a2a3", "a2a3sim", reason="Issue #2665 requests A2/A3 ST coverage; A5 is deferred."
)

_DTYPES = (torch.float16, torch.float32)


def _dtype_name(dtype: torch.dtype) -> str:
    return "fp16" if dtype == torch.float16 else "fp32"


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
    tolerance = 2e-3 if dtype == torch.float16 else 2e-5
    return {"rtol": tolerance, "atol": tolerance}


def _random(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2665)
    return torch.randn(shape, generator=generator).to(dtype)


def _pow_kernel(exponent: int | float):
    @pl.jit
    def pow_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        M, N = a.shape
        with pl.at(level=pl.Level.CORE_GROUP):
            x = pl.load(a, [0, 0], [M, N])
            y = pl.pow(x, exponent)
            pl.store(y, [0, 0], out)
        return out

    return pow_kernel


def _pow_case(exponent: int | float, dtype: torch.dtype):
    a = torch.linspace(0.25, 4, 16 * 32, dtype=dtype).reshape(16, 32)
    if exponent != 0.5:
        a[:, ::2] *= -1
        if exponent >= 0:
            a[0, 0] = 0
    return st.case(
        _pow_kernel(exponent),
        a,
        torch.zeros_like(a),
        name=f"pow_exponent{exponent}_{_dtype_name(dtype)}",
        golden=lambda tensors: torch.pow(tensors["a"].float(), exponent).to(dtype),
        **_tolerance(dtype),
    )


@st.cases(*(_pow_case(exponent, dtype) for exponent in (-2, 0, 1, 2, 3, 0.5) for dtype in _DTYPES))
def test_pow(case_run):
    """Integer powers accept signed bases; fractional powers use positive inputs."""
    case_run.assert_passed()


def _mean_kernel(axis: int, valid_rows: int, valid_cols: int):
    @pl.jit
    def mean_kernel(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
        M, N = a.shape
        with pl.at(level=pl.Level.CORE_GROUP):
            x = pl.load(a, [0, 0], [M, N], valid_shape=[valid_rows, valid_cols])
            y = pl.mean(x, axis=axis)
            pl.store(y, [0, 0], out)
        return out

    return mean_kernel


def _mean_case(axis: int, padded: bool, dtype: torch.dtype):
    valid_rows, valid_cols = (7, 19) if padded else (16, 32)
    a = torch.full((16, 32), 1000, dtype=dtype)
    a[:valid_rows, :valid_cols] = _random((valid_rows, valid_cols), dtype)
    out_shape = (1, 32) if axis in (0, -2) else (16, 1)

    def golden(tensors):
        result = torch.zeros_like(tensors["out"])
        valid = tensors["a"][:valid_rows, :valid_cols].float()
        reduced = torch.mean(valid, dim=axis, keepdim=True).to(dtype)
        result[: reduced.shape[0], : reduced.shape[1]] = reduced
        return result

    return st.case(
        _mean_kernel(axis, valid_rows, valid_cols),
        a,
        torch.zeros(out_shape, dtype=dtype),
        name=f"mean_axis{axis}_padded{padded}_{_dtype_name(dtype)}",
        golden=golden,
        **_tolerance(dtype),
    )


@st.cases(
    *(
        _mean_case(axis, padded, dtype)
        for axis in (0, 1, -2, -1)
        for padded in (False, True)
        for dtype in _DTYPES
    )
)
def test_mean(case_run):
    """Both axis spellings retain dimensions and divide by the valid extent."""
    case_run.assert_passed()


def _small_mean_case(shape: tuple[int, int], axis: int, dtype: torch.dtype):
    rows, columns = shape
    # Keep all three rows when producing [3,1], but exclude a sentinel row
    # when reducing those rows. Every case also excludes trailing columns.
    valid_rows = 2 if rows == 3 and axis == 0 else 7 if rows == 16 else rows
    valid_cols = 11 if columns == 16 else 19
    a = torch.full(shape, 1000, dtype=dtype)
    a[:valid_rows, :valid_cols] = _random((valid_rows, valid_cols), dtype)
    output_shape = (1, columns) if axis == 0 else (rows, 1)

    def golden(tensors):
        expected = torch.zeros_like(tensors["out"])
        valid = tensors["a"][:valid_rows, :valid_cols].float()
        reduced = valid.mean(dim=axis, keepdim=True).to(dtype)
        expected[: reduced.shape[0], : reduced.shape[1]] = reduced
        return expected

    return st.case(
        _mean_kernel(axis, valid_rows, valid_cols),
        a,
        torch.zeros(output_shape, dtype=dtype),
        name=f"mean_small{rows}x{columns}_axis{axis}_{_dtype_name(dtype)}",
        golden=golden,
        **_tolerance(dtype),
    )


@st.cases(
    *(
        _small_mean_case(shape, axis, dtype)
        for shape in ((1, 32), (3, 32), (16, 16))
        for axis in (0, 1)
        for dtype in _DTYPES
    )
)
def test_mean_small_shapes(case_run):
    """Short vectors and scalar results keep their public shape and ignore padding."""
    case_run.assert_passed()


def _clamp_kernel(lower: float | None, upper: float | None):
    # Closure specialization folds scalar literals, not a captured None. Omit
    # an absent bound at the call site, as a user would for one-sided clamping.
    if lower is None:

        @pl.jit
        def clamp_max_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                x = pl.load(a, [0, 0], [M, N])
                y = pl.clamp(x, max=upper)
                pl.store(y, [0, 0], out)
            return out

        return clamp_max_kernel
    if upper is None:

        @pl.jit
        def clamp_min_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
            M, N = a.shape
            with pl.at(level=pl.Level.CORE_GROUP):
                x = pl.load(a, [0, 0], [M, N])
                y = pl.clamp(x, min=lower)
                pl.store(y, [0, 0], out)
            return out

        return clamp_min_kernel

    @pl.jit
    def clamp_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
        M, N = a.shape
        with pl.at(level=pl.Level.CORE_GROUP):
            x = pl.load(a, [0, 0], [M, N])
            y = pl.clamp(x, min=lower, max=upper)
            pl.store(y, [0, 0], out)
        return out

    return clamp_kernel


def _clamp_case(lower: float | None, upper: float | None, dtype: torch.dtype):
    a = torch.tensor([-4, -1, -0.5, -0.0, 0, 0.5, 1, 4], dtype=dtype).repeat(16, 4)
    return st.case(
        _clamp_kernel(lower, upper),
        a,
        torch.zeros_like(a),
        name=f"clamp_min{lower}_max{upper}_{_dtype_name(dtype)}",
        golden=lambda tensors: torch.clamp(tensors["a"].float(), min=lower, max=upper).to(dtype),
        rtol=0,
        atol=0,
    )


@st.cases(
    *(
        _clamp_case(lower, upper, dtype)
        for lower, upper in ((-0.5, None), (None, 0.5), (-0.5, 0.5), (1.0, -1.0))
        for dtype in _DTYPES
    )
)
def test_clamp(case_run):
    """One/two-sided bounds and min > max follow the standalone clamp contract."""
    case_run.assert_passed()


@pl.jit
def sqrt_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    M, N = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        x = pl.load(a, [0, 0], [M, N])
        y = pl.sqrt(x)
        pl.store(y, [0, 0], out)
    return out


def _sqrt_case(dtype: torch.dtype):
    a = torch.linspace(0, 100, 16 * 32, dtype=dtype).reshape(16, 32)
    return st.case(
        sqrt_kernel,
        a,
        torch.zeros_like(a),
        name=f"sqrt_nonnegative_{_dtype_name(dtype)}",
        golden=lambda tensors: torch.sqrt(tensors["a"].float()).to(dtype),
        **_tolerance(dtype),
    )


@st.cases(*(_sqrt_case(dtype) for dtype in _DTYPES))
def test_sqrt(case_run):
    """The existing sqrt API handles exact zero and positive FP16/FP32 inputs."""
    case_run.assert_passed()


@pl.jit
def mean_then_pow_zero_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    M, N = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        x = pl.load(a, [0, 0], [M, N])
        row_mean = pl.mean(x, axis=1)
        y = pl.pow(row_mean, 0)
        pl.store(y, [0, 0], out)
    return out


@pl.jit
def mean_both_axes_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    M, N = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        x = pl.load(a, [0, 0], [M, N])
        row_mean = pl.mean(x, axis=1)
        y = pl.mean(row_mean, axis=0)
        pl.store(y, [0, 0], out)
    return out


def _mean_composition_case(pow_zero: bool, dtype: torch.dtype):
    a = _random((16, 32), dtype)

    def golden(tensors):
        # Each public mean returns the input dtype, including the intermediate
        # result of two consecutive reductions.
        row_mean = tensors["a"].float().mean(dim=1, keepdim=True).to(dtype)
        if pow_zero:
            return row_mean.float().pow(0).to(dtype)
        return row_mean.float().mean(dim=0, keepdim=True).to(dtype)

    name = "mean_then_pow_zero" if pow_zero else "mean_both_axes"
    return st.case(
        mean_then_pow_zero_kernel if pow_zero else mean_both_axes_kernel,
        a,
        torch.zeros((16, 1) if pow_zero else (1, 1), dtype=dtype),
        name=f"{name}_{_dtype_name(dtype)}",
        golden=golden,
        **_tolerance(dtype),
    )


@st.cases(*(_mean_composition_case(pow_zero, dtype) for pow_zero in (True, False) for dtype in _DTYPES))
def test_mean_result_composition(case_run):
    """A column-layout mean result is consumable by pow and the opposite-axis mean."""
    case_run.assert_passed()


@pl.jit
def mean_axes_zero_then_one_kernel(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    M, N = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        x = pl.load(a, [0, 0], [M, N], valid_shape=[2, 19])
        column_mean = pl.mean(x, axis=0)
        y = pl.mean(column_mean, axis=1)
        pl.store(y, [0, 0], out)
    return out


def _mean_axes_zero_then_one_case(dtype: torch.dtype):
    a = torch.full((3, 32), 1000, dtype=dtype)
    a[:2, :19] = _random((2, 19), dtype)

    def golden(tensors):
        valid = tensors["a"][:2, :19].float()
        # Round the first public mean result before the second accumulation.
        column_mean = valid.mean(dim=0, keepdim=True).to(dtype)
        return column_mean.float().mean(dim=1, keepdim=True).to(dtype)

    return st.case(
        mean_axes_zero_then_one_kernel,
        a,
        torch.zeros(1, 1, dtype=dtype),
        name=f"mean_axes0_then1_valid2x19_{_dtype_name(dtype)}",
        golden=golden,
        **_tolerance(dtype),
    )


@st.cases(*(_mean_axes_zero_then_one_case(dtype) for dtype in _DTYPES))
def test_mean_axes_zero_then_one(case_run):
    """The row-layout intermediate excludes its padded columns in the second mean."""
    case_run.assert_passed()


@pl.jit
def pow_zero_rank3_kernel(a: pl.Tensor, out: pl.InOut[pl.Tensor]):
    B, M, N = a.shape
    with pl.at(level=pl.Level.CORE_GROUP):
        # The fully valid middle axis makes this an ND valid region that can
        # flatten to one rectangular tile: [2,16,32] -> [32,32], valid [16,19].
        x = pl.load(a, [0, 0, 0], [B, M, N], valid_shape=[1, M, 19])
        y = pl.pow(x, 0)
        pl.store(y, [0, 0, 0], out)
    return out


def _pow_zero_rank3_case(dtype: torch.dtype):
    a = _random((2, 16, 32), dtype)
    a[0, 0, 0] = 0

    def golden(tensors):
        # Values outside valid_shape are undefined by the tile contract. NaNs
        # make the golden checker ignore that physical padding while retaining
        # exact validation of every logical element.
        expected = torch.full_like(tensors["out"], float("nan"))
        expected[:1, :, :19] = tensors["a"][:1, :, :19].float().pow(0).to(dtype)
        return expected

    return st.case(
        pow_zero_rank3_kernel,
        a,
        torch.zeros_like(a),
        name=f"pow_zero_rank3_valid1x16x19_{_dtype_name(dtype)}",
        golden=golden,
        rtol=0,
        atol=0,
    )


@st.cases(*(_pow_zero_rank3_case(dtype) for dtype in _DTYPES))
def test_pow_zero_preserves_rank3_valid_region(case_run):
    """Power zero writes ones throughout the ND logical valid region."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
