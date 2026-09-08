# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Numerical validation of recipes used by LowerCompositeOps.

This test file mirrors the exact mathematical recipe implemented in
``src/ir/transforms/lower_composite_ops_pass.cpp`` (Cody-Waite range reduction +
degree-9 odd Horner polynomial) in pure NumPy, then compares the result
against ``numpy.sin`` / ``numpy.cos`` over the validated input range
``|x| <= 2*pi*1024``.

The standalone-math tests additionally evaluate the emitted primitive IR using
a small NumPy interpreter. This checks the actual lowering rather than a mirror
of each composite recipe; layouts and target legality are covered by compilation tests.
"""

from typing import Any

import numpy as np
import pytest
from pypto import ir, passes

# ============================================================================
# FP32 constants. These MUST match the values in
# src/ir/transforms/lower_composite_ops_pass.cpp digit-for-digit.
# ============================================================================
PI_INV = np.float32(0.31830988732818603515625)
PI_V2 = np.float32(3.140625)
PI_C1 = np.float32(0.0009670257568359375)
PI_C2 = np.float32(6.2771141529083251953125e-7)
PI_C3 = np.float32(1.21644916362129151821e-10)
PI_C4 = np.float32(-1.0290623200529979163e-13)
PI_HALF_HEAD = np.float32(1.57079637050628662109375)
PI_HALF_TAIL = np.float32(-4.371139000189375e-8)
HALF = np.float32(0.5)
M4 = np.float32(4.0)
NEG2 = np.float32(-2.0)
ONE = np.float32(1.0)
R0 = np.float32(2.604926501e-6)
R1 = np.float32(-1.980894471e-4)
R2 = np.float32(8.333049340e-3)
R3 = np.float32(-1.666665792e-1)


def lowered_sin(x: np.ndarray) -> np.ndarray:
    """Pure-Python mirror of the sin lowering in lower_composite_ops_pass.cpp."""
    x = x.astype(np.float32)

    # Range reduction: k = round(x / pi)
    k_f = (x * PI_INV).astype(np.float32)
    # CAST_ROUND in PTO is round-half-away-from-zero (ISO C lround), not
    # banker's rounding. ``np.round`` follows IEEE round-half-to-even, so it
    # would diverge from the hardware on tie inputs (e.g. 0.5, 1.5).
    k_i = (np.sign(k_f) * np.floor(np.abs(k_f) + 0.5)).astype(np.int32)  # CAST_ROUND
    k_f = k_i.astype(np.float32)

    # 4-part Cody-Waite subtraction: t = x - k * pi
    t = x.astype(np.float32)
    for c in (PI_V2, PI_C1, PI_C2, PI_C3, PI_C4):
        t = (t - (k_f * c).astype(np.float32)).astype(np.float32)

    # sign = floor(k/2) * 4 + k * (-2) + 1
    half_k = (k_f * HALF).astype(np.float32)
    floor_hk_i = np.floor(half_k).astype(np.int32)  # CAST_FLOOR
    floor_hk_f = floor_hk_i.astype(np.float32)
    sign_pre = ((floor_hk_f * M4).astype(np.float32) + (k_f * NEG2).astype(np.float32)).astype(np.float32)
    sign = (sign_pre + ONE).astype(np.float32)

    # Horner: P(t^2) = (((R0*t^2 + R1)*t^2 + R2)*t^2 + R3)*t^2 + 1
    t2 = (t * t).astype(np.float32)
    p = ((t2 * R0).astype(np.float32) + R1).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R2).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R3).astype(np.float32)
    p = ((p * t2).astype(np.float32) + ONE).astype(np.float32)

    # out = sign * (t * P(t^2))
    t_p = (t * p).astype(np.float32)
    return (sign * t_p).astype(np.float32)


def lowered_cos(x: np.ndarray) -> np.ndarray:
    """Pure-Python mirror of the cos lowering in lower_composite_ops_pass.cpp.

    Differs from sin in three places:
      1. k = rint(x * PI_INV + 0.5) (RINT mode = banker's rounding to even)
      2. After the PI_C1 subtraction, +PI_HALF_HEAD is added.
      3. After the PI_C4 subtraction, +PI_HALF_TAIL is added.
    """
    x = x.astype(np.float32)

    # Range reduction: k = rint(x / pi + 0.5). The +0.5 is applied to the
    # FP32 product BEFORE the cast, then RINT (banker's rounding to even).
    pi_inv_x = (x * PI_INV).astype(np.float32)
    k_pre = (pi_inv_x + HALF).astype(np.float32)
    k_i = np.rint(k_pre).astype(np.int32)  # CAST_RINT
    k_f = k_i.astype(np.float32)

    # 4-part Cody-Waite subtraction with PI_HALF_HEAD interleaved between
    # PI_C1 and PI_C2, and PI_HALF_TAIL after PI_C4.
    t = x.astype(np.float32)
    t = (t - (k_f * PI_V2).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C1).astype(np.float32)).astype(np.float32)
    t = (t + PI_HALF_HEAD).astype(np.float32)
    t = (t - (k_f * PI_C2).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C3).astype(np.float32)).astype(np.float32)
    t = (t - (k_f * PI_C4).astype(np.float32)).astype(np.float32)
    t = (t + PI_HALF_TAIL).astype(np.float32)

    # sign = floor(k/2) * 4 + k * (-2) + 1
    half_k = (k_f * HALF).astype(np.float32)
    floor_hk_i = np.floor(half_k).astype(np.int32)
    floor_hk_f = floor_hk_i.astype(np.float32)
    sign_pre = ((floor_hk_f * M4).astype(np.float32) + (k_f * NEG2).astype(np.float32)).astype(np.float32)
    sign = (sign_pre + ONE).astype(np.float32)

    # Horner: same as sin
    t2 = (t * t).astype(np.float32)
    p = ((t2 * R0).astype(np.float32) + R1).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R2).astype(np.float32)
    p = ((p * t2).astype(np.float32) + R3).astype(np.float32)
    p = ((p * t2).astype(np.float32) + ONE).astype(np.float32)

    t_p = (t * p).astype(np.float32)
    return (sign * t_p).astype(np.float32)


# ============================================================================
# Tests
# ============================================================================
# Validated input range: |x| <= 2*pi*1024 (~6435.0). Beyond this the
# range-reduction error grows because k_f loses too many integer bits.
_RANGE = 2.0 * float(np.pi) * 1024.0


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_recipe_matches_numpy_sin(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-_RANGE, _RANGE, size=(2048,)).astype(np.float32)
    expected = np.sin(x).astype(np.float32)
    actual = lowered_sin(x)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_recipe_matches_numpy_cos(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-_RANGE, _RANGE, size=(2048,)).astype(np.float32)
    expected = np.cos(x).astype(np.float32)
    actual = lowered_cos(x)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_recipe_special_points_sin():
    """Verify the recipe matches at well-known points (within FP32 noise)."""
    cases = [
        (0.0, 0.0),
        (float(np.pi) / 2, 1.0),
        (float(np.pi), 0.0),
        (3 * float(np.pi) / 2, -1.0),
        (2 * float(np.pi), 0.0),
    ]
    for x_val, expected in cases:
        actual = lowered_sin(np.array([x_val], dtype=np.float32))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_recipe_special_points_cos():
    cases = [
        (0.0, 1.0),
        (float(np.pi) / 2, 0.0),
        (float(np.pi), -1.0),
        (3 * float(np.pi) / 2, 0.0),
        (2 * float(np.pi), 1.0),
    ]
    for x_val, expected in cases:
        actual = lowered_cos(np.array([x_val], dtype=np.float32))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-5)


def _math_program(name, shape, dtype, valid_shape=None, **kwargs):
    span = ir.Span.unknown()
    view = ir.TileView(valid_shape=valid_shape) if valid_shape is not None else None
    x = ir.Var("x", ir.TileType(shape, dtype, tile_view=view), span)
    call = ir.create_op_call(f"tile.{name}", [x], kwargs, span)
    func = ir.Function("math", [x], [call.type], ir.ReturnStmt([call], span), span, ir.FunctionType.InCore)
    return ir.Program([func], "math_test", span)


class _PrimitiveEvaluator:
    """Evaluate emitted primitive IR, not a duplicated composite recipe.

    This is deliberately limited to the straight-line vector primitives in
    these recipes. Memory layouts and target instructions are tested by the
    compiler tests; unsupported primitives fail loudly here.
    """

    def __init__(self, x):
        self.values = {"x": x}
        unary = {
            "exp": np.exp,
            "log": np.log,
            "recip": np.reciprocal,
        }
        binary = {
            "mul": np.multiply,
            "maximum": np.maximum,
            "minimum": np.minimum,
        }
        self.primitives = {ir.get_op(f"tile.{name}").name: fn for name, fn in unary.items()}
        for name, fn in binary.items():
            self.primitives[ir.get_op(f"tile.{name}").name] = fn
            self.primitives[ir.get_op(f"tile.{name}s").name] = fn
        self.primitives[ir.get_op("tile.create").name] = lambda shape: np.zeros(shape, np.float32)
        self.primitives[ir.get_op("tile.transpose_view").name] = lambda x: x.T
        self.primitives[ir.get_op("tile.set_validshape").name] = lambda x, rows, cols: x
        self.primitives[ir.get_op("tile.reshape").name] = np.reshape

    def expr(self, expr: ir.Expr) -> Any:
        if isinstance(expr, ir.Var):
            return self.values[expr.name_hint]
        if isinstance(expr, ir.ConstInt):
            return expr.value
        if isinstance(expr, ir.ConstFloat):
            return np.float32(expr.value)
        if isinstance(expr, ir.MakeTuple):
            return tuple(self.expr(item) for item in expr.elements)
        assert isinstance(expr, ir.Call), type(expr)
        args = [self.expr(arg) for arg in expr.args]
        if expr.op.name == ir.get_op("tile.col_sum").name:
            input_type = expr.args[0].type
            assert isinstance(input_type, ir.TileType)
            rows = input_type.get_effective_tile_view().valid_shape[0]
            assert isinstance(rows, ir.ConstInt)
            return np.sum(args[0][: rows.value], axis=0, keepdims=True)
        if expr.op.name == ir.get_op("tile.row_sum").name:
            input_type = expr.args[0].type
            assert isinstance(input_type, ir.TileType)
            columns = input_type.get_effective_tile_view().valid_shape[-1]
            assert isinstance(columns, ir.ConstInt)
            return np.sum(args[0][:, : columns.value], axis=-1, keepdims=True)
        if expr.op.name == ir.get_op("tile.fillpad_expand").name:
            input_type = expr.args[0].type
            assert isinstance(input_type, ir.TileType)
            valid = input_type.get_effective_tile_view().valid_shape
            assert all(isinstance(dim, ir.ConstInt) for dim in valid)
            region = tuple(slice(0, dim.value) for dim in valid if isinstance(dim, ir.ConstInt))
            result = np.zeros(args[1], dtype=args[0].dtype) if len(args) == 2 else np.zeros_like(args[0])
            result[region] = args[0][region]
            return result
        if expr.op.name == ir.get_op("tile.full").name:
            assert isinstance(expr.type, ir.TileType)
            dtype = {ir.DataType.FP16: np.float16, ir.DataType.FP32: np.float32, ir.DataType.INT32: np.int32}[
                expr.type.dtype
            ]
            return np.full(args[0], args[1], dtype=dtype)
        if expr.op.name == ir.get_op("tile.cast").name:
            attrs = dict(expr.kwargs)
            target_type = attrs["target_type"]
            assert isinstance(target_type, ir.DataType)
            dtype = {ir.DataType.FP32: np.float32, ir.DataType.FP16: np.float16, ir.DataType.INT32: np.int32}[
                target_type
            ]
            value = args[0]
            if dtype == np.int32:
                mode = attrs["mode"]
                assert isinstance(mode, int)
                value = {1: np.rint, 2: lambda x: np.sign(x) * np.floor(np.abs(x) + 0.5), 3: np.floor}[mode](
                    value
                )
            return value.astype(dtype)
        return self.primitives[expr.op.name](*args)

    def run(self, program: ir.Program, function: str) -> Any:
        func = program.get_function(function)
        assert func is not None
        return self.stmt(func.body)

    def stmt(self, stmt: ir.Stmt) -> Any:
        if isinstance(stmt, ir.SeqStmts):
            result = None
            for child in stmt.stmts:
                result = self.stmt(child)
            return result
        if isinstance(stmt, ir.AssignStmt):
            self.values[stmt.var.name_hint] = self.expr(stmt.value)
            return None
        assert isinstance(stmt, ir.ReturnStmt), type(stmt)
        return self.expr(stmt.value[0])


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("exponent", [-5, -2, -1, 0, 1, 2, 3, 4, 7, 0.5, -0.5, 1.25])
def test_pow_lowered_ir_matches_numpy(dtype, exponent):
    rng = np.random.default_rng(2665)
    x = rng.uniform(0.5, 2, size=(16, 32)).astype(dtype)
    if float(exponent).is_integer():
        x[:, ::2] *= -1
    if exponent == 0:
        x[0, :4] = [0, np.inf, -np.inf, np.nan]
    ir_dtype = ir.DataType.FP16 if dtype == np.float16 else ir.DataType.FP32
    before = _math_program("pow", x.shape, ir_dtype, valid_shape=[7, 19], exponent=float(exponent))
    after = passes.lower_composite_ops()(before)
    actual = _PrimitiveEvaluator(x).run(after, "math")
    expected = np.power(x.astype(np.float32), exponent).astype(dtype)
    np.testing.assert_allclose(actual[:7, :19], expected[:7, :19], rtol=2e-3, atol=2e-6)
    assert actual.dtype == dtype
    ir.assert_structural_equal(passes.lower_composite_ops()(after), after)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_pow_zero_preserves_nd_partial_valid_region(dtype):
    x = np.full((2, 16, 32), np.nan, dtype=dtype)
    ir_dtype = ir.DataType.FP16 if dtype == np.float16 else ir.DataType.FP32
    before = _math_program("pow", x.shape, ir_dtype, valid_shape=[1, 16, 19], exponent=0.0)
    after = passes.lower_composite_ops()(before)
    actual = _PrimitiveEvaluator(x).run(after, "math")
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual[0, :, :19], 1)
    func = after.get_function("math")
    assert func is not None and isinstance(func.body, ir.SeqStmts)
    for stmt in func.body.stmts:
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.value, ir.Call):
            assert stmt.value.op.name != ir.get_op("tile.slice").name
    result = func.body.stmts[-1]
    assert isinstance(result, ir.ReturnStmt)
    result_type = result.value[0].type
    assert isinstance(result_type, ir.TileType)
    for dimension, expected in zip(
        result_type.get_effective_tile_view().valid_shape, (1, 16, 19), strict=True
    ):
        assert isinstance(dimension, ir.ConstInt) and dimension.value == expected
    ir.assert_structural_equal(passes.lower_composite_ops()(after), after)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("axis", [0, 1, -1, -2])
def test_mean_lowered_ir_excludes_padding(dtype, axis):
    rng = np.random.default_rng(2665)
    x = rng.normal(size=(16, 32)).astype(dtype)
    x[7:, :] = 10000
    x[:, 19:] = 10000
    ir_dtype = ir.DataType.FP16 if dtype == np.float16 else ir.DataType.FP32
    before = _math_program("mean", x.shape, ir_dtype, valid_shape=[7, 19], axis=axis)
    after = passes.lower_composite_ops()(before)
    actual = _PrimitiveEvaluator(x).run(after, "math")
    expected = x[:7, :19].astype(np.float32).mean(axis=axis, keepdims=True).astype(dtype)
    np.testing.assert_allclose(
        actual[: expected.shape[0], : expected.shape[1]], expected, rtol=2e-3, atol=2e-6
    )
    assert actual.dtype == dtype
    ir.assert_structural_equal(passes.lower_composite_ops()(after), after)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize(
    "bounds", [{"min": -1.0}, {"max": 1.0}, {"min": -1.0, "max": 1.0}, {"min": 2.0, "max": -2.0}]
)
def test_clamp_lowered_ir_matches_numpy(dtype, bounds):
    x = np.linspace(-3, 3, 16 * 32).reshape(16, 32).astype(dtype)
    ir_dtype = ir.DataType.FP16 if dtype == np.float16 else ir.DataType.FP32
    before = _math_program("clamp", x.shape, ir_dtype, **bounds)
    after = passes.lower_composite_ops()(before)
    actual = _PrimitiveEvaluator(x).run(after, "math")
    expected = np.clip(x, bounds.get("min"), bounds.get("max"))
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype
    ir.assert_structural_equal(passes.lower_composite_ops()(after), after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
