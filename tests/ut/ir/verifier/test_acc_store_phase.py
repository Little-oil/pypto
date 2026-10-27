# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for the final accumulator/store phase-pairing verifier.

On A2/A3, ``acc_phase=pl.AccPhase.Final`` is producer-side check-and-set and
``st_phase=pl.STPhase.Final`` is consumer-side check-and-clear. Either missing side can
stall the device rather than producing a normal runtime error, so
``AccStorePhaseValid`` rejects the mismatch immediately after Inline helpers
are expanded, while the IR still carries the user's source spans.
"""

import pypto
import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import passes


def _verify(program):
    props = passes.IRPropertySet()
    props.insert(passes.IRProperty.AccStorePhaseValid)
    return passes.PropertyVerifierRegistry.verify(props, program)


def _gemv_program(
    *,
    producer_phase=pl.AccPhase.Final,
    store_phase=pl.STPhase.Final,
):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=producer_phase)
            result_alias = result
            out = pl.store(result_alias, [0, 0], out, st_phase=store_phase)

    return Prog


def _double_store_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
            out2: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
            out2 = pl.store(result, [0, 0], out2, st_phase=pl.STPhase.Final)

    return Prog


def _unconsumed_final_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            _ = result

    return Prog


def _gemv_acc_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 256], pl.FP32],
            b: pl.Tensor[[256, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs0 = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs0 = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.gemv(lhs0, rhs0, acc_phase=pl.AccPhase.Partial)
            lhs1 = pl.load(a, [0, 128], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs1 = pl.load(b, [128, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv_acc(acc, lhs1, rhs1, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _gemv_bias_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            bias: pl.Tensor[[1, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            bias_tile = pl.load(bias, [0, 0], [1, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv_bias(lhs, rhs, bias_tile, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _cross_if_boundary_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            cond: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            if cond < 1:
                out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _pair_inside_if_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            cond: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            if cond < 1:
                lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
                rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
                result = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
                out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def _inline_final_producer_program():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.Inline)
        def final_gemv(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
        ) -> pl.Tile[[16, 64], pl.FP32, pl.Mem.Acc, pl.TileView(valid_shape=[1, 64])]:
            lhs = pl.load(a, [0, 0], [1, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 64], target_memory=pl.MemorySpace.Mat)
            return pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)

        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[1, 128], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ):
            result = self.final_gemv(a, b)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    return Prog


def test_matching_final_gemv_and_store_are_accepted():
    assert _verify(_gemv_program()) == []


def test_all_phase_aware_gemv_variants_are_accepted():
    assert _verify(_gemv_acc_program()) == []
    assert _verify(_gemv_bias_program()) == []


def test_final_producer_with_plain_store_is_rejected():
    diagnostics = _verify(_gemv_program(store_phase=pl.STPhase.Unspecified))
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "st_phase=pl.STPhase.Unspecified" in diagnostics[0].message
    assert "st_phase=pl.STPhase.Final" in diagnostics[0].message


def test_invalid_store_phase_reports_a_diagnostic():
    """Malformed IR is diagnosed without formatting the invalid enum value."""

    class _StampInvalidStorePhase(ir.IRMutator):
        def visit_call(self, op):
            rewritten = super().visit_call(op)
            assert isinstance(rewritten, ir.Call)
            if rewritten.op.name != ir.get_op("tile.store").name:
                return rewritten
            kwargs = dict(rewritten.kwargs)
            kwargs["st_phase"] = 99
            return ir.Call(
                rewritten.op,
                list(rewritten.args),
                kwargs,
                dict(rewritten.attrs),
                rewritten.type,
                rewritten.span,
            )

    malformed = _StampInvalidStorePhase().visit_program(_gemv_program())
    diagnostics = _verify(malformed)

    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "must encode pl.STPhase.Unspecified (0) or pl.STPhase.Final (3)" in diagnostics[0].message
    assert "got 99" in diagnostics[0].message


def test_final_store_without_final_producer_is_rejected():
    diagnostics = _verify(_gemv_program(producer_phase=pl.AccPhase.Unspecified))
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "unit flag that was never set" in diagnostics[0].message


def test_unconsumed_final_producer_is_rejected():
    diagnostics = _verify(_unconsumed_final_program())
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "must be consumed exactly once" in diagnostics[0].message


def test_second_final_store_of_same_value_is_rejected():
    diagnostics = _verify(_double_store_program())
    assert len(diagnostics) == 1
    assert diagnostics[0].rule_name == "AccStorePhaseValid"
    assert "does not have a live matching" in diagnostics[0].message


def test_partial_producer_and_plain_store_remain_legal():
    assert (
        _verify(_gemv_program(producer_phase=pl.AccPhase.Partial, store_phase=pl.STPhase.Unspecified)) == []
    )


def test_pair_may_not_cross_a_control_flow_boundary():
    diagnostics = _verify(_cross_if_boundary_program())
    assert diagnostics
    assert all(diag.rule_name == "AccStorePhaseValid" for diag in diagnostics)
    assert any("before entering an if statement" in diag.message for diag in diagnostics)


def test_balanced_pair_inside_a_branch_is_accepted():
    assert _verify(_pair_inside_if_program()) == []


def test_phase_property_is_produced_after_inline_not_structural():
    assert not passes.get_structural_properties().contains(passes.IRProperty.AccStorePhaseValid)
    assert passes.inline_functions().get_produced_properties().contains(passes.IRProperty.AccStorePhaseValid)


def test_default_pipeline_accepts_pair_joined_by_inline_expansion():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        PassManager.get_strategy(OptimizationStrategy.Default).run_passes(_inline_final_producer_program())
    finally:
        backend.reset_for_testing()


def test_default_pipeline_rejects_mismatch_after_inline_functions():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    try:
        with pytest.raises(pypto.Error, match="AccStorePhaseValid"):
            PassManager.get_strategy(OptimizationStrategy.Default).run_passes(
                _gemv_program(store_phase=pl.STPhase.Unspecified)
            )
    finally:
        backend.reset_for_testing()


def _matmul_program(*, producer_phase=pl.AccPhase.Final, store_phase=pl.STPhase.Final):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=producer_phase)
            result_alias = result
            out = pl.store(result_alias, [0, 0], out, st_phase=store_phase)

    return Prog


def _matmul_acc_program(*, producer_phase=pl.AccPhase.Final, store_phase=pl.STPhase.Final):
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            acc = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
            result = pl.tile.matmul_acc(acc, lhs, rhs, acc_phase=producer_phase)
            out = pl.store(result, [0, 0], out, st_phase=store_phase)

    return Prog


@pytest.mark.parametrize("make_program", [_matmul_program, _matmul_acc_program])
def test_matmul_final_pair_and_partial_plain_store_are_accepted(make_program):
    assert _verify(make_program()) == []
    assert _verify(make_program(producer_phase=pl.AccPhase.Partial, store_phase=pl.STPhase.Unspecified)) == []


@pytest.mark.parametrize("make_program", [_matmul_program, _matmul_acc_program])
@pytest.mark.parametrize("producer_phase", [pl.AccPhase.Unspecified, pl.AccPhase.Partial])
def test_matmul_final_store_requires_final_producer(make_program, producer_phase):
    diagnostics = _verify(make_program(producer_phase=producer_phase))
    assert len(diagnostics) == 1
    assert "unit flag that was never set" in diagnostics[0].message


@pytest.mark.parametrize("make_program", [_matmul_program, _matmul_acc_program])
def test_matmul_final_producer_requires_final_store(make_program):
    diagnostics = _verify(make_program(store_phase=pl.STPhase.Unspecified))
    assert len(diagnostics) == 1
    assert "Use st_phase=pl.STPhase.Final" in diagnostics[0].message


def test_matmul_final_value_cannot_be_drained_twice():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    diagnostics = _verify(Prog)
    assert len(diagnostics) == 1
    assert "does not have a live matching" in diagnostics[0].message


def test_matmul_pair_cannot_cross_a_conditional_store():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            cond: pl.Scalar[pl.INDEX],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            if cond > 0:
                out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    diagnostics = _verify(Prog)
    assert any("before entering an if statement" in d.message for d in diagnostics)


def test_matmul_balanced_pairs_inside_loop_and_branch_are_accepted():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            cond: pl.Scalar[pl.INDEX],
        ):
            for i in pl.range(3):
                if i < cond:
                    lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
                    rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
                    result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
                    out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    assert _verify(Prog) == []


@pytest.mark.parametrize("drain_first", [False, True])
def test_matmul_acc_must_not_modify_a_final_value_before_its_store(drain_first):
    # Both programs have a balanced Final/store count. Only the ordering tells
    # whether the in-place accumulator update waits on an uncleared unit flag.
    @pl.program
    class BeforeDrain:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            alias = result
            pl.tile.matmul_acc(alias, lhs, rhs, acc_phase=pl.AccPhase.Partial)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    @pl.program
    class AfterDrain:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)
            pl.tile.matmul_acc(result, lhs, rhs, acc_phase=pl.AccPhase.Partial)

    diagnostics = _verify(AfterDrain if drain_first else BeforeDrain)
    if drain_first:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert "undrained final accumulator producer" in diagnostics[0].message


def test_view_cannot_hide_modification_of_an_undrained_final_accumulator():
    @pl.program
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[64, 128], pl.BF16],
            b: pl.Tensor[[128, 128], pl.BF16],
            out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.MemorySpace.Mat)
            rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat)
            result = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
            # This full-window slice aliases result's physical Acc. Draining
            # the old SSA value later cannot make this intervening update safe.
            view = pl.tile.slice(result, [64, 128], [0, 0])
            pl.tile.matmul_acc(view, lhs, rhs, acc_phase=pl.AccPhase.Partial)
            out = pl.store(result, [0, 0], out, st_phase=pl.STPhase.Final)

    diagnostics = _verify(Prog)
    assert len(diagnostics) == 1
    assert "input to 'tile.slice'" in diagnostics[0].message
    assert "undrained final accumulator producer" in diagnostics[0].message


@pytest.mark.parametrize("alias_expr", ["seed", "pl.tile.slice(seed, [64, 128], [0, 0])"])
@pytest.mark.parametrize("drain_first", [False, True])
def test_alias_created_before_inplace_final_cannot_bypass_drain(alias_expr, drain_first):
    """The earlier SSA name still addresses the storage written by Final."""
    drain = "out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)"
    update = "pl.tile.matmul_acc(earlier, lhs, rhs, acc_phase=pl.AccPhase.Partial)"
    statements = [drain, update] if drain_first else [update, drain]
    tail = "\n".join(f"        {statement}" for statement in statements)
    program = pl.parse_program(f"""
@pl.program
class EarlierAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        earlier = {alias_expr}
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
{tail}
""")
    diagnostics = _verify(program)
    if drain_first:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert "undrained final accumulator producer" in diagnostics[0].message


def test_original_acc_input_cannot_be_used_before_final_drain():
    program = pl.parse_program("""
@pl.program
class OriginalInput:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
        pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "undrained final accumulator producer" in diagnostics[0].message


def test_original_acc_input_cannot_be_plain_stored_before_final_drain():
    program = pl.parse_program("""
@pl.program
class OriginalStore:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
        out = pl.store(seed, [0, 0], out)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "input to 'tile.store'" in diagnostics[0].message


def test_storage_alias_does_not_replace_exact_final_result_for_store():
    program = pl.parse_program("""
@pl.program
class WrongValue:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
        out = pl.store(seed, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert any("must consume the exact SSA value" in d.message for d in diagnostics)
    assert any("must be consumed exactly once" in d.message for d in diagnostics)


def test_fresh_rebinding_is_distinct_from_older_final_storage():
    program = pl.parse_program("""
@pl.program
class FreshRebind:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    assert _verify(program) == []


def test_loop_carried_accumulator_preserves_earlier_alias_provenance():
    program = pl.parse_program("""
@pl.program
class LoopAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        earlier = seed
        for step, (carried,) in pl.range(2, init_values=(seed,)):
            next_acc = pl.tile.matmul_acc(carried, lhs, rhs, acc_phase=pl.AccPhase.Partial)
            reduced = pl.yield_(next_acc)
        final = pl.tile.matmul_acc(reduced, lhs, rhs, acc_phase=pl.AccPhase.Final)
        pl.tile.matmul_acc(earlier, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "undrained final accumulator producer" in diagnostics[0].message


def test_branch_result_retains_possible_input_storage_aliases():
    program = pl.parse_program("""
@pl.program
class BranchAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]], choose_first: pl.Scalar[pl.BOOL]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed0 = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        seed1 = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        if choose_first:
            selected = pl.yield_(seed0)
        else:
            selected = pl.yield_(seed1)
        final = pl.tile.matmul_acc(selected, lhs, rhs, acc_phase=pl.AccPhase.Final)
        pl.tile.matmul_acc(seed0, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "undrained final accumulator producer" in diagnostics[0].message


def test_loop_yields_account_for_aliasing_on_later_iterations():
    # The two carries are initially distinct. The first yield makes them alias,
    # so checking only the first iteration would accept a later wait-before-clear.
    program = pl.parse_program("""
@pl.program
class LaterIterationAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed0 = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        seed1 = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        for step, (first, second) in pl.range(2, init_values=(seed0, seed1)):
            final = pl.tile.matmul_acc(first, lhs, rhs, acc_phase=pl.AccPhase.Final)
            pl.tile.matmul_acc(second, lhs, rhs, acc_phase=pl.AccPhase.Partial)
            out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
            next_first, next_second = pl.yield_(second, second)
""")
    diagnostics = _verify(program)
    assert len(diagnostics) == 1
    assert "undrained final accumulator producer" in diagnostics[0].message


@pytest.mark.parametrize(
    "aliases",
    [
        "pair = (seed, seed)\n        earlier = pair[0]",
        "pair = (seed, (seed, seed))\n        forwarded = pair\n"
        "        nested = forwarded[1]\n        earlier = nested[0]",
    ],
    ids=["tuple", "nested_tuple_alias"],
)
@pytest.mark.parametrize("drain_first", [False, True])
def test_tuple_projection_created_before_final_retains_storage(aliases, drain_first):
    drain = "out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)"
    update = "pl.tile.matmul_acc(earlier, lhs, rhs, acc_phase=pl.AccPhase.Partial)"
    statements = [drain, update] if drain_first else [update, drain]
    tail = "\n".join(f"        {statement}" for statement in statements)
    program = pl.parse_program(f"""
@pl.program
class TupleAlias:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        {aliases}
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.Final)
{tail}
""")
    diagnostics = _verify(program)
    if drain_first:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert "undrained final accumulator producer" in diagnostics[0].message


@pytest.mark.parametrize("final_phase", [False, True])
def test_unresolved_tile_tuple_carry_is_rejected_only_with_final_protocol(final_phase):
    phase = "Final" if final_phase else "Partial"
    store_phase = "Final" if final_phase else "Unspecified"
    program = pl.parse_program(f"""
@pl.program
class TupleCarry:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        pair = (seed, seed)
        for step, (carried,) in pl.range(2, init_values=(pair,)):
            result_pair = pl.yield_(carried)
        earlier = result_pair[0]
        final = pl.tile.matmul_acc(seed, lhs, rhs, acc_phase=pl.AccPhase.{phase})
        pl.tile.matmul_acc(earlier, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.{store_phase})
""")
    diagnostics = _verify(program)
    if final_phase:
        assert len(diagnostics) == 1
        assert "unresolved tile-bearing tuple provenance" in diagnostics[0].message
        assert "Unpack the tiles" in diagnostics[0].message
    else:
        assert diagnostics == []


def test_scalar_tuple_carry_does_not_restrict_final_accumulator_protocol():
    program = pl.parse_program("""
@pl.program
class ScalarTupleCarry:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        pair = (1, 2)
        for step, (carried,) in pl.range(2, init_values=(pair,)):
            result_pair = pl.yield_(carried)
        final = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    assert _verify(program) == []


def test_registered_fresh_multioutput_tuple_does_not_restrict_final_protocol():
    """Fresh builtin outputs cannot alias an unrelated Cube accumulator."""
    program = pl.parse_program("""
@pl.program
class FreshTupleOutputs:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               q: pl.Tensor[[64, 128], pl.FP16], out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        qvec = pl.load(q, [0, 0], [64, 128], target_memory=pl.Mem.Vec)
        quantized, scales = pl.tile.quant_mx(qvec, group_axis=1)
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        final = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
""")
    assert _verify(program) == []


def test_tuple_projection_cannot_replace_exact_final_value_for_store():
    program = pl.parse_program("""
@pl.program
class FinalTupleProjection:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        final = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)
        pair = (final, final)
        out = pl.store(pair[0], [0, 0], out, st_phase=pl.STPhase.Final)
""")
    diagnostics = _verify(program)
    assert any("must consume the exact SSA value" in d.message for d in diagnostics)


@pytest.mark.parametrize("region", ["for step in pl.range(2):", "if choose_first:"])
@pytest.mark.parametrize("producer_count", [2, 3])
@pytest.mark.parametrize("newest_first", [False, True])
@pytest.mark.parametrize("drain_all_before_use", [False, True])
def test_draining_merged_storage_preserves_other_final_tokens(
    region, producer_count, newest_first, drain_all_before_use
):
    # Pre-SSA rebindings inside a region conservatively merge fresh storage
    # identities. Each exact Final value still has its own drain obligation;
    # draining one must not hide an unsafe read of another in either order.
    statements = []
    for index in range(producer_count):
        statements.extend(
            [
                "c = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.Final)",
                f"held{index} = c",
            ]
        )
    order = list(range(producer_count))
    if newest_first:
        order.reverse()
    stores = [f"out = pl.store(held{i}, [0, 0], out, st_phase=pl.STPhase.Final)" for i in order]
    update = f"pl.tile.matmul_acc(held{order[-1]}, lhs, rhs, acc_phase=pl.AccPhase.Partial)"
    if drain_all_before_use:
        statements.extend([*stores, update])
    else:
        statements.extend([stores[0], update, *stores[1:]])
    # After the last drain, the same abstract storage can begin a fresh pair.
    statements.extend(
        [
            f"next_final = pl.tile.matmul_acc(held{order[-1]}, lhs, rhs, acc_phase=pl.AccPhase.Final)",
            "out = pl.store(next_final, [0, 0], out, st_phase=pl.STPhase.Final)",
        ]
    )
    body = "\n".join(f"            {statement}" for statement in statements)
    program = pl.parse_program(f"""
@pl.program
class MergedFinalTokens:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]], choose_first: pl.Scalar[pl.BOOL]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        {region}
{body}
""")
    diagnostics = _verify(program)
    if drain_all_before_use:
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert "undrained final accumulator producer" in diagnostics[0].message


@pytest.mark.parametrize("producer", ["matmul", "gemv"])
@pytest.mark.parametrize("chained_choice", [False, True])
@pytest.mark.parametrize("access", ["independent", "selected_consumer", "selected_producer"])
def test_branch_choices_preserve_independent_accumulators(producer, chained_choice, access):
    # A selection may refer to either source; it does not make the two source
    # allocations alias. Reads/writes through the selection still need a guard.
    rows, dtype = (64, "BF16") if producer == "matmul" else (1, "FP32")
    extra = (
        """
        if choose_first:
            selected_again = pl.yield_(selected)
        else:
            selected_again = pl.yield_(seed1)
        selected = selected_again
"""
        if chained_choice
        else ""
    )
    acc = f"pl.tile.{producer}_acc"
    if access == "independent":
        body = f"""
        final0 = {acc}(seed0, lhs, rhs, acc_phase=pl.AccPhase.Final)
        final1 = {acc}(seed1, lhs, rhs, acc_phase=pl.AccPhase.Final)
        out = pl.store(final0, [0, 0], out, st_phase=pl.STPhase.Final)
        out = pl.store(final1, [0, 0], out, st_phase=pl.STPhase.Final)
"""
    else:
        final_source = "selected" if access == "selected_producer" else "seed0"
        consumer_source = "seed0" if access == "selected_producer" else "selected"
        body = f"""
        final = {acc}({final_source}, lhs, rhs, acc_phase=pl.AccPhase.Final)
        {acc}({consumer_source}, lhs, rhs, acc_phase=pl.AccPhase.Partial)
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.Final)
"""
    program = pl.parse_program(f"""
@pl.program
class IndependentChoices:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[{rows}, 128], pl.{dtype}], b: pl.Tensor[[128, 128], pl.{dtype}],
               out: pl.Out[pl.Tensor[[{rows}, 128], pl.FP32]],
               tmp: pl.Out[pl.Tensor[[{rows}, 128], pl.FP32]], choose_first: pl.Scalar[pl.BOOL]):
        lhs = pl.load(a, [0, 0], [{rows}, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        seed0 = pl.tile.{producer}(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        seed1 = pl.tile.{producer}(lhs, rhs, acc_phase=pl.AccPhase.Partial)
        if choose_first:
            selected = pl.yield_(seed0)
        else:
            selected = pl.yield_(seed1)
{extra}
        tmp = pl.store(selected, [0, 0], tmp)
{body}
""")
    diagnostics = _verify(program)
    if access == "independent":
        assert diagnostics == []
    else:
        assert len(diagnostics) == 1
        assert "undrained final accumulator producer" in diagnostics[0].message


@pytest.mark.parametrize("space", ["Mat", "Vec", "Acc", None])
@pytest.mark.parametrize("tuple_flow", ["loop", "rebind"])
@pytest.mark.parametrize("final_phase", [False, True])
def test_unrelated_tuple_provenance_only_restricts_possible_acc_storage(space, tuple_flow, final_phase):
    # Known non-Acc tiles cannot share the producer's unit flags. Acc tuple
    # control provenance remains conservative, and default kernels stay legal.
    flow = (
        """
        for step, (carried,) in pl.range(2, init_values=(pair,)):
            ignored = pl.yield_(carried)
"""
        if tuple_flow == "loop"
        else """
        if choose_first:
            pair = (unrelated, (unrelated, 1))
"""
    )
    phase = "Final" if final_phase else "Unspecified"
    allocation = (
        f"pl.tile.create([16, 16], pl.FP32, target_memory=pl.Mem.{space})" if space else "unknown_space"
    )
    program = pl.parse_program(f"""
@pl.program
class UnrelatedTuple:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]], choose_first: pl.Scalar[pl.BOOL],
               unknown_space: pl.Tile[[16, 16], pl.FP32]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        unrelated = {allocation}
        pair = (unrelated, (unrelated, 1))
{flow}
        final = pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.{phase})
        out = pl.store(final, [0, 0], out, st_phase=pl.STPhase.{phase})
""")
    diagnostics = _verify(program)
    if final_phase and space in ("Acc", None):
        assert len(diagnostics) == 1
        assert "unresolved tile-bearing tuple provenance" in diagnostics[0].message
    else:
        assert diagnostics == []


@pytest.mark.parametrize(
    "producer_phase, store_phase",
    [("Unspecified", "Unspecified"), ("Unspecified", "Final"), ("Final", "Final")],
)
def test_lazy_storage_index_keeps_nested_producer_and_orphan_store_checks(producer_phase, store_phase):
    # The first Final may be nested in a store argument. Without any Final
    # producer, the verifier must still reject a store waiting on an unset flag.
    program = pl.parse_program(f"""
@pl.program
class NestedFirstProducer:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, a: pl.Tensor[[64, 128], pl.BF16], b: pl.Tensor[[128, 128], pl.BF16],
               out: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
        lhs = pl.load(a, [0, 0], [64, 128], target_memory=pl.Mem.Mat)
        rhs = pl.load(b, [0, 0], [128, 128], target_memory=pl.Mem.Mat)
        out = pl.store(pl.tile.matmul(lhs, rhs, acc_phase=pl.AccPhase.{producer_phase}),
                       [0, 0], out, st_phase=pl.STPhase.{store_phase})
""")
    diagnostics = _verify(program)
    if producer_phase == "Unspecified" and store_phase == "Final":
        assert len(diagnostics) == 1
        assert "must consume the exact SSA value" in diagnostics[0].message
    else:
        assert diagnostics == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
