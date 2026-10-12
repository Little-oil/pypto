# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exact user-selected L2 hints reach ISA on A2/A3 and A5 with stock PTOAS."""

import pypto.language as pl
import pytest
from pypto import backend, codegen, ir
from pypto.backend import BackendType, pto_backend
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir import OptimizationStrategy, PassManager
from pypto.pypto_core import passes

M, K, N = 256, 128, 256
ROWS, COLS = 32, 128


@pytest.fixture(params=[BackendType.Ascend910B, BackendType.Ascend950])
def target(request):
    return request.param


@pytest.fixture(autouse=True)
def _setup_backend(target):
    backend.reset_for_testing()
    backend.set_backend_type(target)
    yield
    backend.reset_for_testing()


@pl.program
class DeclaredHint:
    """Scope-level declaration on one of the two matmul operands."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class PlainMatmul:
    """The same kernel with no declaration — the comparison reference."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class TwoDeclaredTensors:
    """Both declared operands retain their selected load hint."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm2"):
            pl.set_l2_hint(a, load=pl.LoadL2Hint.NotAllocKeep)
            pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocDrop)
            c: pl.Tensor[[M, N], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
            out = pl.assemble(out, c, [0, 0])
        return out


@pl.program
class TwoHintedLoadsOfOneTensor:
    """Two reads of one tensor retain different user-selected NotAlloc enums."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        top: pl.Tile[[16, COLS], pl.FP32] = pl.load(x, [0, 0], [16, COLS], l2_hint=pl.LoadL2Hint.NotAllocKeep)
        bottom: pl.Tile[[16, COLS], pl.FP32] = pl.load(
            x, [16, 0], [16, COLS], l2_hint=pl.LoadL2Hint.NotAllocDrop
        )
        out_0: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(top, [0, 0], out)
        out_1: pl.Tensor[[ROWS, COLS], pl.FP32] = pl.store(bottom, [16, 0], out_0)
        return out_1


@pl.program
class ExplicitDefaultLoad:
    """A declared parameter with one access explicitly opted back into the cache.

    The declaration is written the way pass 9 leaves it — ``load_l2_hint`` on the
    outlined kernel, ``(param index, hint)`` — because the override has to be
    stated at the access, and a hand-written InCore kernel is where an access is
    spelled out. See the pass-level pair in
    ``tests/ut/ir/transforms/test_convert_tensor_to_tile_ops.py``.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        x: pl.Tensor[[ROWS, COLS], pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ) -> pl.Tensor[[ROWS, COLS], pl.FP32]:
        # (param index, hint-as-int); 5 is LoadL2Hint.NotAllocClean. The parser takes
        # integer literals here only, which is also how pass 9 writes the attr.
        pl.func_attr({"load_l2_hint": [(0, 5)]})
        t: pl.Tile[[16, COLS], pl.FP32] = pl.load(
            x, [0, 0], [16, COLS], l2_hint=pl.LoadL2Hint.NormalFirstVictim
        )
        return pl.store(t, [0, 0], out)


def _incore_functions(program_cls, planner=passes.MemoryPlanner.PYPTO):
    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program_cls)
    functions = [f for f in optimized.functions.values() if f.func_type != pl.FunctionType.Orchestration]
    assert len(functions) == 1, f"expected one in-core function, got {[f.name for f in functions]}"
    return functions, ir.Program(functions, functions[0].name, optimized.span)


def _incore_mlir(program_cls, planner=passes.MemoryPlanner.PYPTO):
    _, module = _incore_functions(program_cls, planner)
    mlir = codegen.PTOCodegen().generate(
        module, emit_source_loc=False, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS
    )
    assert isinstance(mlir, str)
    return mlir


def _assert_no_legacy_offset(*sources):
    for source in sources:
        assert "cache_policy" not in source
        assert "__pypto_l2_cache_offset" not in source
        assert "get_l2_cache_offset" not in source


def _compile_with_stock_ptoas(program_cls, tmp_path, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    functions, module = _incore_functions(program_cls, planner)
    mlir = codegen.PTOCodegen().generate(
        module, emit_source_loc=False, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS
    )
    assert isinstance(mlir, str)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_transfers", str(tmp_path), planner)
    wrapper = pto_backend._generate_kernel_wrapper(functions[0], cpp)
    _assert_no_legacy_offset(mlir, cpp, wrapper)
    assert "__pypto_l2_hint" not in cpp
    assert (tmp_path / "ptoas" / "hinted_transfers.cpp").read_text() == cpp
    return mlir, cpp


def _tload_lines(mlir):
    return [line.strip() for line in mlir.splitlines() if "pto.tload" in line]


def test_scope_declaration_adds_only_the_selected_hint():
    declared = _incore_mlir(DeclaredHint)
    plain = _incore_mlir(PlainMatmul)
    assert len(_tload_lines(declared)) == 2
    assert declared.count("__pypto_l2_hint_begin load NotAllocClean") == 1
    stripped = "\n".join(line for line in declared.split("\n") if "__pypto_l2_hint" not in line)
    assert stripped == plain
    _assert_no_legacy_offset(declared, plain)


def test_each_declared_tensor_gets_its_own_hinted_load():
    mlir = _incore_mlir(TwoDeclaredTensors)
    assert len(_tload_lines(mlir)) == 2
    assert mlir.count("__pypto_l2_hint_begin load NotAllocKeep") == 1
    assert mlir.count("__pypto_l2_hint_begin load NotAllocDrop") == 1
    _assert_no_legacy_offset(mlir)


def test_each_access_keeps_its_selected_hint_without_a_synthetic_parameter():
    mlir = _incore_mlir(TwoHintedLoadsOfOneTensor)
    assert len(_tload_lines(mlir)) == 2
    assert mlir.count("__pypto_l2_hint_begin load NotAllocKeep") == 1
    assert mlir.count("__pypto_l2_hint_begin load NotAllocDrop") == 1
    signature = next(line for line in mlir.splitlines() if line.strip().startswith("func.func @"))
    assert signature.count("!pto.ptr") == 2
    _assert_no_legacy_offset(mlir)


def test_ordinary_load_emits_no_hint():
    mlir = _incore_mlir(PlainMatmul)
    assert len(_tload_lines(mlir)) == 2
    assert "__pypto_l2_hint" not in mlir
    _assert_no_legacy_offset(mlir)


def test_explicit_default_overrides_nonallocating_scope_hint():
    mlir = _incore_mlir(ExplicitDefaultLoad)
    assert mlir.count("__pypto_l2_hint_begin load NormalFirstVictim") == 1
    assert "__pypto_l2_hint_begin load NotAlloc" not in mlir
    _assert_no_legacy_offset(mlir)


@pytest.mark.parametrize("load_hint", list(pl.LoadL2Hint))
def test_explicit_load_hint_is_not_silently_dropped(load_hint):
    @pl.program
    class HintedLoad:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile = pl.load(x, [0, 0], [32, 32], l2_hint=load_hint)
            return pl.store(tile, [0, 0], out)

    mlir = _incore_mlir(HintedLoad)
    assert mlir.count(f"__pypto_l2_hint_begin load {load_hint.name}") == 1
    _assert_no_legacy_offset(mlir)


@pytest.mark.parametrize("store_hint", list(pl.StoreL2Hint))
def test_explicit_store_hint_is_not_silently_dropped(store_hint):
    @pl.program
    class HintedStore:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile = pl.load(x, [0, 0], [32, 32])
            return pl.store(tile, [0, 0], out, l2_hint=store_hint)

    mlir = _incore_mlir(HintedStore)
    assert mlir.count(f"__pypto_l2_hint_begin store {store_hint.name}") == 1
    _assert_no_legacy_offset(mlir)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("load_hint", list(pl.LoadL2Hint))
@pytest.mark.parametrize("store_hint", list(pl.StoreL2Hint))
def test_l2_hints_reach_isa_with_stock_ptoas(tmp_path, planner, load_hint, store_hint):
    @pl.program
    class HintedTransfers:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile = pl.load(x, [0, 0], [32, 32], l2_hint=load_hint)
            return pl.store(tile, [0, 0], out, l2_hint=store_hint)

    _, cpp = _compile_with_stock_ptoas(HintedTransfers, tmp_path, planner)
    assert cpp.count(f"TLOAD<pto::TLoadL2Hint::{load_hint.name}>") == 1
    assert cpp.count(f"TSTORE<pto::TStoreL2Hint::{store_hint.name}>") == 1
    assert "TLOAD(" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_distinct_hints_on_one_tensor_reach_stock_ptoas(tmp_path, planner):
    _, cpp = _compile_with_stock_ptoas(TwoHintedLoadsOfOneTensor, tmp_path, planner)
    assert cpp.count("TLOAD<pto::TLoadL2Hint::NotAllocKeep>") == 1
    assert cpp.count("TLOAD<pto::TLoadL2Hint::NotAllocDrop>") == 1
    assert "NotAllocClean" not in cpp
    assert "TLOAD(" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_ordinary_transfers_remain_ordinary_with_stock_ptoas(tmp_path, planner):
    @pl.program
    class OrdinaryTransfers:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            tile = pl.load(x, [0, 0], [32, 32])
            return pl.store(tile, [0, 0], out)

    mlir, cpp = _compile_with_stock_ptoas(OrdinaryTransfers, tmp_path, planner)
    assert "__pypto_l2_hint" not in mlir
    assert cpp.count("TLOAD(") == 1
    assert cpp.count("TSTORE(") == 1
    assert "TLoadL2Hint" not in cpp
    assert "TStoreL2Hint" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_l2_hints_stay_with_transfers_in_control_flow(tmp_path, planner):
    @pl.program
    class HintedControlFlow:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[64, 32], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 32], pl.FP32]],
            flag: pl.Scalar[pl.INT32],
        ) -> pl.Tensor[[64, 32], pl.FP32]:
            for i, (out_iter,) in pl.range(2, init_values=(out,)):
                offset = i * 32
                if flag == 0:
                    tile = pl.load(x, [offset, 0], [32, 32], l2_hint=pl.LoadL2Hint.NotAllocDrop)
                else:
                    tile = pl.load(x, [offset, 0], [32, 32])
                updated = pl.store(tile, [offset, 0], out_iter, l2_hint=pl.StoreL2Hint.NotAllocClean)
                result = pl.yield_(updated)
            return result

    _, cpp = _compile_with_stock_ptoas(HintedControlFlow, tmp_path, planner)
    assert cpp.count("TLOAD<pto::TLoadL2Hint::NotAllocDrop>") == 1
    assert cpp.count("TLOAD(") == 1
    assert cpp.count("TSTORE<pto::TStoreL2Hint::NotAllocClean>") == 1


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
def test_explicit_default_hints_override_scope_hints_with_stock_ptoas(tmp_path, planner):
    @pl.program
    class ExplicitDefaults:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            pl.func_attr({"load_l2_hint": [(0, 6)], "store_l2_hint": [(1, 4)]})
            tile = pl.load(x, [0, 0], [32, 32], l2_hint=pl.LoadL2Hint.NormalFirstVictim)
            return pl.store(tile, [0, 0], out, l2_hint=pl.StoreL2Hint.NormalFirstVictim)

    _, cpp = _compile_with_stock_ptoas(ExplicitDefaults, tmp_path, planner)
    assert cpp.count("TLOAD<pto::TLoadL2Hint::NormalFirstVictim>") == 1
    assert cpp.count("TSTORE<pto::TStoreL2Hint::NormalFirstVictim>") == 1
    assert "NotAlloc" not in cpp


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("load_hint", list(pl.LoadL2Hint))
def test_scope_hints_reach_stock_ptoas_without_runtime_offset(tmp_path, planner, load_hint):
    store_hint = list(pl.StoreL2Hint)[list(pl.LoadL2Hint).index(load_hint) % len(pl.StoreL2Hint)]

    @pl.program
    class ScopeHints:
        @pl.function
        def main(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.set_l2_hint(x, load=load_hint)
                pl.set_l2_hint(out, store=store_hint)
                out = pl.assemble(out, pl.add(x, 2.0), [0, 0])
            return out

    _, cpp = _compile_with_stock_ptoas(ScopeHints, tmp_path, planner)
    assert cpp.count(f"TLOAD<pto::TLoadL2Hint::{load_hint.name}>") == 1
    assert cpp.count(f"TSTORE<pto::TStoreL2Hint::{store_hint.name}>") == 1


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("target", [BackendType.Ascend950], indirect=True)
@pytest.mark.parametrize("with_relu", [False, True])
def test_store_l2_hint_preserves_epilogue_with_stock_ptoas(tmp_path, planner, with_relu):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class HintedWriteback:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[16, 64], pl.INT8],
            y: pl.Tensor[[32, 64], pl.INT8],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
        ) -> pl.Tensor[[16, 32], pl.FP16]:
            lhs = pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Mat)
            rhs = pl.load(y, [0, 0], [32, 64], target_memory=pl.Mem.Mat)
            acc = pl.tile.matmul(
                pl.tile.move(lhs, target_memory=pl.Mem.Left),
                pl.tile.move(pl.tile.transpose_view(rhs), target_memory=pl.Mem.Right),
            )
            return pl.store(
                acc,
                [0, 0],
                out,
                atomic=pl.AtomicType.Add,
                pre_quant=1.0 / 1024,
                pre_relu=with_relu,
                l2_hint=pl.StoreL2Hint.NormalPersistent,
            )

    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedWriteback)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_writeback", str(tmp_path), planner)
    store = next(line for line in cpp.splitlines() if "TSTORE<" in line)
    assert "TSTORE<pto::TStoreL2Hint::NormalPersistent," in store
    assert "AtomicType::AtomicAdd" in store
    assert ("ReluPreMode::NormalRelu" in store) == with_relu
    assert store.rsplit(">(", 1)[1].count(",") == 2  # Destination, accumulator, packed scale.
    assert "981467136" in cpp  # FP32 bits of 1 / 1024.
    assert "__pypto_l2_hint" not in cpp
    _assert_no_legacy_offset(mlir, cpp)


@pytest.mark.parametrize("planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.PTOAS])
@pytest.mark.parametrize("target", [BackendType.Ascend910B], indirect=True)
def test_store_l2_hint_preserves_phase_with_stock_ptoas(tmp_path, planner):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class HintedPhase:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[1, 128], pl.FP32],
            y: pl.Tensor[[128, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[1, 64], pl.FP32]],
        ) -> pl.Tensor[[1, 64], pl.FP32]:
            lhs = pl.load(x, [0, 0], [1, 128], target_memory=pl.Mem.Mat)
            rhs = pl.load(y, [0, 0], [128, 64], target_memory=pl.Mem.Mat)
            acc = pl.tile.gemv(lhs, rhs, acc_phase=pl.AccPhase.Final)
            return pl.store(
                acc, [0, 0], out, st_phase=pl.STPhase.Final, l2_hint=pl.StoreL2Hint.NormalPersistent
            )

    with passes.PassContext([], memory_planner=planner):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(HintedPhase)
    mlir = codegen.PTOCodegen().generate(optimized, emit_tile_addr=planner != passes.MemoryPlanner.PTOAS)
    cpp = pto_backend._compile_pto_module(mlir, "hinted_phase", str(tmp_path), planner)
    store = next(line for line in cpp.splitlines() if "TSTORE<" in line)
    assert "TSTORE<pto::TStoreL2Hint::NormalPersistent, STPhase::Final>" in store
    assert "__pypto_l2_hint" not in cpp
    _assert_no_legacy_offset(mlir, cpp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
