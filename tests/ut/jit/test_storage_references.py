# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile the original public current/next and independently live Mat phi examples."""

import re
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.runtime import ExecutionMode, RunConfig

pytestmark = pytest.mark.usefixtures("ascend_backend")


@pl.jit.incore
def rotate(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    current = pl.load(a, [0, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
    weight = pl.load(b, [0, 0], [64, 16], target_memory=pl.MemorySpace.Mat)
    right = pl.move(weight, target_memory=pl.MemorySpace.Right)
    for block in pl.range(8):
        if block + 1 < 8:
            following = pl.load(a, [(block + 1) * 64, 0], [64, 64], target_memory=pl.MemorySpace.Mat)
        else:
            following = current
        left = pl.move(current, target_memory=pl.MemorySpace.Left)
        result = pl.tile.matmul(left, right)
        pl.store(result, [block * 64, 0], out)
        current = following
    return out


@pl.jit
def rotation_entry(
    a: pl.Tensor[[512, 64], pl.FP16],
    b: pl.Tensor[[64, 16], pl.FP16],
    out: pl.Out[pl.Tensor[[512, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = rotate(a, b, out)
    return out


@pl.jit.incore
def choose_live_inputs(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    a = pl.load(x, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    b = pl.load(x, [16, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    rhs = pl.load(y, [0, 0], [16, 16], target_memory=pl.MemorySpace.Mat)
    if pl.read(flag, [0]) > 0:
        selected = pl.yield_(a)
    else:
        selected = pl.yield_(b)
    product = pl.matmul(selected, rhs, out_dtype=pl.FP32)
    pl.store(product, [0, 0], out)
    keep_a = pl.matmul(a, rhs, out_dtype=pl.FP32)
    pl.store(keep_a, [16, 0], out)
    keep_b = pl.matmul(b, rhs, out_dtype=pl.FP32)
    pl.store(keep_b, [32, 0], out)
    return out


@pl.jit
def live_inputs_entry(
    x: pl.Tensor[[32, 16], pl.BF16],
    y: pl.Tensor[[16, 16], pl.BF16],
    flag: pl.Tensor[[1], pl.INT32],
    out: pl.Out[pl.Tensor[[48, 16], pl.FP32]],
):
    with pl.spmd(1):
        out = choose_live_inputs(x, y, flag, out)
    return out


def _assert_no_same_space_mat_move(text: str) -> None:
    for line in text.splitlines():
        if "pto.tmov" in line:
            for space in ("mat", "acc"):
                assert len(re.findall(rf"loc\s*=\s*{space}(?:,|>)", line)) < 2, line


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("entry", [rotation_entry, live_inputs_entry], ids=["3039", "3049"])
def test_original_public_reference_scenarios_compile(tmp_path: Path, planner, enabled, entry):
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    with passes.PassContext([], memory_planner=planner, enable_buffer_ir=enabled):
        compiled = entry.compile(
            config=RunConfig(
                backend_type=BackendType.Ascend910,
                execution_mode=ExecutionMode.ONBOARD,
                save_kernels=True,
                save_kernels_dir=str(tmp_path),
                dump_passes=True,
            )
        )
    sources = [path for path in compiled.output_dir.rglob("*.pto") if path.parent.name == "ptoas"]
    assert sources
    text = "\n".join(path.read_text() for path in sources)
    _assert_no_same_space_mat_move(text)
    allocations = [line for line in text.splitlines() if "pto.alloc_tile" in line]
    assert allocations
    # Check the actual planner route, including when JIT artifacts are cached:
    # PTOAS owns addressless placement; the addressed planners supply addresses.
    if planner == passes.MemoryPlanner.PTOAS:
        assert all(" addr = " not in line for line in allocations)
    else:
        assert all(" addr = " in line for line in allocations)
    if entry is live_inputs_entry:
        assert re.search(r"scf\.if .* -> \(!pto\.tile_buf<", text)
        assert "arith.select" not in text
    else:
        assert "arith.select" in text
        # The user loop carries its selected Tile handle. Internal allocation
        # instance searches may also introduce loops carrying scalar indices.
        assert any(
            "scf.for" in line and "iter_args" in line and "!pto.tile_buf" in line
            for line in text.splitlines()
        )
        assert "pto.tile_buf_addr" in text
    generated = "\n".join(path.read_text() for path in compiled.output_dir.rglob("*.cpp"))
    assert "TMOV" in generated
    assert "MTE2" in generated and "MTE1" in generated


def test_while_reference_selection_preserves_identity_views_and_scalar_condition():
    """A stalled update forwards the current view; the While condition stays scalar."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            pool: pl.Tensor[[32, 32], pl.FP16],
            trips: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
        ) -> pl.Tensor[[16, 32], pl.FP16]:
            seed = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
            for current, step in pl.while_(init_values=(seed, 0)):
                pl.cond(step < trips)
                if step % 2 == 0:
                    loaded = pl.tile.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
                    alias = pl.tile.reshape(loaded, [16, 32])
                    selected = pl.yield_(alias)
                else:
                    selected = pl.yield_(current)
                _old = pl.tile.extract(current, 0, 0, [16, 32], target_memory=pl.Mem.Left)
                result, _steps = pl.yield_(selected, step + 1)
            _last = pl.tile.extract(result, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            return out

    after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
    function = after.get_function("kernel")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    loop = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.WhileStmt))
    assert len(loop.iter_args) == 2
    assert isinstance(loop.iter_args[0].type, ir.TileType) and loop.iter_args[0].type.memref is None
    assert isinstance(loop.iter_args[1].type, ir.ScalarType)
    assert isinstance(loop.return_vars[0].type, ir.TileType) and loop.return_vars[0].type.memref is None
    assert isinstance(loop.condition, ir.Lt)
    assert loop.condition.left.same_as(loop.iter_args[1])
    descriptors: dict[str, ir.MemRef] = {}

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if op.var.name_hint in {"loaded", "alias"}:
                tile_type = op.var.type
                assert isinstance(tile_type, ir.TileType)
                memory = tile_type.memref
                assert memory is not None
                descriptors[op.var.name_hint] = memory
            super().visit_assign_stmt(op)

    Collect().visit_stmt(function.body)
    assert descriptors.keys() == {"loaded", "alias"}
    ir.assert_structural_equal(descriptors["loaded"], descriptors["alias"])
    version = descriptors["loaded"].slot_index_
    assert descriptors["loaded"].slot_count_ == 2
    assert isinstance(version, ir.Call) and version.op.name == ir.get_op("tile.storage_version").name
    protected = version.args[1]
    assert isinstance(protected, ir.MakeTuple) and len(protected.elements) == 1
    assert protected.elements[0].same_as(loop.iter_args[0])


def test_borrowed_loop_initializer_is_not_migrated_into_owned_storage():
    """A native carry keeps its borrowed initializer outside owned allocations."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            seed: pl.Tile[[16, 32], pl.FP16, pl.Mem.Mat],
            pool: pl.Tensor[[32, 32], pl.FP16],
            out: pl.Out[pl.Tensor[[16, 32], pl.FP16]],
        ) -> pl.Tensor[[16, 32], pl.FP16]:
            for i, (current,) in pl.range(2, init_values=(seed,)):
                if i % 2 == 0:
                    first = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
                    selected = pl.yield_(first)
                else:
                    second = pl.tile.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
                    selected = pl.yield_(second)
                _old = pl.tile.extract(current, 0, 0, [16, 32], target_memory=pl.Mem.Left)
                result = pl.yield_(selected)
            _last = pl.tile.extract(result, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            return out

    after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(Before))
    function = after.get_function("kernel")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    loop = next(stmt for stmt in function.body.stmts if isinstance(stmt, ir.ForStmt))
    assert isinstance(loop.iter_args[0].type, ir.TileType)
    assert loop.iter_args[0].initValue.same_as(function.params[0])


@pytest.mark.parametrize("count", [8, 64])
def test_many_independently_live_producers_keep_distinct_allocations(count):
    """Ifs select existing handles without pooling or searching for free slots."""
    statements = [
        f"v{i} = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)" for i in range(count)
    ]
    statements.append("selected0 = v0")
    for i in range(1, count):
        statements.extend(
            [
                f"if flag == {i}:",
                f"    selected{i} = pl.yield_(v{i})",
                "else:",
                f"    selected{i} = pl.yield_(selected{i - 1})",
            ]
        )
    statements.append(
        f"_chosen = pl.tile.extract(selected{count - 1}, 0, 0, [16, 32], target_memory=pl.Mem.Left)"
    )
    statements.extend(
        f"_keep{i} = pl.tile.extract(v{i}, 0, 0, [16, 32], target_memory=pl.Mem.Left)" for i in range(count)
    )
    body = "\n".join(f"        {statement}" for statement in statements)
    program = pl.parse_program(
        "@pl.program\nclass ManyLive:\n"
        "    @pl.function(type=pl.FunctionType.InCore)\n"
        "    def kernel(self, pool: pl.Tensor[[16, 32], pl.FP16], flag: pl.Scalar[pl.INDEX]):\n"
        f"{body}\n"
    )
    # Native references are classified after provisional storage contracts
    # have been resolved, rather than during InitMemRef.
    with passes.PassContext([], enable_buffer_ir=False):
        after = passes.materialize_semantic_aliases()(passes.init_mem_ref()(program))
    function = after.get_function("kernel")
    assert function is not None
    bases = []
    references = []
    producers = {f"v{i}" for i in range(count)}

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if op.var.name_hint in producers:
                assert isinstance(op.var.type, ir.TileType)
                memory = op.var.type.memref
                assert memory is not None and memory.slot_count_ == 1
                assert memory.slot_index_ is None
                bases.append(memory.base_.name_hint)
            super().visit_assign_stmt(op)

        def visit_if_stmt(self, op: ir.IfStmt) -> None:
            assert isinstance(op.return_vars[0].type, ir.TileType)
            assert op.return_vars[0].type.memref is None
            references.append(op.return_vars[0])
            super().visit_if_stmt(op)

        def visit_for_stmt(self, op: ir.ForStmt) -> None:
            pytest.fail("Selecting an existing handle must not introduce a search loop")

    Collect().visit_stmt(function.body)
    assert len(bases) == len(set(bases)) == count
    assert len(references) == count - 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
