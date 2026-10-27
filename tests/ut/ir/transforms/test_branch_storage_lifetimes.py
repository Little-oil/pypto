# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Real platform capacity and path-sensitive alias liveness from issue #3035."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pypto.language as pl
import pytest
from pypto import backend, ir, passes
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.ir.compile import compile as compile_program
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import ExecutionMode


@pl.program
class BranchGather:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[64, 512], pl.BF16],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[64, 512], pl.BF16]],
        state: pl.Out[pl.Tensor[[64, 256], pl.FP32]],
    ) -> tuple[pl.Tensor[[64, 512], pl.BF16], pl.Tensor[[64, 256], pl.FP32]]:
        first = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        second = pl.tile.full([32, 256], dtype=pl.FP32, value=0.0)
        flag = pl.tensor.read(flags, [0])
        for _i, (s0, s1) in pl.range(2, init_values=(first, second)):
            seed = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
            if flag != 0:
                filled = pl.tile.gather_row(seed, pool, [0, 0], [0, 0], [64, 512])
                selected = filled
            else:
                fresh = pl.tile.full([64, 512], dtype=pl.BF16, value=0.0)
                for j, (carry,) in pl.range(64, init_values=(fresh,)):
                    if j < flag:
                        next_value = pl.tile.gather_row(carry, pool, [j, 0], [j, 0], [1, 512])
                    else:
                        next_value = carry
                    gathered = pl.yield_(next_value)
                selected = gathered
            pl.tile.store(selected, [0, 0], out)
            next0 = pl.tile.add(s0, 1.0)
            next1 = pl.tile.add(s1, 2.0)
            result0, result1 = pl.yield_(next0, next1)
        state0 = pl.tile.store(result0, [0, 0], state)
        state1 = pl.tile.store(result1, [32, 0], state0)
        return out, state1


@pl.program
class ConditionalReset:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(
        self,
        pool: pl.Tensor[[128, 256], pl.FP32],
        flags: pl.Tensor[[1], pl.INT32],
        out: pl.Out[pl.Tensor[[128, 256], pl.FP32]],
    ) -> pl.Tensor[[128, 256], pl.FP32]:
        held = pl.tile.load(pool, [64, 0], [64, 256], target_memory=pl.Mem.Vec)
        current = pl.tile.load(pool, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
        flag = pl.tensor.read(flags, [0])
        if flag != 0:
            reset = pl.tile.full([64, 256], dtype=pl.FP32, value=0.0)
            selected = reset
        else:
            selected = current
        saved = pl.tile.store(selected, [0, 0], out)
        result = pl.tile.store(held, [64, 0], saved)
        return result


def _plan(program: ir.Program, instruments: list) -> ir.Program:
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext(instruments, enable_buffer_ir=False, memory_planner=passes.MemoryPlanner.PYPTO):
        ssa = passes.convert_to_ssa()(program)
        initialized = passes.init_mem_ref()(ssa)
        aliases = passes.materialize_semantic_aliases()(initialized)
        return passes.memory_reuse()(aliases)


def _tile_assignments(program: ir.Program) -> dict[str, ir.AssignStmt]:
    found: dict[str, ir.AssignStmt] = {}

    class Collector(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            if isinstance(op.var.type, ir.TileType):
                found[op.var.name_hint] = op
            super().visit_assign_stmt(op)

    for function in program.functions.values():
        Collector().visit_stmt(function.body)
    return found


def _vec_memrefs(program: ir.Program) -> list[tuple[str, ir.MemRef]]:
    # Native phis and their views select storage without owning an allocation.
    # Capacity and physical alias checks apply to the actual producers only.
    return [
        (name, memory)
        for name, assignment in _tile_assignments(program).items()
        if isinstance(assignment.var.type, ir.TileType)
        and assignment.var.type.memory_space == ir.MemorySpace.Vec
        and (memory := assignment.var.type.memref) is not None
    ]


@pytest.mark.parametrize("program", [BranchGather, ConditionalReset], ids=["gather", "reset"])
def test_original_branch_capacity_examples_fit_without_changing_tiles(program, pass_verification_instruments):
    planned = _plan(program, pass_verification_instruments)
    addressed = passes.allocate_memory_addr()(planned)
    ends = []
    for _, memref in _vec_memrefs(addressed):
        assert isinstance(memref.byte_offset_, ir.ConstInt)
        ends.append(memref.byte_offset_.value + memref.size_)
    assert ends and max(ends) <= 188416


def test_nested_phi_aliases_preserve_result_families_and_external_seed(pass_verification_instruments):
    # The inner phi records every alias of fresh. Flattening it into the outer
    # phi must expand that allocation once, without merging the two result slots
    # or treating an independently observed outside seed as branch-local.
    alias_count = 24
    aliases = "\n".join(
        f"                alias_{index} = pl.tile.reshape("
        f"{'fresh' if index == 0 else f'alias_{index - 1}'}, [32, 256])"
        for index in range(alias_count)
    )
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, pool: pl.Tensor[[32, 256], pl.FP32], flag: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[96, 256], pl.FP32]]) -> pl.Tensor[[96, 256], pl.FP32]:
        seed = pl.tile.load(pool, [0, 0], [32, 256], target_memory=pl.Mem.Vec)
        if flag > 0:
            if flag > 1:
                fresh = pl.tile.full([32, 256], dtype=pl.FP32, value=1.0)
{aliases}
                other = pl.tile.full([32, 256], dtype=pl.FP32, value=2.0)
                inner_a, inner_b = pl.yield_(alias_{alias_count - 1}, other)
            else:
                alternate_a = pl.tile.full([32, 256], dtype=pl.FP32, value=3.0)
                alternate_b = pl.tile.full([32, 256], dtype=pl.FP32, value=4.0)
                inner_a, inner_b = pl.yield_(alternate_a, alternate_b)
            outer_a, outer_b = pl.yield_(inner_a, inner_b)
        else:
            outer_a, outer_b = pl.yield_(seed, seed)
        selected_a = pl.tile.reshape(outer_a, [32, 256])
        selected_b = pl.tile.reshape(outer_b, [32, 256])
        saved_a = pl.tile.store(selected_a, [0, 0], out)
        saved_b = pl.tile.store(selected_b, [32, 0], saved_a)
        result = pl.tile.store(seed, [64, 0], saved_b)
        return result
""")
    planned = _plan(before, pass_verification_instruments)
    values = _vec_memrefs(planned)
    fresh = next(memref for name, memref in values if name.startswith("fresh"))
    alias_memrefs = [memref for name, memref in values if name.startswith("alias_")]
    assert len(alias_memrefs) == alias_count
    assert all(memref.base_.same_as(fresh.base_) for memref in alias_memrefs)
    # Each executed branch retains two different selected payloads alongside
    # the independently read seed. The selected views themselves own no bytes.
    for prefixes in (("seed", "fresh", "other"), ("seed", "alternate_a", "alternate_b")):
        observed = [next(memory for name, memory in values if name.startswith(prefix)) for prefix in prefixes]
        assert all(
            not left.base_.same_as(right.base_)
            for index, left in enumerate(observed)
            for right in observed[index + 1 :]
        )
    assignments = _tile_assignments(planned)
    for prefix in ("selected_a", "selected_b"):
        selected = next(assignment for name, assignment in assignments.items() if name.startswith(prefix))
        assert isinstance(selected.var.type, ir.TileType)
        assert (
            isinstance(selected.value, ir.Call) and selected.value.op.name == ir.get_op("tile.reshape").name
        )
        source = selected.value.args[0]
        assert isinstance(source, ir.Var) and source.name_hint.startswith(prefix.replace("selected", "outer"))
        assert isinstance(source.type, ir.TileType)
        assert selected.var.type.memref is not None and source.type.memref is not None
        ir.assert_structural_equal(selected.var.type.memref, source.type.memref)
    assert passes.allocate_memory_addr()(planned) is not None


def _assert_deep_then_phi_chain(depth: int) -> None:
    rows = 32

    def nested(level: int, indent: int) -> tuple[list[str], str]:
        pad = " " * indent
        if level == depth:
            return [f"{pad}fresh = pl.tile.full([{rows}, 256], dtype=pl.FP32, value=1.0)"], "fresh"
        body, source = nested(level + 1, indent + 4)
        result = f"phi_{level}"
        return [
            f"{pad}if flag > {level}:",
            *body,
            f"{pad}    {result} = pl.yield_({source})",
            f"{pad}else:",
            f"{pad}    alternate_{level} = pl.tile.full([{rows}, 256], dtype=pl.FP32, value=2.0)",
            f"{pad}    {result} = pl.yield_(alternate_{level})",
        ], result

    body, source = nested(0, 8)
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, pool: pl.Tensor[[{rows}, 256], pl.FP32], flag: pl.Scalar[pl.INDEX],
               out: pl.Out[pl.Tensor[[{2 * rows}, 256], pl.FP32]]) -> pl.Tensor[[{2 * rows}, 256], pl.FP32]:
        seed = pl.tile.load(pool, [0, 0], [{rows}, 256], target_memory=pl.Mem.Vec)
{chr(10).join(body)}
        selected = pl.tile.reshape({source}, [{rows}, 256])
        saved = pl.tile.store(selected, [0, 0], out)
        result = pl.tile.store(seed, [{rows}, 0], saved)
        return result
""")
    planned = _plan(before, [passes.VerificationInstrument(passes.VerificationMode.BEFORE_AND_AFTER)])
    values = _vec_memrefs(planned)
    seed = next(memory for name, memory in values if name.startswith("seed"))
    # Canonical yield copies can add names such as alternate_0_mv. Count the
    # original payload producers, while capacity below still covers all copies.
    sources = []
    for name, assignment in _tile_assignments(planned).items():
        if (
            name.startswith(("fresh", "alternate_"))
            and isinstance(assignment.value, ir.Call)
            and assignment.value.op.name == ir.get_op("tile.full").name
        ):
            assert isinstance(assignment.var.type, ir.TileType) and assignment.var.type.memref is not None
            sources.append(assignment.var.type.memref)
    assert len(sources) == depth + 1
    assert all(not seed.base_.same_as(source.base_) for source in sources)
    selected = next(
        assignment for name, assignment in _tile_assignments(planned).items() if name.startswith("selected")
    )
    assert isinstance(selected.var.type, ir.TileType)
    assert isinstance(selected.value, ir.Call) and selected.value.op.name == ir.get_op("tile.reshape").name
    source = selected.value.args[0]
    assert isinstance(source, ir.Var) and source.name_hint.startswith("phi_0")
    assert isinstance(source.type, ir.TileType)
    assert selected.var.type.memref is not None and source.type.memref is not None
    ir.assert_structural_equal(selected.var.type.memref, source.type.memref)
    addressed = passes.allocate_memory_addr()(planned)
    ends = []
    for _, memory in _vec_memrefs(addressed):
        assert isinstance(memory.byte_offset_, ir.ConstInt)
        ends.append(memory.byte_offset_.value + memory.size_)
    # All branch payloads are mutually exclusive. Only the selected value
    # and the independently observed seed need storage after the join.
    assert ends and max(ends) <= 2 * rows * 256 * 4


def test_deep_then_phi_chain_preserves_liveness_and_capacity():
    # Every outer then arm yields the inner phi. A nested phi appearing again
    # among its allocation's aliases used to make raw-use traversal exponential.
    # Bound the child process to contain that regression, without a timing
    # assertion on a successful plan or leaving a runaway worker behind.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1])['_assert_deep_then_phi_chain'](30)",
            str(Path(__file__).resolve()),
        ],
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            # Preserve the parent's import path for both editable and wheel
            # installs; forcing python/ would shadow the installed extension.
            "PYTHONPATH": os.pathsep.join(sys.path),
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("forward_view", [False, True], ids=["direct", "no_access_view"])
def test_native_if_read_keeps_both_incoming_allocations_live(forward_view, pass_verification_instruments):
    # Neither input is read by name after the join. A later load must not
    # overwrite either possible source before the selected handle is consumed.
    view = "viewed = pl.tile.reshape(selected, [16, 32])" if forward_view else "viewed = selected"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[48, 32], pl.FP16], flag: pl.Scalar[pl.INDEX]
    ) -> tuple[pl.Tile[[16, 32], pl.FP16, pl.Mem.Left], pl.Tile[[16, 32], pl.FP16, pl.Mem.Left]]:
        a = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        b = pl.tile.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
        if flag > 0:
            selected = pl.yield_(a)
        else:
            selected = pl.yield_(b)
        {view}
        later = pl.tile.load(pool, [32, 0], [16, 32], target_memory=pl.Mem.Mat)
        chosen = pl.tile.extract(viewed, 0, 0, [16, 32], target_memory=pl.Mem.Left)
        other = pl.tile.extract(later, 0, 0, [16, 32], target_memory=pl.Mem.Left)
        return chosen, other
""")
    planned = _plan(before, pass_verification_instruments)
    memories: dict[str, ir.MemRef] = {}

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            name = op.var.name_hint.split("__", 1)[0]
            if name in {"a", "b", "later"}:
                assert isinstance(op.var.type, ir.TileType) and op.var.type.memref is not None
                memories[name] = op.var.type.memref
            if name == "viewed":
                assert isinstance(op.var.type, ir.TileType) and op.var.type.memref is None
            super().visit_assign_stmt(op)

    function = planned.get_function("kernel")
    assert function is not None
    Collect().visit_stmt(function.body)
    assert memories.keys() == {"a", "b", "later"}
    assert len({memory.base_.name_hint for memory in memories.values()}) == 3


def test_native_if_consumers_keep_mat_capacity_retry_from_overwriting_sources(pass_verification_instruments):
    # Three held tiles, the selected payload and a later load are all live.
    # Each is 128 KiB: the 640-KiB live set cannot fit in the 512-KiB Mat space.
    # The retry must retain the post-join read of the selected incoming source.
    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            pool: pl.Tensor[[768, 512], pl.FP16],
            flag: pl.Scalar[pl.INDEX],
        ) -> tuple[
            pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
            pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
            pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
            pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
            pl.Tile[[16, 32], pl.FP16, pl.Mem.Left],
        ]:
            held0 = pl.tile.load(pool, [0, 0], [128, 512], target_memory=pl.Mem.Mat)
            held1 = pl.tile.load(pool, [128, 0], [128, 512], target_memory=pl.Mem.Mat)
            held2 = pl.tile.load(pool, [256, 0], [128, 512], target_memory=pl.Mem.Mat)
            if flag > 0:
                first = pl.tile.load(pool, [384, 0], [128, 512], target_memory=pl.Mem.Mat)
                selected = pl.yield_(first)
            else:
                second = pl.tile.load(pool, [512, 0], [128, 512], target_memory=pl.Mem.Mat)
                selected = pl.yield_(second)
            later = pl.tile.load(pool, [640, 0], [128, 512], target_memory=pl.Mem.Mat)
            read0 = pl.tile.extract(held0, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            read1 = pl.tile.extract(held1, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            read2 = pl.tile.extract(held2, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            chosen = pl.tile.extract(selected, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            last = pl.tile.extract(later, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            return read0, read1, read2, chosen, last

    planned = _plan(Before, pass_verification_instruments)
    memories = {}
    for name, assignment in _tile_assignments(planned).items():
        stem = name.split("__", 1)[0]
        if stem in {"first", "second", "later"}:
            assert isinstance(assignment.var.type, ir.TileType)
            memory = assignment.var.type.memref
            assert memory is not None and memory.size_ == 128 * 1024
            memories[stem] = memory
    assert memories.keys() == {"first", "second", "later"}
    assert all(not memories["later"].base_.same_as(memories[source].base_) for source in ("first", "second"))
    with pytest.raises(ValueError, match=r"Mat buffer usage .* exceeds platform limit"):
        passes.allocate_memory_addr()(planned)


@pytest.mark.parametrize("bounds", ["0", "1", "2", "trips"], ids=["zero", "one", "many", "runtime"])
@pytest.mark.parametrize("forward_view", [False, True], ids=["direct", "no_access_view"])
def test_unused_loop_result_keeps_payload_only_for_possible_backedges(
    bounds, forward_view, pass_verification_instruments
):
    # The result is unused, but a possible next iteration reads the selected
    # payload through carry. Candidate must then survive overwrite, including
    # when a NoAccess view forwards it. Zero/one-trip yields only bind storage.
    view = "viewed = pl.tile.reshape(candidate, [16, 32])" if forward_view else ""
    selected_source = "viewed" if forward_view else "candidate"
    before = pl.parse_program(f"""
@pl.program
class Before:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, pool: pl.Tensor[[32, 32], pl.FP16], trips: pl.Scalar[pl.INDEX],
               flag: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[16, 32], pl.FP16]]
    ) -> pl.Tensor[[16, 32], pl.FP16]:
        seed = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
        for _i, (carry,) in pl.range({bounds}, init_values=(seed,)):
            _read = pl.tile.extract(carry, 0, 0, [16, 32], target_memory=pl.Mem.Left)
            candidate = pl.tile.load(pool, [0, 0], [16, 32], target_memory=pl.Mem.Mat)
            {view}
            overwrite = pl.tile.load(pool, [16, 0], [16, 32], target_memory=pl.Mem.Mat)
            if flag != 0:
                selected = pl.yield_({selected_source})
            else:
                selected = pl.yield_(overwrite)
            _unused = pl.yield_(selected)
        return out
""")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext(pass_verification_instruments, enable_buffer_ir=False):
        initialized = passes.init_mem_ref()(before)
        prepared = passes.materialize_semantic_aliases()(initialized)
    initial_function = initialized.get_function("kernel")
    assert initial_function is not None and isinstance(initial_function.body, ir.SeqStmts)
    initial_loop = next(
        statement for statement in initial_function.body.stmts if isinstance(statement, ir.ForStmt)
    )
    assert isinstance(initial_loop.iter_args[0].type, ir.TileType)
    function = prepared.get_function("kernel")
    assert function is not None and isinstance(function.body, ir.SeqStmts)
    loop = next(statement for statement in function.body.stmts if isinstance(statement, ir.ForStmt))
    repeats = bounds in {"2", "trips"}
    assert isinstance(loop.iter_args[0].type, ir.TileType)
    memories: dict[str, ir.MemRef] = {}

    class Collect(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            if op.var.name_hint in {"seed", "candidate", "overwrite", "viewed"}:
                assert isinstance(op.var.type, ir.TileType) and op.var.type.memref is not None
                memories[op.var.name_hint] = op.var.type.memref
            super().visit_assign_stmt(op)

    Collect().visit_stmt(function.body)
    expected_names = {"seed", "candidate", "overwrite"} | ({"viewed"} if forward_view else set())
    assert memories.keys() == expected_names
    candidate, overwrite = memories["candidate"], memories["overwrite"]
    assert candidate.slot_count_ == overwrite.slot_count_ == (2 if repeats else 1)
    if repeats:
        # Each producer selects an execution instance of its own allocation.
        # Both choices must exclude the incoming handle, even through a view.
        assert not candidate.base_.same_as(overwrite.base_)
        for memory in (candidate, overwrite):
            version = memory.slot_index_
            assert isinstance(version, ir.Call) and version.op.name == ir.get_op("tile.storage_version").name
            protected = version.args[1]
            assert isinstance(protected, ir.MakeTuple) and len(protected.elements) == 1
            assert protected.elements[0].same_as(loop.iter_args[0])
        # The initializer may remain independent or seed a safely shared bank
        # at slot zero; it is never forced into a family-wide slot pool.
        seed = memories["seed"]
        if seed.base_.same_as(candidate.base_) or seed.base_.same_as(overwrite.base_):
            assert seed.slot_count_ == 2
            assert isinstance(seed.slot_index_, ir.ConstInt) and seed.slot_index_.value == 0
        else:
            assert seed.slot_count_ == 1 and seed.slot_index_ is None
    else:
        assert all(memory.slot_count_ == 1 and memory.slot_index_ is None for memory in memories.values())
    if forward_view:
        ir.assert_structural_equal(memories["candidate"], memories["viewed"])


@pytest.mark.parametrize("trips", [1, 2], ids=["single_trip", "backedge"])
def test_acc_capture_is_retained_only_across_possible_backedges(trips: int, pass_verification_instruments):
    """A last read before replacement permits reuse unless it can recur on a backedge."""

    @pl.program
    class Before:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            a: pl.Tensor[[128, 64], pl.BF16],
            b: pl.Tensor[[64, 512], pl.BF16],
            flag: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[128, 512], pl.FP32]],
        ) -> pl.Tensor[[128, 512], pl.FP32]:
            am = pl.tile.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
            bm = pl.tile.load(b, [0, 0], [64, 512], target_memory=pl.Mem.Mat)
            other = pl.tile.load(a, [64, 0], [64, 64], target_memory=pl.Mem.Mat)
            left = pl.tile.extract(am, 0, 0, [64, 64], target_memory=pl.Mem.Left)
            right = pl.tile.extract(bm, 0, 0, [64, 512], target_memory=pl.Mem.Right)
            other_left = pl.tile.extract(other, 0, 0, [64, 64], target_memory=pl.Mem.Left)
            seed = pl.tile.matmul(left, right)
            for _i, (carry,) in pl.range(trips, init_values=(seed,)):
                _old_used = pl.tile.store(seed, [0, 0], out)
                if flag > 0:
                    fresh = pl.tile.matmul(other_left, right)
                    selected = pl.yield_(fresh)
                else:
                    selected = pl.yield_(carry)
                result = pl.yield_(selected)
            final = pl.tile.store(result, [64, 0], out)
            return final

    if trips == 1:
        planned = _plan(Before, pass_verification_instruments)
        # The old read cannot recur. One full-capacity Acc allocation must fit.
        assert passes.allocate_memory_addr()(planned) is not None
    else:
        # The next iteration reads the original seed again, requiring two
        # independent full-capacity Acc versions rather than overwriting it.
        with pytest.raises(ValueError, match=r"Acc buffer usage .* exceeds platform limit"):
            planned = _plan(Before, pass_verification_instruments)
            passes.allocate_memory_addr()(planned)


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("program", [BranchGather, ConditionalReset], ids=["gather", "reset"])
@pytest.mark.parametrize("enable_buffer_ir", [False, True], ids=["tile", "buffer"])
def test_original_capacity_examples_compile_through_default_pipeline(
    program, enable_buffer_ir, tmp_path: Path, pass_verification_instruments
):
    """Exercise the public compiler, including default Buffer selection/planning.

    Storage ownership and branch reachability are checked on the existing Tile
    planning path, using the real 188416-byte Vec limit.
    """
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    with passes.PassContext(
        pass_verification_instruments,
        memory_planner=passes.MemoryPlanner.PYPTO,
        enable_buffer_ir=enable_buffer_ir,
    ):
        compiled = compile_program(
            program,
            strategy=OptimizationStrategy.Default,
            output_dir=str(tmp_path),
            backend_type=BackendType.Ascend910,
            execution_mode=ExecutionMode.ONBOARD,
            dump_passes=True,
            skip_ptoas=False,
        )

    source_files = list((Path(compiled.output_dir) / "ptoas").glob("*.pto"))
    assert source_files
    ends = []
    for source in source_files:
        text = source.read_text()
        constants = {
            match[1]: int(match[2])
            for match in re.finditer(r"(%\w+) = arith.constant (\d+) : (?:i64|index)\b", text)
        }
        for line in text.splitlines():
            if "pto.alloc_tile" not in line or "loc=vec" not in line:
                continue
            address = re.search(r"addr = (%\w+)", line)
            shape = re.search(r"dtype=(\w+), rows=(\d+), cols=(\d+)", line)
            assert address and shape, line
            bits = {"bf16": 16, "f32": 32, "ui8": 8}[shape[1]]
            ends.append(constants[address[1]] + int(shape[2]) * int(shape[3]) * bits // 8)
        cpp = source.with_suffix(".cpp")
        assert cpp.is_file() and "TSTORE(" in cpp.read_text()
    assert ends and max(ends) <= 188416


def _observed_reset(observation: str) -> ir.Program:
    alias = "viewed = pl.tile.reshape(current, [64, 256])" if observation == "alias_same_arm" else ""
    observed = "viewed" if observation == "alias_same_arm" else "current"
    same_arm = (
        f"pl.tile.store({observed}, [64, 0], out)" if observation in {"same_arm", "alias_same_arm"} else ""
    )
    after_join = "saved = pl.tile.store(current, [64, 0], saved)" if observation == "after_join" else ""
    if observation == "backedge":
        body = """
        for i in pl.range(2):
            if i == 0:
                reset = pl.tile.full([64, 256], dtype=pl.FP32, value=0.0)
                selected = reset
            else:
                selected = current
            out = pl.tile.store(selected, [i * 64, 0], out)
        result = pl.tile.store(held, [128, 0], out)
"""
    else:
        body = f"""
        {alias}
        flag = pl.tensor.read(flags, [0])
        if flag != 0:
            reset = pl.tile.full([64, 256], dtype=pl.FP32, value=0.0)
            {same_arm}
            selected = reset
        else:
            selected = current
        saved = pl.tile.store(selected, [0, 0], out)
        {after_join}
        result = pl.tile.store(held, [128, 0], saved)
"""
    return pl.parse_program(
        f"""
@pl.program
class ObservedReset:
    @pl.function(type=pl.FunctionType.AIV)
    def kernel(self, pool: pl.Tensor[[128, 256], pl.FP32], flags: pl.Tensor[[1], pl.INT32],
               out: pl.Out[pl.Tensor[[192, 256], pl.FP32]]) -> pl.Tensor[[192, 256], pl.FP32]:
        held = pl.tile.load(pool, [64, 0], [64, 256], target_memory=pl.Mem.Vec)
        current = pl.tile.load(pool, [0, 0], [64, 256], target_memory=pl.Mem.Vec)
{body}
        return result
"""
    )


@pytest.mark.parametrize("observation", ["same_arm", "alias_same_arm", "after_join", "backedge"])
def test_observable_old_values_remain_separate_even_when_capacity_overflows(
    observation, pass_verification_instruments
):
    # Three 64-KiB values really coexist on at least one execution path. A path
    # refinement must retain this real overflow, rather than treating every
    # old seed or sibling-associated alias as mutually exclusive.
    planned = _plan(_observed_reset(observation), pass_verification_instruments)
    values = _vec_memrefs(planned)
    current = next(memref for name, memref in values if name.startswith("current"))
    reset = next(memref for name, memref in values if name.startswith("reset"))
    assert not current.base_.same_as(reset.base_)
    with pytest.raises(ValueError, match="Vec buffer usage.*exceeds platform limit"):
        passes.allocate_memory_addr()(planned)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
