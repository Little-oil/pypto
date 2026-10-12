# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parser round-trip tests for the ``pl.set_l2_hint(...)`` declaration.

``pl.set_l2_hint(t, load=...)`` emits no IR statement of its own: the parser
hoists it onto the enclosing scope as ``attrs['load_l2_hint_vars']``, a list of
``(Var, LoadL2Hint-as-int)`` pairs. These tests assert what a *reparse* of the
printed program recovers — which scope carries the declaration, which tensor it
names, and which policy — for the three shapes with distinct printer paths:

1. a ``with pl.at(...)`` scope;
2. a ``for i in pl.spmd(...)`` kernel, where the declaration attaches to the
   nested InCore carrier whose header the printer inlines away;
3. a ``for i in pl.spmd(...)`` kernel whose body is otherwise empty, where the
   printer would otherwise fall back to a ``pass`` filler.

Each case also asserts the print -> parse -> print fixpoint. The printer-side
counterpart (what text is emitted, and where) lives in
``tests/ut/ir/printing/test_l2_hint_printer.py``.

Note these assert textual stability rather than
``ir.assert_structural_equal``: the ``load_l2_hint_vars`` attr value type is not
yet handled by the structural comparator.
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.parser.diagnostics import ParserSyntaxError

BYPASS = int(ir.LoadL2Hint.NotAllocClean)
DEFAULT = int(ir.LoadL2Hint.NormalFirstVictim)

_LOAD_L2_HINT_VARS_ATTR = "load_l2_hint_vars"


def _collect_decls(node) -> list[tuple[str, list[tuple[str, int]]]]:
    """Walk an IR body and collect every scope carrying a L2-hint declaration.

    Returns ``(scope class name, [(tensor name, policy int), ...])`` per
    carrying scope, in traversal order, so a test can assert both *what* was
    declared and *which* scope kind ended up holding it.
    """
    found: list[tuple[str, list[tuple[str, int]]]] = []

    def walk(n) -> None:
        if isinstance(n, ir.ScopeStmt):
            entries = dict(n.attrs).get(_LOAD_L2_HINT_VARS_ATTR)
            if entries is not None:
                found.append((type(n).__name__, [(var.name_hint, int(policy)) for var, policy in entries]))
        if isinstance(n, ir.SeqStmts):
            for s in n.stmts:
                walk(s)
        elif isinstance(n, ir.IfStmt):
            walk(n.then_body)
            if n.else_body is not None:
                walk(n.else_body)
        elif getattr(n, "body", None) is not None:
            walk(n.body)

    walk(node)
    return found


def _program_decls(prog) -> list[tuple[str, list[tuple[str, int]]]]:
    decls: list[tuple[str, list[tuple[str, int]]]] = []
    for func in prog.functions.values():
        decls.extend(_collect_decls(func.body))
    return decls


def _reparse(prog):
    """print -> parse, asserting the reprint is a fixpoint. Returns the reparsed program."""
    text = ir.python_print(prog)
    reparsed = pl.parse_program(text)
    reprinted = ir.python_print(reparsed)
    assert reprinted == text, (
        f"round-trip is not a fixpoint\n--- first ---\n{text}\n--- second ---\n{reprinted}"
    )
    return reparsed


# ─── (i) a ``with pl.at(...)`` scope ──────────────────────────────────────


def _at_scope_program():
    @pl.program
    class AtScope:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
            return out

    return AtScope


def test_parse_at_scope_records_declaration_on_the_scope():
    """The marker leaves no statement behind; it lands on the enclosing scope's attrs."""
    prog = _at_scope_program()
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_roundtrip_at_scope_recovers_declaration():
    """A reparse of the printed ``with pl.at(...)`` recovers the same declaration."""
    prog = _at_scope_program()
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── (ii) a ``for i in pl.spmd(...)`` kernel ──────────────────────────────


def _spmd_for_program():
    @pl.program
    class SpmdFor:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            b: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for i in pl.spmd(4):
                pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
                offset = i * 128
                tile_b: pl.Tile[[128, 128], pl.FP32] = pl.load(b, [offset, 0], [128, 128])
                out = pl.store(tile_b, [offset, 0], out)
            return out

    return SpmdFor


def test_parse_spmd_for_attaches_declaration_to_the_incore_carrier():
    """The declaration rides the InCore carrier, not the Spmd scope around it.

    The carrier is what the outliner turns into the per-block kernel whose
    params the policy resolves against, so attaching it anywhere else would
    leave nothing downstream to consume it.
    """
    assert _program_decls(_spmd_for_program()) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_roundtrip_spmd_for_recovers_declaration():
    """The printer inlines the carrier's header away; the reparse still recovers it."""
    prog = _spmd_for_program()
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── (iii) a scope whose body is otherwise empty ──────────────────────────


def _empty_spmd_body_program():
    @pl.program
    class EmptyBody:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            b: pl.Tensor[[512, 128], pl.FP32],
            out: pl.Out[pl.Tensor[[512, 128], pl.FP32]],
        ) -> pl.Tensor[[512, 128], pl.FP32]:
            for i in pl.spmd(4):
                pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
            return out

    return EmptyBody


def test_roundtrip_empty_spmd_body_recovers_declaration():
    """A body holding only the declaration survives; the ``pass`` filler would erase it."""
    prog = _empty_spmd_body_program()
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


# ─── Policies, multiplicity and spelling ──────────────────────────────────


def test_roundtrip_preserves_both_policies_and_their_order():
    """DEFAULT and BYPASS both survive, distinctly, in the order they were written."""

    @pl.program
    class Multi:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
                pl.set_l2_hint(a, load=pl.LoadL2Hint.NormalFirstVictim)
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                out = pl.assemble(out, c, [0, 0])
            return out

    expected = [("InCoreScopeStmt", [("b", BYPASS), ("a", DEFAULT)])]
    assert _program_decls(Multi) == expected
    assert _program_decls(_reparse(Multi)) == expected


def test_roundtrip_normalises_position_of_a_late_declaration():
    """A declaration written mid-body reparses identically — position carries no meaning."""

    @pl.program
    class Late:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[256, 128], pl.FP32],
            b: pl.Tensor[[128, 256], pl.FP32],
            out: pl.Out[pl.Tensor[[256, 256], pl.FP32]],
        ) -> pl.Tensor[[256, 256], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
                c: pl.Tensor[[256, 256], pl.FP32] = pl.matmul(a, b, out_dtype=pl.FP32)
                pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
                out = pl.assemble(out, c, [0, 0])
            return out

    assert _program_decls(_reparse(Late)) == [("InCoreScopeStmt", [("b", BYPASS)])]


def test_parse_accepts_the_submodule_spelling_and_normalises_it():
    """``pl.tensor.set_l2_hint`` parses too, and prints as ``pl.set_l2_hint``.

    The marker function lives in ``tensor_ops``, so both spellings reach the
    parser; normalising on print keeps the round-trip a fixpoint from either.
    """
    source = """
import pypto.language as pl


@pl.program
class Alt:
    @pl.function
    def main(self, a: pl.Tensor[[256, 128], pl.FP32], b: pl.Tensor[[128, 256], pl.FP32],
             out: pl.Out[pl.Tensor[[256, 256], pl.FP32]]) -> pl.Tensor[[256, 256], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            pl.tensor.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)
            c: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.matmul(a, b, out_dtype=pl.FP32)
            out: pl.Tensor[[256, 256], pl.FP32] = pl.tensor.assemble(out, c, [0, 0])
        return out
"""
    prog = pl.parse_program(source)
    assert _program_decls(prog) == [("InCoreScopeStmt", [("b", BYPASS)])]

    text = ir.python_print(prog)
    assert "pl.set_l2_hint(b, load=pl.LoadL2Hint.NotAllocClean)" in text, text
    assert "pl.tensor.set_l2_hint(" not in text, text
    assert _program_decls(_reparse(prog)) == _program_decls(prog)


def test_repeating_the_same_policy_for_one_tensor_is_accepted():
    """A restated declaration is redundant, not an error: one entry survives."""

    @pl.program
    class Prog:
        @pl.function
        def main(
            self,
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ) -> pl.Tensor[[64, 64], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                pl.set_l2_hint(a, load=pl.LoadL2Hint.NotAllocClean)
                pl.set_l2_hint(a, load=pl.LoadL2Hint.NotAllocClean)
                t: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                out = pl.store(t, [0, 0], out)
            return out

    decls = _program_decls(Prog)
    assert sum(len(entries) for _scope, entries in decls) == 1, decls


def test_conflicting_policies_for_one_tensor_are_rejected():
    """Two different policies for one tensor in one scope contradict each other.

    Keeping the first silently resolved this toward BYPASS — the direction that
    also asserts the coherency contract the second statement retracts — so the
    parser rejects it instead of picking a winner.
    """
    with pytest.raises(ParserSyntaxError, match="conflicting load hints"):

        @pl.program
        class Prog:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[64, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            ) -> pl.Tensor[[64, 64], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="k"):
                    pl.set_l2_hint(a, load=pl.LoadL2Hint.NotAllocClean)
                    pl.set_l2_hint(a, load=pl.LoadL2Hint.NormalFirstVictim)
                    t: pl.Tile[[64, 64], pl.FP32] = pl.load(a, [0, 0], [64, 64])
                    out = pl.store(t, [0, 0], out)
                return out


@pytest.mark.parametrize("load_hint", list(pl.LoadL2Hint))
@pytest.mark.parametrize("store_hint", list(pl.StoreL2Hint))
def test_l2_hint_roundtrip(load_hint, store_hint):
    """Preserve the exact load/store enum, including the explicit zero hint."""

    @pl.program
    class Hinted:
        @pl.function(type=pl.FunctionType.InCore)
        def main(
            self, x: pl.Tensor[[32, 32], pl.FP32], out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]
        ) -> pl.Tensor[[32, 32], pl.FP32]:
            t = pl.load(x, [0, 0], [32, 32], l2_hint=load_hint)
            out = pl.store(t, [0, 0], out, l2_hint=store_hint)
            return out

    text = ir.python_print(Hinted)
    assert f"l2_hint=pl.LoadL2Hint.{load_hint.name}" in text
    assert f"l2_hint=pl.StoreL2Hint.{store_hint.name}" in text
    ir.assert_structural_equal(Hinted, _reparse(Hinted))
    ir.assert_structural_equal(Hinted, ir.deserialize(ir.serialize(Hinted)))


@pytest.mark.parametrize("load_hint", list(pl.LoadL2Hint))
@pytest.mark.parametrize("store_hint", list(pl.StoreL2Hint))
def test_scope_both_directions_roundtrip_and_serialize(load_hint, store_hint):
    source = f"""
@pl.program
class HintedScope:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32],
             out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_l2_hint(x, load=pl.LoadL2Hint.{load_hint.name})
            pl.set_l2_hint(out, store=pl.StoreL2Hint.{store_hint.name})
            t = pl.load(x, [0, 0], [32, 32])
            out = pl.store(t, [0, 0], out)
        return out
"""
    program = pl.parse_program(source)
    ir.assert_structural_equal(program, _reparse(program))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))
    text = ir.python_print(program)
    assert f"load=pl.LoadL2Hint.{load_hint.name}" in text
    assert f"store=pl.StoreL2Hint.{store_hint.name}" in text


def test_combined_scope_declaration_normalizes_and_deduplicates_directions():
    @pl.program
    class Both:
        @pl.function
        def main(self, x: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.set_l2_hint(x, load=pl.LoadL2Hint.NormalPersistent, store=pl.StoreL2Hint.NotAllocClean)
                pl.set_l2_hint(x, store=pl.StoreL2Hint.NotAllocClean)
                t = pl.load(x, [0, 0], [32, 32])
                x = pl.store(t, [0, 0], x)
            return x

    text = ir.python_print(Both)
    assert text.count("pl.set_l2_hint(") == 2
    ir.assert_structural_equal(Both, _reparse(Both))


@pytest.mark.parametrize(
    "declaration, diagnostic",
    [
        ("pl.set_l2_hint(x)", "takes one tensor"),
        ("pl.set_l2_hint(x, pl.LoadL2Hint.NormalPersistent)", "takes one tensor"),
        ("pl.set_l2_hint(x, store=pl.LoadL2Hint.NormalPersistent)", "Expected pl.StoreL2Hint"),
        ("pl.set_l2_hint(x, load=pl.StoreL2Hint.NormalPersistent)", "Expected pl.LoadL2Hint"),
        ("pl.set_l2_hint(x, unknown=pl.LoadL2Hint.NormalPersistent)", "invalid or repeated"),
        ("pl.set_l2_hint(x, load=2)", "Expected pl.LoadL2Hint"),
        (
            "pl.set_l2_hint(x, store=pl.StoreL2Hint.NormalPersistent)\n"
            "            pl.set_l2_hint(x, store=pl.StoreL2Hint.NormalFirstVictim)",
            "conflicting store hints",
        ),
    ],
)
def test_scope_rejects_invalid_l2_declarations(declaration, diagnostic):
    source = f"""
@pl.program
class Invalid:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32],
             out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            {declaration}
            t = pl.load(x, [0, 0], [32, 32])
            out = pl.store(t, [0, 0], out)
        return out
"""
    with pytest.raises(ParserSyntaxError, match=diagnostic):
        pl.parse_program(source)


@pytest.mark.parametrize("direction, hint, invalid", [("load", 5, 3), ("store", 4, 5)])
def test_deserialize_rejects_invalid_direction_specific_scope_encoding(direction, hint, invalid):
    enum_name = "LoadL2Hint" if direction == "load" else "StoreL2Hint"
    source = f"""
@pl.program
class Declared:
    @pl.function
    def main(self, x: pl.Tensor[[32, 32], pl.FP32],
             out: pl.Out[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_l2_hint({"x" if direction == "load" else "out"}, {direction}=pl.{enum_name}.NotAllocClean)
            t = pl.load(x, [0, 0], [32, 32])
            out = pl.store(t, [0, 0], out)
        return out
"""
    payload = ir.serialize(pl.parse_program(source))
    encoded = b"\xa6policy" + bytes([hint])
    assert payload.count(encoded) == 1
    malformed = payload.replace(encoded, b"\xa6policy" + bytes([invalid]))
    with pytest.raises(ValueError, match=f"invalid {direction} encoding"):
        ir.deserialize(malformed)


@pytest.mark.parametrize("load_hint", [None, pl.LoadL2Hint.NormalPersistent])
def test_scope_optional_closure_direction_preserves_other_hint(load_hint):
    @pl.program
    class Optional:
        @pl.function
        def main(self, x: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.set_l2_hint(x, load=load_hint, store=pl.StoreL2Hint.NormalFirstVictim)
                t = pl.load(x, [0, 0], [32, 32])
                x = pl.store(t, [0, 0], x)
            return x

    text = ir.python_print(Optional)
    assert "store=pl.StoreL2Hint.NormalFirstVictim" in text
    assert ("load=pl.LoadL2Hint.NormalPersistent" in text) == (load_hint is not None)
    ir.assert_structural_equal(Optional, _reparse(Optional))


@pytest.mark.parametrize("direction, other", [("load", "store"), ("store", "load")])
def test_scope_literal_none_preserves_other_zero_hint(direction, other):
    enum_name = "StoreL2Hint" if other == "store" else "LoadL2Hint"
    source = f"""
@pl.program
class Optional:
    @pl.function
    def main(self, x: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.set_l2_hint(x, {direction}=None, {other}=pl.{enum_name}.NormalFirstVictim)
            t = pl.load(x, [0, 0], [32, 32])
            x = pl.store(t, [0, 0], x)
        return x
"""
    program = pl.parse_program(source)
    text = ir.python_print(program)
    assert f"{direction}=pl." not in text
    assert f"{other}=pl.{enum_name}.NormalFirstVictim" in text
    ir.assert_structural_equal(program, _reparse(program))
    ir.assert_structural_equal(program, ir.deserialize(ir.serialize(program)))


@pytest.mark.parametrize(
    "load_hint, store_hint, diagnostic",
    [
        (pl.StoreL2Hint.NormalFirstVictim, None, "Expected pl.LoadL2Hint"),
        (None, pl.LoadL2Hint.NormalFirstVictim, "Expected pl.StoreL2Hint"),
        (0, pl.StoreL2Hint.NormalFirstVictim, "Expected pl.LoadL2Hint"),
        (False, pl.StoreL2Hint.NormalFirstVictim, "Expected pl.LoadL2Hint"),
        (None, None, "at least one non-None hint"),
    ],
)
def test_scope_rejects_invalid_captured_hints(load_hint, store_hint, diagnostic):
    with pytest.raises(ParserSyntaxError, match=diagnostic):

        @pl.program
        class Invalid:
            @pl.function
            def main(self, x: pl.InOut[pl.Tensor[[32, 32], pl.FP32]]) -> pl.Tensor[[32, 32], pl.FP32]:
                with pl.at(level=pl.Level.CORE_GROUP):
                    pl.set_l2_hint(x, load=load_hint, store=store_hint)
                    t = pl.load(x, [0, 0], [32, 32])
                    x = pl.store(t, [0, 0], x)
                return x


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
