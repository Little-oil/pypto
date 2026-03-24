# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for PTO backend codegen for paged attention operations."""

import pypto.language as pl
import pytest
from pypto import backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen


@pl.program
class PagedAttention:
    """
    Case1:
    batch 256
    num_heads 16
    kv_head_num 1
    head_dim 128
    block_size 128
    max_num_blocks_per_req 256
    scale_value 1
    Q(256, 16, 128) BF16
    K(16384, 128, 1, 128) BF16
    V(16384, 128, 1, 128) BF16
    block_table(256, 256) INT32
    context_lens(256, ) INT32
    out(524288, ) FP32
    """

    """
    orchestration config

    q_tile_size 16

    num_head_tiles 1
    sij_size 16 * 128 float
    pij_size 16 * 128 uint16
    mij_size 16 float
    lij_size 16 float
    oi_new_size 16 * 128 float

    mi_size 16 float
    li_size 16 float
    oi_size 16 * 128 float

    qi(256, 16, 128) BF16
    out()
    """

    # M K N 16 128 128

    # AIC kernels

    @pl.function(type=pl.FunctionType.InCore)
    def qk_matmul(
        self,
        qi: pl.Tensor[[16, 128], pl.BF16],
        kj: pl.Tensor[[128, 128], pl.BF16, pl.DN],
        s_ij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        q_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        k_tile_T: pl.Tile[[128, 128], pl.BF16] = pl.load(
            kj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
        )
        s_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(q_tile, k_tile_T)
        updated_sij: pl.Tensor[[16, 128], pl.FP32] = pl.store(s_tile, [0, 0], s_ij)
        return updated_sij

    @pl.function(type=pl.FunctionType.InCore)
    def pv_matmul(
        self,
        pij: pl.Tensor[[16, 128], pl.BF16],
        vj: pl.Tensor[[128, 128], pl.BF16],
        oij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        p_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
            pij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
        )
        v_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
            vj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
        )
        o_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(p_tile, v_tile)
        updated_oij: pl.Tensor[[16, 128], pl.FP32] = pl.store(o_tile, [0, 0], oij)
        return updated_oij

    # AIV kernels

    @pl.function(type=pl.FunctionType.InCore)
    def softmax_prepare(
        self,
        sij: pl.Tensor[[16, 128], pl.FP32],
        pij: pl.Tensor[[16, 128], pl.BF16],
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        scale_value: pl.Scalar[pl.FP32],
    ):
        sij_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(sij, [0, 0], [16, 128])
        # sij_dyn_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(
        #     sij, [0, 0], [16, 128]
        # )
        # TODO: <TileType::Vec, float, M, N, BLayout::RowMajor, M, -1
        # sij_pad_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(
        #     sij, [0, 0], [16, 128]
        # )
        # TODO: <TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>
        _pij_tile_bf16: pl.Tile[[16, 128], pl.BF16] = pl.load(pij, [0, 0], [16, 128])
        tmp_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.sub(sij_tile, sij_tile)
        sij_padded = pl.tile.fillpad(sij_tile)
        sij_scaled = pl.tile.muls(sij_padded, scale_value)
        max_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_max(sij_scaled, tmp_tile)
        pij_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_sub(sij_scaled, max_tile)
        pij_tile = pl.tile.exp(pij_tile)
        pij_bf16_tile = pl.tile.cast(pij_tile, mode="round", target_type=pl.BF16)
        pij_fp16_tile = pl.tile.cast(pij_bf16_tile, mode="round", target_type=pl.FP16)
        sum_tile: pl.Tile[[16, 1], pl.FP16] = pl.tile.row_sum(pij_fp16_tile, tmp_tile)
        pl.store(max_tile, [0, 0], mij)
        pl.store(sum_tile, [0, 0], lij)
        pl.store(pij_bf16_tile, [0, 0], pij)

    @pl.function(type=pl.FunctionType.InCore)
    def online_update(
        self,
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        oi_new: pl.Tensor[[16, 128], pl.FP32],
        mi: pl.Tensor[[16, 1], pl.FP32],
        li: pl.Tensor[[16, 1], pl.FP32],
        oi: pl.Tensor[[16, 128], pl.FP32],
        dst: pl.Tensor[[16, 128], pl.FP32],
    ):
        oi_new_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi_new, [0, 0], [16, 128])
        oi_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi, [0, 0], [16, 128])
        mij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mij, [0, 0], [16, 1])
        lij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(lij, [0, 0], [16, 1])
        mi_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mi, [0, 0], [16, 1])
        li_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(li, [0, 0], [16, 1])

        mi_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.maximum(mi_tile, mij_tile)

        alpha_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mi_tile, mi_new_tile)
        alpha_tile = pl.tile.exp(alpha_tile)

        beta_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mij_tile, mi_new_tile)
        beta_tile = pl.tile.exp(beta_tile)

        li_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(alpha_tile, li_tile)
        lij_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(beta_tile, lij_tile)
        li_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(li_scaled, lij_scaled)

        oi_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_tile, alpha_tile)
        oi_new_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_new_tile, beta_tile)
        oi_updated_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.add(oi_scaled, oi_new_scaled)

        dst_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_div(oi_updated_tile, li_new_tile)

        pl.store(mi_new_tile, [0, 0], mi)
        pl.store(li_new_tile, [0, 0], li)
        pl.store(oi_updated_tile, [0, 0], oi)
        pl.store(dst_tile, [0, 0], dst)


def test_tile_ops_codegen():
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    program = PagedAttention
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized_program = pm.run_passes(program)
    codegen_instance = codegen.PTOCodegen()

    for func in optimized_program.functions.values():
        func_name = func.name
        single_func_program = ir.Program([func], func_name, optimized_program.span)
        mlir_code = codegen_instance.generate(single_func_program)
        assert mlir_code, f"Generated MLIR code for {func_name} should not be empty"


@pl.program
class UnalignedPagedAttention:
    """Unaligned paged attention: softmax_prepare uses valid_len + valid_shapes + fillpad.

    Other kernels (qk_matmul, pv_matmul, online_update) are identical to PagedAttention.
    """

    @pl.function(type=pl.FunctionType.InCore)
    def qk_matmul(
        self,
        qi: pl.Tensor[[16, 128], pl.BF16],
        kj: pl.Tensor[[128, 128], pl.BF16, pl.DN],
        s_ij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        q_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(qi, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat)
        k_tile_T: pl.Tile[[128, 128], pl.BF16] = pl.load(
            kj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat, transpose=True
        )
        s_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(q_tile, k_tile_T)
        updated_sij: pl.Tensor[[16, 128], pl.FP32] = pl.store(s_tile, [0, 0], s_ij)
        return updated_sij

    @pl.function(type=pl.FunctionType.InCore)
    def pv_matmul(
        self,
        pij: pl.Tensor[[16, 128], pl.BF16],
        vj: pl.Tensor[[128, 128], pl.BF16],
        oij: pl.Tensor[[16, 128], pl.FP32],
    ) -> pl.Tensor[[16, 128], pl.FP32]:
        p_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(
            pij, [0, 0], [16, 128], target_memory=pl.MemorySpace.Mat
        )
        v_tile: pl.Tile[[128, 128], pl.BF16] = pl.load(
            vj, [0, 0], [128, 128], target_memory=pl.MemorySpace.Mat
        )
        o_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.matmul(p_tile, v_tile)
        updated_oij: pl.Tensor[[16, 128], pl.FP32] = pl.store(o_tile, [0, 0], oij)
        return updated_oij

    @pl.function(type=pl.FunctionType.InCore)
    def softmax_prepare(
        self,
        sij: pl.Tensor[[16, 128], pl.FP32],
        pij: pl.Tensor[[16, 128], pl.BF16],
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        scale_value: pl.Scalar[pl.FP32],
        valid_len: pl.Scalar[pl.INDEX],
    ):
        sij_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(sij, [0, 0], [16, 128], valid_shapes=[16, valid_len])
        _pij_tile: pl.Tile[[16, 128], pl.BF16] = pl.load(pij, [0, 0], [16, 128])
        tmp_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.sub(sij_tile, sij_tile)
        sij_padded: pl.Tile[[16, 128], pl.FP32] = pl.tile.fillpad(sij_tile, pad_value=pl.PadValue.min)
        sij_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.muls(sij_padded, scale_value)
        max_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.row_max(sij_scaled, tmp_tile)
        pij_fp32_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_sub(sij_scaled, max_tile)
        pij_fp32_tile = pl.tile.exp(pij_fp32_tile)
        pij_bf16_tile: pl.Tile[[16, 128], pl.BF16] = pl.tile.cast(
            pij_fp32_tile, mode="round", target_type=pl.BF16
        )
        pij_fp16_tile: pl.Tile[[16, 128], pl.FP16] = pl.tile.cast(
            pij_bf16_tile, mode="round", target_type=pl.FP16
        )
        sum_tile: pl.Tile[[16, 1], pl.FP16] = pl.tile.row_sum(pij_fp16_tile, tmp_tile)
        pl.store(max_tile, [0, 0], mij)
        pl.store(sum_tile, [0, 0], lij)
        pl.store(pij_bf16_tile, [0, 0], pij)
        return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement

    @pl.function(type=pl.FunctionType.InCore)
    def online_update(
        self,
        mij: pl.Tensor[[16, 1], pl.FP32],
        lij: pl.Tensor[[16, 1], pl.FP32],
        oi_new: pl.Tensor[[16, 128], pl.FP32],
        mi: pl.Tensor[[16, 1], pl.FP32],
        li: pl.Tensor[[16, 1], pl.FP32],
        oi: pl.Tensor[[16, 128], pl.FP32],
        dst: pl.Tensor[[16, 128], pl.FP32],
    ):
        oi_new_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi_new, [0, 0], [16, 128])
        oi_tile: pl.Tile[[16, 128], pl.FP32] = pl.load(oi, [0, 0], [16, 128])
        mij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mij, [0, 0], [16, 1])
        lij_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(lij, [0, 0], [16, 1])
        mi_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(mi, [0, 0], [16, 1])
        li_tile: pl.Tile[[16, 1], pl.FP32] = pl.load(li, [0, 0], [16, 1])

        mi_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.maximum(mi_tile, mij_tile)

        alpha_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mi_tile, mi_new_tile)
        alpha_tile = pl.tile.exp(alpha_tile)

        beta_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.sub(mij_tile, mi_new_tile)
        beta_tile = pl.tile.exp(beta_tile)

        li_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(alpha_tile, li_tile)
        lij_scaled: pl.Tile[[16, 1], pl.FP32] = pl.tile.mul(beta_tile, lij_tile)
        li_new_tile: pl.Tile[[16, 1], pl.FP32] = pl.tile.add(li_scaled, lij_scaled)

        oi_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_tile, alpha_tile)
        oi_new_scaled: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_mul(oi_new_tile, beta_tile)
        oi_updated_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.add(oi_scaled, oi_new_scaled)

        dst_tile: pl.Tile[[16, 128], pl.FP32] = pl.tile.row_expand_div(oi_updated_tile, li_new_tile)

        pl.store(mi_new_tile, [0, 0], mi)
        pl.store(li_new_tile, [0, 0], li)
        pl.store(oi_updated_tile, [0, 0], oi)
        pl.store(dst_tile, [0, 0], dst)
        return  # noqa: PLR1711 - DSL requires explicit return to build IR return statement


def test_unaligned_tile_ops_codegen():
    """Test PTO codegen for unaligned paged attention (softmax with valid_len + fillpad)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B_PTO)

    program = UnalignedPagedAttention
    pm = PassManager.get_strategy(OptimizationStrategy.Default)
    optimized_program = pm.run_passes(program)
    codegen_instance = codegen.PTOCodegen()

    for func in optimized_program.functions.values():
        func_name = func.name
        single_func_program = ir.Program([func], func_name, optimized_program.span)
        mlir_code = codegen_instance.generate(single_func_program)
        assert mlir_code, f"Generated MLIR code for {func_name} should not be empty"
        if func_name == "softmax_prepare":
            # Verify that the softmax_prepare MLIR contains fillpad with min padding
            assert (
                "fillpad" in mlir_code.lower()
                or "setvalidshape" in mlir_code.lower()
                or "PadValue" in mlir_code
            ), f"softmax_prepare MLIR should contain fillpad/padding related ops:\n{mlir_code}"

            # Verify sij_padded alloc_tile has STATIC v_col and pad=3 (min).
            # TFILLPAD reads the padding boundary from the INPUT tile's v_col,
            # NOT the output's.  After fillpad all positions are valid (real data
            # or pad value), so the output must keep static v_col so downstream
            # compute ops (TMULS, TROWMAX, …) process the full tile width.
            padded_alloc_lines = [
                line for line in mlir_code.split("\n") if "sij_padded" in line and "alloc_tile" in line
            ]
            assert padded_alloc_lines, f"softmax_prepare MLIR should have sij_padded alloc_tile:\n{mlir_code}"
            padded_line = padded_alloc_lines[0]
            assert "v_col=?" not in padded_line, (
                f"sij_padded alloc_tile should have STATIC v_col (not v_col=?) "
                f"so downstream ops process all columns, got:\n{padded_line}"
            )
            assert "pad=3" in padded_line, (
                f"sij_padded alloc_tile should have pad=3 (min padding), got:\n{padded_line}"
            )

            # Verify sij_tile alloc_tile has dynamic v_col (PTOAS needs it for TFILLPAD boundary)
            sij_tile_alloc_lines = [
                line for line in mlir_code.split("\n") if "sij_tile" in line and "alloc_tile" in line
            ]
            if sij_tile_alloc_lines:
                assert "v_col=?" in sij_tile_alloc_lines[0], (
                    f"sij_tile alloc_tile should have dynamic v_col (v_col=?) "
                    f"so PTOAS creates both static and dynamic view tiles, "
                    f"got:\n{sij_tile_alloc_lines[0]}"
                )

            # Verify tload outs uses a STATIC tile buffer (not the dynamic sij_tile).
            # TLOAD with dynamic v_col causes DMA row stride mismatch, so we load
            # into a static temp tile and then tmov to the dynamic tile.
            tload_lines = [line for line in mlir_code.split("\n") if "pto.tload" in line]
            assert tload_lines, f"softmax_prepare MLIR should contain pto.tload:\n{mlir_code}"
            for tload_line in tload_lines:
                assert "v_col=?" not in tload_line, (
                    f"pto.tload outs type must use STATIC v_col (not v_col=?) "
                    f"so PTOAS uses the static buffer tile for DMA, got:\n{tload_line}"
                )

            # Verify tmov copies from static tload buffer to dynamic tile
            tmov_lines = [line for line in mlir_code.split("\n") if "pto.tmov" in line]
            assert tmov_lines, (
                f"softmax_prepare MLIR should contain pto.tmov "
                f"(copies tload result to dynamic tile):\n{mlir_code}"
            )


if __name__ == "__main__":
    pytest.main([__file__])
