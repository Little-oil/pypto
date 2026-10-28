# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tensor comparisons retain their dynamic valid region through PTOAS."""

import pypto.language as pl
import pytest
from pypto import codegen, ir
from pypto.backend import BackendType
from pypto.backend._ptoas_locate import find_ptoas_binary
from pypto.backend.pto_backend import _run_ptoas


@pytest.mark.ptoas_compile
@pytest.mark.parametrize("ascend_backend", [BackendType.Ascend910, BackendType.Ascend950], indirect=True)
def test_dynamic_cmp_valid_region_compiles(ascend_backend, default_pass_manager, tmp_path):
    """Exercise the GLM vision cmp/cast tail without a model or device dependency."""
    if find_ptoas_binary() is None:
        pytest.skip("PTOAS is not available")

    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def kernel(
            self,
            x: pl.Tensor[[8, 128], pl.FP32],
            rows: pl.Scalar[pl.INDEX],
            cols: pl.Scalar[pl.INDEX],
            out: pl.Out[pl.Tensor[[8, 128], pl.INT32]],
        ):
            source = pl.slice(x, [8, 128], [0, 0], valid_shape=[rows, cols])
            predicate = pl.cmp(source, 0.0, cmp_type=5)
            values = pl.cast(predicate, pl.INT32)
            out = pl.assemble(out, values, [0, 0])
            return out

    lowered = default_pass_manager.run_passes(Program)
    kernels = [func for func in lowered.functions.values() if func.func_type != pl.FunctionType.Orchestration]
    assert len(kernels) == 1
    mlir = codegen.PTOCodegen().generate(ir.Program(kernels, "CmpValidRegion", lowered.span))
    assert "pto.tcmps" in mlir and "pto.tsel" in mlir
    # Both dynamic extents must reach the final store, rather than being widened
    # by the comparison's materialized full-capacity one/zero tiles.
    assert "!pto.partition_tensor_view<?x?xi32>" in mlir
    source, output = tmp_path / "cmp_valid.pto", tmp_path / "cmp_valid.cpp"
    source.write_text(mlir)
    arch = "a2" if ascend_backend == BackendType.Ascend910 else "a5"
    _run_ptoas(str(source), str(output), [f"--pto-arch={arch}", "--pto-level=level3"])
    assert output.is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
