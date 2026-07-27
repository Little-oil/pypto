# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the English/zh-CN documentation path parity check."""

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_docs_parity() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "lint" / "check_docs_en_zh_parity.py"
    spec = importlib.util.spec_from_file_location("pypto_check_docs_en_zh_parity", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


docs_parity = _load_docs_parity()


def test_allowlisted_ptoas_docs_may_be_zh_only():
    en = {"dev/paired.md"}
    zh = {
        "dev/paired.md",
        "dev/ptoas-op-addition-plan.md",
        "dev/ptoas-op-status.md",
    }

    assert docs_parity._find_unpaired_paths(en, zh) == ([], [])


def test_other_zh_only_path_is_reported():
    en = {"dev/paired.md"}
    zh = {
        "dev/paired.md",
        "dev/ptoas-op-addition-plan.md",
        "dev/ptoas-op-status.md",
        "dev/unexpected.md",
    }

    assert docs_parity._find_unpaired_paths(en, zh) == ([], ["dev/unexpected.md"])


def test_en_only_path_is_reported_even_with_allowlisted_zh_path():
    en = {"dev/paired.md", "dev/missing-translation.md"}
    zh = {
        "dev/paired.md",
        "dev/ptoas-op-addition-plan.md",
        "dev/ptoas-op-status.md",
    }

    assert docs_parity._find_unpaired_paths(en, zh) == (["dev/missing-translation.md"], [])
