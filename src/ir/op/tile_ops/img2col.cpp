/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::ir {
namespace {

TypePtr DeduceImg2colType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 4) << "tile.img2col requires src, pos_m, pos_k, shape";
  const auto& span = args[0]->span_;
  auto src = As<TileType>(args[0]->GetType());
  CHECK_SPAN(src && src->shape_.size() == 2, span) << "tile.img2col requires a 2D source tile";
  CHECK_SPAN(src->memory_space_ == MemorySpace::Mat, span) << "tile.img2col requires a Mat source";
  const auto dtype = src->dtype_;
  CHECK_SPAN(dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32 ||
                 dtype == DataType::INT8,
             span)
      << "tile.img2col supports FP16, BF16, FP32 and INT8";
  const int64_t c0 = dtype == DataType::INT8 ? 32 : (dtype == DataType::FP32 ? 8 : 16);
  auto rows = As<ConstInt>(src->shape_[0]);
  auto channels = As<ConstInt>(src->shape_[1]);
  CHECK_SPAN(rows && channels, span) << "tile.img2col requires a static source shape";
  CHECK_SPAN(rows->value_ > 0 && rows->value_ % 16 == 0 && channels->value_ > 0 &&
                 channels->value_ <= 65535 && channels->value_ % c0 == 0,
             span)
      << "tile.img2col requires H*W divisible by 16 and channels divisible by C0=32/sizeof(dtype)";
  const auto view = tile_view_semantics::GetEffectiveTileView(*src);
  CHECK_SPAN(
      view.blayout == TileLayout::col_major && view.slayout == TileLayout::row_major && view.fractal == 512,
      span)
      << "tile.img2col requires a canonical NZ source layout";
  CHECK_SPAN(view.stride.empty(), span) << "tile.img2col does not support a strided source view";
  CHECK_SPAN(tile_view_semantics::ShapeExprListsEquivalent(GetValidShape(src), src->shape_), span)
      << "tile.img2col requires a fully valid source tile";

  auto attr = [&](const char* name, int fallback, int64_t lower, int64_t upper) {
    const int64_t value = GetKwargOr<int>(kwargs, name, fallback);
    CHECK_SPAN(value >= lower && value <= upper, span)
        << "tile.img2col " << name << " must be in [" << lower << ", " << upper << "]";
    return value;
  };
  const int64_t h = attr("fmap_h", 0, 1, 65535);
  const int64_t w = attr("fmap_w", 0, 1, 65535);
  const int64_t kh = attr("kernel_h", 0, 1, 511);
  const int64_t kw = attr("kernel_w", 0, 1, 511);
  const int64_t sh = attr("stride_h", 1, 1, 255);
  const int64_t sw = attr("stride_w", 1, 1, 255);
  const int64_t dh = attr("dilation_h", 1, 1, 255);
  const int64_t dw = attr("dilation_w", 1, 1, 255);
  const int64_t pt = attr("pad_top", 0, 0, 255);
  const int64_t pb = attr("pad_bottom", 0, 0, 255);
  const int64_t pl = attr("pad_left", 0, 0, 255);
  const int64_t pr = attr("pad_right", 0, 0, 255);
  CHECK_SPAN(h * w == rows->value_, span) << "tile.img2col source rows must equal H*W";
  const int64_t padded_h = h + pt + pb - dh * (kh - 1) - 1;
  const int64_t padded_w = w + pl + pr - dw * (kw - 1) - 1;
  CHECK_SPAN(padded_h >= 0 && padded_w >= 0, span) << "tile.img2col kernel exceeds padded image";
  const std::array<int64_t, 2> bounds = {(padded_h / sh + 1) * (padded_w / sw + 1),
                                         channels->value_ * kh * kw};
  auto shape = As<MakeTuple>(args[3]);
  CHECK_SPAN(shape && shape->elements_.size() == 2, span) << "tile.img2col shape must be a static pair";
  for (size_t axis = 0; axis < 2; ++axis) {
    auto dim = As<ConstInt>(shape->elements_[axis]);
    const int64_t alignment = axis == 0 ? 16 : c0;
    CHECK_SPAN(dim && dim->value_ > 0 && dim->value_ <= 65535 && dim->value_ % alignment == 0, span)
        << "tile.img2col shape must be positive, uint16-sized and aligned to (16, C0)";
    CHECK_SPAN(dim->value_ <= bounds[axis], span) << "tile.img2col shape exceeds unfolded image";
    auto index_type = As<ScalarType>(args[axis + 1]->GetType());
    CHECK_SPAN(index_type && index_type->dtype_.IsIndexLike(), span)
        << "tile.img2col positions must be index-like scalars";
    if (auto pos = As<ConstInt>(args[axis + 1])) {
      CHECK_SPAN(pos->value_ >= 0 && pos->value_ <= 65535 && pos->value_ + dim->value_ <= bounds[axis], span)
          << "tile.img2col position is outside the unfolded image";
      CHECK_SPAN(axis == 0 || pos->value_ % c0 == 0, span) << "tile.img2col pos_k must be C0-aligned";
    }
  }
  TileView result_view;
  result_view.blayout = TileLayout::row_major;
  result_view.slayout = TileLayout::row_major;
  return std::make_shared<TileType>(shape->elements_, dtype, std::nullopt, result_view, MemorySpace::Left);
}

REGISTER_OP("tile.img2col")
    .set_op_category("TileOp")
    .set_description("Unfold an L1 feature map into an L0A tile using TIMG2COL")
    .add_argument("src", "Full NZ feature map [H*W, C] in Mat memory")
    .add_argument("pos_m", "Starting flattened output spatial position")
    .add_argument("pos_k", "Starting C1, KH, KW, C0 position")
    .add_argument("shape", "Static destination shape [M, K]")
    .set_attr<int>("fmap_h")
    .set_attr<int>("fmap_w")
    .set_attr<int>("kernel_h")
    .set_attr<int>("kernel_w")
    .set_attr<int>("stride_h")
    .set_attr<int>("stride_w")
    .set_attr<int>("dilation_h")
    .set_attr<int>("dilation_w")
    .set_attr<int>("pad_top")
    .set_attr<int>("pad_bottom")
    .set_attr<int>("pad_left")
    .set_attr<int>("pad_right")
    .set_input_memory(0, MemorySpace::Mat)
    .set_output_memory(MemorySpace::Left)
    .not_inplace_safe()
    .f_deduce_type(DeduceImg2colType);

}  // namespace
}  // namespace pypto::ir
