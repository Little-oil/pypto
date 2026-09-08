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
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto::ir {
namespace {

using OpAttrs = std::vector<std::pair<std::string, std::any>>;

// Composite results are fresh values. In particular, never propagate the
// input's MemRef, strides, or distributed-window identity onto a result.
TypePtr DeduceComposite(const std::vector<ExprPtr>& args, const std::string& name, bool tile) {
  CHECK(args.size() == 1) << name << " requires one input, got " << args.size();
  const auto input = As<ShapedType>(args[0]->GetType());
  CHECK(tile ? static_cast<bool>(As<TileType>(args[0]->GetType()))
             : static_cast<bool>(AsTensorTypeLike(args[0]->GetType())))
      << name << (tile ? " requires a Tile" : " requires a Tensor");
  CHECK(input->dtype_ == DataType::FP16 || input->dtype_ == DataType::FP32)
      << name << " requires FP16 or FP32 input, got " << input->dtype_.ToString();
  CHECK(!input->shape_.empty()) << name << " requires non-scalar input";

  if (tile) {
    const auto tile_type = As<TileType>(args[0]->GetType());
    TileView view;
    view.valid_shape = GetValidShape(tile_type);
    InheritTileViewLayout(view, tile_type);
    return std::make_shared<TileType>(input->shape_, input->dtype_, std::nullopt, view);
  }
  return MakeFreshTensorType(input->shape_, input->dtype_,
                             GetValidShape(AsTensorTypeLike(args[0]->GetType())));
}

TypePtr DeducePower(const std::vector<ExprPtr>& args, const OpAttrs& kwargs, bool tile) {
  const auto result = DeduceComposite(args, "pow", tile);
  const double exponent = GetKwargOr<double>(kwargs, "exponent", std::numeric_limits<double>::quiet_NaN());
  CHECK(std::isfinite(exponent) && std::abs(exponent) <= 2147483647.0)
      << "pow requires a finite scalar exponent with abs(exponent) <= 2**31-1, got " << exponent;
  return result;
}

TypePtr DeduceClamp(const std::vector<ExprPtr>& args, const OpAttrs& kwargs, bool tile) {
  const auto result = DeduceComposite(args, "clamp", tile);
  bool has_bound = false;
  for (const auto& kwarg : kwargs) {
    const auto& key = kwarg.first;
    if (key != "min" && key != "max") continue;
    has_bound = true;
    const double bound = GetKwargOr<double>(kwargs, key, 0.0);
    CHECK(std::isfinite(bound) && std::abs(bound) <= std::numeric_limits<float>::max())
        << "clamp requires finite FP32-representable scalar bounds, got " << key << "=" << bound;
  }
  CHECK(has_bound) << "clamp requires at least one of min or max";
  return result;
}

TypePtr DeduceMean(const std::vector<ExprPtr>& args, const OpAttrs& kwargs, bool tile) {
  const auto base = DeduceComposite(args, "mean", tile);
  const auto input = As<ShapedType>(base);
  CHECK(input->shape_.size() == 2) << "mean requires rank-2 input, got rank " << input->shape_.size();
  const int requested_axis = GetKwargOr<int>(kwargs, "axis", -1);
  int axis = requested_axis;
  if (axis < 0) axis += 2;
  CHECK(axis == 0 || axis == 1) << "mean requires axis 0, 1, -1, or -2, got " << requested_axis;
  auto valid = tile ? GetValidShape(As<TileType>(base)) : GetValidShape(AsTensorTypeLike(base));
  CheckReductionInputNonEmpty(valid, "mean", args[0]->span_);
  const auto count = As<ConstInt>(valid[axis]);
  CHECK(count && count->value_ > 0) << "mean requires a positive static valid extent on the reduced axis";
  auto shape = input->shape_;
  shape[axis] = valid[axis] = std::make_shared<ConstInt>(1, DataType::INDEX, args[0]->span_);
  if (tile) {
    const int64_t block = 32 / input->dtype_.GetByte();
    if (const auto extent = As<ConstInt>(shape[1 - axis]); extent && extent->value_ % block != 0) {
      shape[1 - axis] = std::make_shared<ConstInt>(((extent->value_ + block - 1) / block) * block,
                                                   DataType::INDEX, args[0]->span_);
    }
    TileView view;
    view.valid_shape = valid;
    view.blayout = axis == 1 ? TileLayout::col_major : TileLayout::row_major;
    return std::make_shared<TileType>(shape, input->dtype_, std::nullopt, view);
  }
  return MakeFreshTensorType(shape, input->dtype_, valid);
}

}  // namespace

REGISTER_OP("tile.pow")
    .set_op_category("TileOp")
    .set_description("Elementwise scalar power; FP32 intermediates")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<double>("exponent")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeducePower(args, kwargs, true);
    });

REGISTER_OP("tensor.pow")
    .set_op_category("TensorOp")
    .set_description("Elementwise scalar power; FP32 intermediates")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<double>("exponent")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeducePower(args, kwargs, false);
    });

REGISTER_OP("tile.clamp")
    .set_op_category("TileOp")
    .set_description("Elementwise clamp with optional scalar lower/upper bounds")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<double>("min")
    .set_attr<double>("max")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeduceClamp(args, kwargs, true);
    });

REGISTER_OP("tensor.clamp")
    .set_op_category("TensorOp")
    .set_description("Elementwise clamp with optional scalar lower/upper bounds")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<double>("min")
    .set_attr<double>("max")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeduceClamp(args, kwargs, false);
    });

REGISTER_OP("tile.mean")
    .set_op_category("TileOp")
    .set_description("Rank-2 arithmetic mean with keepdim; FP32 accumulation")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<int>("axis")
    .set_input_memory(0, MemorySpace::Vec)
    .set_output_memory(MemorySpace::Vec)
    .functional_execution_memory_access()
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeduceMean(args, kwargs, true);
    });

REGISTER_OP("tensor.mean")
    .set_op_category("TensorOp")
    .set_description("Rank-2 arithmetic mean with keepdim; FP32 accumulation")
    .add_argument("input", "FP16 or FP32 input")
    .set_attr<int>("axis")
    .f_deduce_type([](const std::vector<ExprPtr>& args, const OpAttrs& kwargs) {
      return DeduceMean(args, kwargs, false);
    });

}  // namespace pypto::ir
