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

#include "pypto/ir/transforms/utils/buffer_root_collector.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/result_alias_utils.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace buffer_root {
namespace {

using op_predicates::IsBuiltinOp;

}  // namespace

BufferRootCollector::BufferRootCollector(ProgramPtr program, AmbiguousRootPolicy ambiguous_policy)
    : program_(std::move(program)), ambiguous_policy_(ambiguous_policy) {}

void BufferRootCollector::Initialize(const std::vector<VarPtr>& params) {
  for (const auto& param : params) {
    RecordRootCandidates(param, {param.get()});
  }
}

void BufferRootCollector::VisitStmt_(const IfStmtPtr& if_stmt) {
  if (track_tile_storage_) {
    static_cast<void>(StorageValue(if_stmt->condition_));
    const auto then_values = StorageRegion(if_stmt->then_body_);
    const auto else_values =
        if_stmt->else_body_ ? StorageRegion(*if_stmt->else_body_) : std::vector<std::optional<StorageId>>{};
    for (size_t i = 0; i < if_stmt->return_vars_.size(); ++i) {
      const auto& var = if_stmt->return_vars_[i];
      if (As<TupleType>(var->GetType())) {
        MarkUnresolvedTileTuple(var->GetType());
        continue;
      }
      auto storage = i < then_values.size() ? then_values[i] : std::nullopt;
      if (i < else_values.size()) storage = StorageChoice(storage, else_values[i]);
      if (!if_stmt->else_body_) storage = StorageChoice(storage, StorageValue(var));
      BindStorage(var, storage);
    }
  } else {
    IRVisitor::VisitStmt_(if_stmt);
  }
  if (if_stmt->return_vars_.empty() || !if_stmt->else_body_.has_value()) return;

  auto then_yield = transform_utils::GetLastYieldStmt(if_stmt->then_body_);
  auto else_yield = transform_utils::GetLastYieldStmt(*if_stmt->else_body_);
  if (!then_yield || !else_yield) return;

  for (size_t i = 0;
       i < if_stmt->return_vars_.size() && i < then_yield->value_.size() && i < else_yield->value_.size();
       ++i) {
    auto roots = ResolveRootCandidates(then_yield->value_[i]);
    auto else_roots = ResolveRootCandidates(else_yield->value_[i]);
    for (const Var* root : else_roots) {
      if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
    }
    const bool unresolved = roots.empty() || IsAmbiguous(then_yield->value_[i]) ||
                            IsAmbiguous(else_yield->value_[i]) ||
                            ResolveRootCandidates(then_yield->value_[i]).empty() || else_roots.empty();
    RecordRootCandidates(if_stmt->return_vars_[i], roots, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const ForStmtPtr& for_stmt) {
  auto init_roots = InitializeLoopCarryRoots(for_stmt->iter_args_);
  if (track_tile_storage_) {
    static_cast<void>(StorageValue(for_stmt->start_));
    static_cast<void>(StorageValue(for_stmt->stop_));
    static_cast<void>(StorageValue(for_stmt->step_));
    StorageLoop(for_stmt->iter_args_, for_stmt->return_vars_, for_stmt->body_);
  } else {
    IRVisitor::VisitStmt_(for_stmt);
  }
  RecordLoopReturnRoots(for_stmt->body_, for_stmt->return_vars_, init_roots,
                        transform_utils::EvalConstTripCount(for_stmt).value_or(0) > 0);
}

void BufferRootCollector::VisitStmt_(const WhileStmtPtr& while_stmt) {
  auto init_roots = InitializeLoopCarryRoots(while_stmt->iter_args_);
  if (track_tile_storage_) {
    static_cast<void>(StorageValue(while_stmt->condition_));
    StorageLoop(while_stmt->iter_args_, while_stmt->return_vars_, while_stmt->body_);
  } else {
    IRVisitor::VisitStmt_(while_stmt);
  }
  // A while condition may be false before the first iteration, so both the
  // initial and yielded roots remain possible.
  RecordLoopReturnRoots(while_stmt->body_, while_stmt->return_vars_, init_roots,
                        /*guaranteed_to_run=*/false);
}

std::vector<BufferRootCollector::RootCandidates> BufferRootCollector::InitializeLoopCarryRoots(
    const std::vector<IterArgPtr>& iter_args) {
  std::vector<RootCandidates> init_roots;
  init_roots.reserve(iter_args.size());
  for (const auto& iter_arg : iter_args) {
    auto roots = ResolveRootCandidates(iter_arg->initValue_);
    init_roots.push_back(roots);
    RecordRootCandidates(iter_arg, roots, IsAmbiguous(iter_arg->initValue_));
  }
  return init_roots;
}

void BufferRootCollector::RecordLoopReturnRoots(const StmtPtr& body, const std::vector<VarPtr>& return_vars,
                                                const std::vector<RootCandidates>& init_roots,
                                                bool guaranteed_to_run) {
  // Resolve after visiting the body so fresh-rebind assignments already have
  // roots. An in-place yield of the IterArg still resolves to its init root.
  auto yield = transform_utils::GetLastYieldStmt(body);
  if (!yield) return;

  for (size_t i = 0; i < return_vars.size() && i < yield->value_.size(); ++i) {
    auto roots = ResolveRootCandidates(yield->value_[i]);
    bool unresolved = roots.empty() || IsAmbiguous(yield->value_[i]);
    if (!guaranteed_to_run && i < init_roots.size()) {
      if (init_roots[i].empty()) unresolved = true;
      for (const Var* root : init_roots[i]) {
        if (std::find(roots.begin(), roots.end(), root) == roots.end()) roots.push_back(root);
      }
    }
    RecordRootCandidates(return_vars[i], roots, unresolved);
  }
}

void BufferRootCollector::VisitStmt_(const AssignStmtPtr& assign) {
  if (track_tile_storage_) {
    if (As<TupleType>(assign->var_->GetType())) {
      BindTileTuple(assign->var_, TileTuple(assign->value_));
    } else {
      BindStorage(assign->var_, StorageValue(assign->value_));
    }
  }
  // Submit (pl.submit inside a pl.manual_scope) is a sibling call-like kind;
  // route it through the Call-shaped view so a task launch's Out/InOut roots are
  // tracked identically to a plain Call (results keyed on the stable binding
  // Var). The view preserves args_ and the TASK_ID-augmented return type, so the
  // tuple path maps the submit-result projections to the callee's Out roots and
  // leaves the trailing TASK_ID element unmapped
  // (see .claude/rules/pass-submit-awareness.md).
  if (auto call = transform_utils::AsCallOrSubmitView(assign->value_)) {
    const std::string& op_name = call->op_->name_;
    if (IsOp(call, "tensor.create") || IsOp(call, "tensor.slice")) {
      RecordRootCandidates(assign->var_, {assign->var_.get()});
    } else if (IsOp(call, "tensor.assemble")) {
      if (call->args_.size() == 3) {
        RecordRootCandidates(assign->var_, ResolveRootCandidates(call->args_[0]),
                             IsAmbiguous(call->args_[0]));
      }
    } else if (!IsBuiltinOp(op_name)) {
      auto out_roots = CollectCallOutputRoots(call);
      if (As<TupleType>(call->GetType())) {
        std::vector<RootInfo> roots;
        roots.reserve(out_roots.size());
        for (const auto& entry : out_roots) roots.push_back(entry.info);
        tuple_output_roots_[assign->var_.get()] = std::move(roots);
      } else {
        auto info = SelectReturnRootInfo(out_roots, call->GetType());
        if (!info.roots.empty()) {
          RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
        }
      }
    }
  } else if (auto tuple_get = As<TupleGetItemExpr>(assign->value_)) {
    if (auto tuple_var = AsVarLike(tuple_get->tuple_)) {
      auto it = tuple_output_roots_.find(tuple_var.get());
      if (it != tuple_output_roots_.end() && tuple_get->index_ < static_cast<int>(it->second.size()) &&
          !it->second[tuple_get->index_].roots.empty()) {
        const auto& info = it->second[tuple_get->index_];
        RecordRootCandidates(assign->var_, info.roots, info.ambiguous);
      }
    }
  } else if (AsVarLike(assign->value_)) {
    RecordRootCandidates(assign->var_, ResolveRootCandidates(assign->value_), IsAmbiguous(assign->value_));
  }
  if (!track_tile_storage_) IRVisitor::VisitStmt_(assign);
}

BufferRootCollector::RootCandidates BufferRootCollector::ResolveRootCandidates(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  if (!var) return {};
  auto it = root_candidates_.find(var.get());
  return it != root_candidates_.end() ? it->second : RootCandidates{};
}

bool BufferRootCollector::IsAmbiguous(const ExprPtr& expr) const {
  auto var = AsVarLike(expr);
  return var && ambiguous_buffer_vars.count(var.get()) != 0;
}

void BufferRootCollector::RecordRootCandidates(const VarPtr& var, const RootCandidates& roots,
                                               bool unresolved) {
  if (!var) return;
  RootCandidates unique;
  unique.reserve(roots.size());
  for (const Var* root : roots) {
    if (root && std::find(unique.begin(), unique.end(), root) == unique.end()) unique.push_back(root);
  }
  if (unique.empty()) return;

  root_candidates_[var.get()] = unique;
  const bool ambiguous = unresolved || unique.size() > 1;
  if (ambiguous) {
    ambiguous_buffer_vars.insert(var.get());
    if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
      buffer_roots[var.get()] = unique.front();
    } else {
      buffer_roots.erase(var.get());
    }
  } else {
    ambiguous_buffer_vars.erase(var.get());
    buffer_roots[var.get()] = unique.front();
  }
}

std::vector<BufferRootCollector::OutputRoot> BufferRootCollector::CollectCallOutputRoots(
    const CallPtr& call) const {
  auto callee = program_->GetFunction(call->op_->name_);
  if (!callee) return {};

  std::vector<OutputRoot> roots;
  for (size_t i = 0; i < callee->param_directions_.size() && i < call->args_.size(); ++i) {
    if (callee->param_directions_[i] != ParamDirection::Out &&
        callee->param_directions_[i] != ParamDirection::InOut) {
      continue;
    }
    roots.push_back(OutputRoot{RootInfo{ResolveRootCandidates(call->args_[i]), IsAmbiguous(call->args_[i])},
                               call->args_[i]->GetType()});
  }
  return roots;
}

BufferRootCollector::RootInfo BufferRootCollector::SelectReturnRootInfo(
    const std::vector<OutputRoot>& out_roots, const TypePtr& return_type) const {
  if (out_roots.empty()) return RootInfo{{}, false};
  if (out_roots.size() == 1) return out_roots[0].info;

  const OutputRoot* match = nullptr;
  bool ambiguous = false;
  for (const auto& candidate : out_roots) {
    if (!candidate.info.roots.empty() && TypesMatchShapeDtype(candidate.type, return_type)) {
      if (match == nullptr) {
        match = &candidate;
      } else if (match->info.roots != candidate.info.roots ||
                 match->info.ambiguous != candidate.info.ambiguous) {
        ambiguous = true;
      }
    }
  }
  if (match && !ambiguous) return match->info;
  // No provable unambiguous type match (0 matches, or >1 distinct candidates).
  // The fallback depends on what the consumer needs when the owning buffer
  // can't be pinned down:
  //   kSkip        — record no root. Fusion / aliasing is an optimization, so
  //                  skipping it (no root -> no aliasing) is always safe,
  //                  whereas guessing could re-alias a scratch onto the output.
  //   kFirstOutput — fall back to the first Out/InOut root, matching the naive
  //                  pre-dedup behavior. DeriveCallDirections needs *some* root
  //                  so a later write to the returned var still promotes to
  //                  InOut; a null root would silently drop the WAW/InOut dep.
  if (ambiguous_policy_ == AmbiguousRootPolicy::kFirstOutput) {
    return out_roots[0].info;
  }
  return RootInfo{{}, false};
}

bool BufferRootCollector::TypesMatchShapeDtype(const TypePtr& a, const TypePtr& b) {
  auto ta = As<TensorType>(a);
  auto tb = As<TensorType>(b);
  if (!ta || !tb) return false;
  if (ta->dtype_ != tb->dtype_) return false;
  if (ta->shape_.size() != tb->shape_.size()) return false;
  for (size_t i = 0; i < ta->shape_.size(); ++i) {
    if (!AreExprsEqual(ta->shape_[i], tb->shape_[i])) return false;
  }
  return true;
}

BufferRootCollector::StorageId BufferRootCollector::StorageAliasGroup(StorageId storage) {
  return FindStorage(group_parents_, storage);
}

bool BufferRootCollector::IsAmbiguousStorage(StorageId storage) {
  return ambiguous_[FindStorage(parents_, storage)];
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::CallArgumentStorage(const CallPtr& call,
                                                                                       size_t index) {
  const auto found = calls_.find(call.get());
  if (found == calls_.end() || index >= found->second.args.size()) return std::nullopt;
  return StorageRoot(found->second.args[index]);
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::CallResultStorage(const CallPtr& call) {
  const auto found = calls_.find(call.get());
  return found == calls_.end() ? std::nullopt : StorageRoot(found->second.result);
}

void BufferRootCollector::VisitExpr_(const CallPtr& call) {
  if (track_tile_storage_) {
    static_cast<void>(StorageValue(call));
  } else {
    IRVisitor::VisitExpr_(call);
  }
}

void BufferRootCollector::VisitExpr_(const IterArgPtr& arg) {
  if (!track_tile_storage_) IRVisitor::VisitExpr_(arg);
}

void BufferRootCollector::VisitStmt_(const YieldStmtPtr& yield) {
  if (!track_tile_storage_) {
    IRVisitor::VisitStmt_(yield);
    return;
  }
  for (size_t i = 0; i < yield->value_.size(); ++i) {
    const auto storage = StorageValue(yield->value_[i]);
    if (yields_.empty()) continue;
    auto& values = yields_.back();
    if (values.size() <= i) values.resize(i + 1);
    values[i] = StorageChoice(values[i], storage);
  }
}

bool BufferRootCollector::MayContainTile(const TypePtr& type) {
  const auto cached = tile_types_.find(type.get());
  if (cached != tile_types_.end()) return cached->second;
  bool contains = As<TileType>(type) != nullptr;
  if (const auto tuple = As<TupleType>(type)) {
    for (const auto& element : tuple->types_) contains |= MayContainTile(element);
  }
  tile_types_.emplace(type.get(), contains);
  return contains;
}

void BufferRootCollector::MarkUnresolvedTileTuple(const TypePtr& type) {
  if (MayContainTile(type) && unresolved_tuple_type_set_.insert(type.get()).second) {
    unresolved_tuple_types_.push_back(type);
  }
}

BufferRootCollector::TupleStoragePtr BufferRootCollector::UnknownTileTuple(const TypePtr& type) {
  MarkUnresolvedTileTuple(type);
  return std::make_shared<TupleStorage>(TupleStorage{{}, std::nullopt, MayContainTile(type)});
}

BufferRootCollector::TupleStoragePtr BufferRootCollector::FreshTileTuple(const TypePtr& type) {
  TupleStorage value;
  if (As<TileType>(type)) {
    value.storage = FreshStorage();
    value.may_contain_tile = MayContainTile(type);
  } else if (const auto tuple = As<TupleType>(type)) {
    for (const auto& element : tuple->types_) {
      auto child = FreshTileTuple(element);
      value.may_contain_tile |= child->may_contain_tile;
      value.elements.push_back(std::move(child));
    }
  }
  return std::make_shared<TupleStorage>(std::move(value));
}

void BufferRootCollector::BindTileTuple(const VarPtr& var, const TupleStoragePtr& tuple) {
  const auto old = tuple_variables_.find(var.get());
  if (!yields_.empty() && old != tuple_variables_.end() && old->second != tuple &&
      (old->second->may_contain_tile || tuple->may_contain_tile)) {
    // Joining different tuple trees would require a separate tuple CFG
    // analysis. Leave the rejection policy to the consumer.
    MarkUnresolvedTileTuple(var->GetType());
  }
  tuple_variables_[var.get()] = tuple;
}

BufferRootCollector::TupleStoragePtr BufferRootCollector::ProjectTileTuple(
    const std::shared_ptr<const TupleGetItemExpr>& projection) {
  const auto tuple = TileTuple(projection->tuple_);
  if (projection->index_ < 0 || static_cast<size_t>(projection->index_) >= tuple->elements.size()) {
    return UnknownTileTuple(projection->GetType());
  }
  return tuple->elements[static_cast<size_t>(projection->index_)];
}

BufferRootCollector::TupleStoragePtr BufferRootCollector::TileTuple(const ExprPtr& expr) {
  if (const auto var = AsVarLike(expr)) {
    const auto found = tuple_variables_.find(var.get());
    if (found != tuple_variables_.end()) return found->second;
    auto tuple = UnknownTileTuple(var->GetType());
    tuple_variables_.emplace(var.get(), tuple);
    return tuple;
  }
  if (const auto projection = As<TupleGetItemExpr>(expr)) return ProjectTileTuple(projection);
  if (const auto tuple = As<MakeTuple>(expr)) {
    TupleStorage result;
    for (const auto& element : tuple->elements_) {
      TupleStoragePtr provenance;
      if (As<TupleType>(element->GetType())) {
        provenance = TileTuple(element);
      } else {
        const auto storage = StorageValue(element);
        provenance =
            std::make_shared<TupleStorage>(TupleStorage{{}, storage, MayContainTile(element->GetType())});
      }
      result.may_contain_tile |= provenance->may_contain_tile;
      result.elements.push_back(std::move(provenance));
    }
    return std::make_shared<TupleStorage>(std::move(result));
  }
  if (const auto call = As<Call>(expr)) {
    static_cast<void>(StorageValue(call));
    const auto* entry = LookupOpEntry(call->op_);
    // Registered value-returning multi-output ops own their outputs; the
    // registry forbids them from declaring a reused destination argument.
    // Quantization/gather tuples therefore have independent tile storage,
    // rather than making all provenance in the function unresolved.
    if (entry && entry->GetOutputArity() > 1 &&
        !op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
      const auto found = call_tuples_.find(call.get());
      if (found != call_tuples_.end()) return found->second;
      auto result = FreshTileTuple(call->GetType());
      call_tuples_.emplace(call.get(), result);
      return result;
    }
  } else {
    VisitExpr(expr);
  }
  return UnknownTileTuple(expr->GetType());
}

BufferRootCollector::StorageId BufferRootCollector::FreshStorage() {
  const auto id = parents_.size();
  parents_.push_back(id);
  ranks_.push_back(0);
  group_parents_.push_back(id);
  group_ranks_.push_back(0);
  ambiguous_.push_back(false);
  return id;
}

BufferRootCollector::StorageId BufferRootCollector::FindStorage(std::vector<StorageId>& parents,
                                                                StorageId id) {
  while (parents[id] != id) {
    parents[id] = parents[parents[id]];
    id = parents[id];
  }
  return id;
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::StorageRoot(
    std::optional<StorageId> storage) {
  return storage ? std::make_optional(FindStorage(parents_, *storage)) : std::nullopt;
}

BufferRootCollector::StorageId BufferRootCollector::UniteStorage(std::vector<StorageId>& parents,
                                                                 std::vector<size_t>& ranks, StorageId lhs,
                                                                 StorageId rhs) {
  lhs = FindStorage(parents, lhs);
  rhs = FindStorage(parents, rhs);
  if (lhs == rhs) return lhs;
  if (ranks[lhs] < ranks[rhs]) std::swap(lhs, rhs);
  parents[rhs] = lhs;
  if (ranks[lhs] == ranks[rhs]) ++ranks[lhs];
  return lhs;
}

void BufferRootCollector::ConnectStorageChoices(StorageId lhs, StorageId rhs) {
  static_cast<void>(UniteStorage(group_parents_, group_ranks_, lhs, rhs));
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::StorageChoice(
    std::optional<StorageId> lhs, std::optional<StorageId> rhs) {
  lhs = StorageRoot(lhs);
  rhs = StorageRoot(rhs);
  if (!lhs) return rhs;
  if (!rhs || lhs == rhs) return lhs;
  const auto choice = FreshStorage();
  ambiguous_[choice] = true;
  ConnectStorageChoices(choice, *lhs);
  ConnectStorageChoices(choice, *rhs);
  return choice;
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::MergeStorage(
    std::optional<StorageId> lhs, std::optional<StorageId> rhs) {
  lhs = StorageRoot(lhs);
  rhs = StorageRoot(rhs);
  if (!lhs) return rhs;
  if (!rhs || lhs == rhs) return lhs;
  // Loop carries can name the yielded storage on a later iteration. Update
  // earlier call operands through the identity forest before storage queries.
  ConnectStorageChoices(*lhs, *rhs);
  const bool ambiguous = ambiguous_[*lhs] || ambiguous_[*rhs];
  const auto root = UniteStorage(parents_, ranks_, *lhs, *rhs);
  ambiguous_[root] = ambiguous;
  return root;
}

void BufferRootCollector::BindStorage(const VarPtr& var, std::optional<StorageId> storage) {
  if (!var || !As<TileType>(var->GetType()) || !storage) return;
  const auto old = variables_.find(var.get());
  // A rebind changes this name's candidates without changing earlier aliases
  // of either source. Loop init/yield identities are joined separately.
  if (!yields_.empty() && old != variables_.end()) storage = StorageChoice(old->second, storage);
  if (storage.has_value()) {
    variables_[var.get()] = *storage;
  }
}

std::optional<BufferRootCollector::StorageId> BufferRootCollector::StorageValue(const ExprPtr& expr) {
  if (!expr) return std::nullopt;
  if (auto var = AsVarLike(expr)) {
    if (As<TupleType>(var->GetType())) {
      static_cast<void>(TileTuple(var));
      return std::nullopt;
    }
    if (!As<TileType>(var->GetType())) return std::nullopt;
    const auto found = variables_.find(var.get());
    if (found != variables_.end()) return found->second;
    const auto storage = FreshStorage();
    variables_.emplace(var.get(), storage);
    return storage;
  }
  if (auto projection = As<TupleGetItemExpr>(expr)) return ProjectTileTuple(projection)->storage;
  if (As<MakeTuple>(expr)) {
    static_cast<void>(TileTuple(expr));
    return std::nullopt;
  }
  if (auto call = As<Call>(expr)) {
    CallStorage storage;
    for (const auto& arg : call->args_) storage.args.push_back(StorageValue(arg));
    if (As<TileType>(call->GetType())) {
      if (const auto reused = ResultAliasedArgIndex(call)) {
        storage.result = storage.args[*reused];
      } else if (call->op_ && op_predicates::IsBufferAliasingViewOp(call->op_->name_) &&
                 !storage.args.empty()) {
        storage.result = storage.args[0];
      }
      if (!storage.result) storage.result = FreshStorage();
    }
    const auto [position, inserted] = calls_.emplace(call.get(), storage);
    if (!inserted) {
      // A shared expression can occur more than once in hand-built IR.
      // Conservatively retain every operand/storage interpretation.
      for (size_t i = 0; i < storage.args.size(); ++i) {
        position->second.args[i] = StorageChoice(position->second.args[i], storage.args[i]);
      }
      position->second.result = StorageChoice(position->second.result, storage.result);
    }
    return position->second.result;
  }
  VisitExpr(expr);
  return std::nullopt;
}

std::vector<std::optional<BufferRootCollector::StorageId>> BufferRootCollector::StorageRegion(
    const StmtPtr& body) {
  yields_.emplace_back();
  if (body) VisitStmt(body);
  auto values = std::move(yields_.back());
  yields_.pop_back();
  return values;
}

void BufferRootCollector::StorageLoop(const std::vector<IterArgPtr>& arguments,
                                      const std::vector<VarPtr>& results, const StmtPtr& body) {
  std::vector<std::optional<StorageId>> initial;
  for (const auto& argument : arguments) {
    if (argument && As<TupleType>(argument->GetType())) {
      BindTileTuple(argument, TileTuple(argument->initValue_));
      MarkUnresolvedTileTuple(argument->GetType());
      initial.emplace_back(std::nullopt);
      continue;
    }
    initial.push_back(argument ? StorageValue(argument->initValue_) : std::nullopt);
    BindStorage(argument, initial.back());
  }
  const auto yielded = StorageRegion(body);
  for (size_t i = 0; i < arguments.size(); ++i) {
    auto storage = i < yielded.size() ? MergeStorage(initial[i], yielded[i]) : initial[i];
    if (i < results.size()) BindStorage(results[i], storage);
  }
}

}  // namespace buffer_root
}  // namespace ir
}  // namespace pypto
