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

/**
 * @file verify_acc_store_phase.cpp
 * @brief Verify the final accumulator-producer / final-store unit-flag protocol.
 *
 * On A2/A3, ``acc_phase=pl.AccPhase.Final`` lowers to a producer-side
 * check-and-set. The matching ``st_phase=pl.STPhase.Final`` store performs the
 * consumer-side check-and-clear. Omitting either side is not a benign metadata
 * mismatch: the device can wait forever on an unset flag, or leave a set flag
 * behind and stall a later accumulator producer.
 *
 * A function-level count is insufficient. Two unrelated producers and stores
 * may have equal counts while clearing the wrong value, and a store that exists
 * only on one branch does not post-dominate a producer before that branch. This
 * verifier therefore tracks the exact SSA value created by each final producer.
 * A plain SSA alias may carry the obligation, but a view/copy/call may not: the
 * verifier cannot prove that such an operation preserves the hardware tile and
 * must reject rather than silently accept a possible deadlock.
 *
 * Control-flow policy is deliberately conservative. A final producer and its
 * final store must be in the same straight-line region. A balanced pair inside
 * a branch or loop body is legal; an obligation may not cross an if/loop/scope
 * boundary. This makes zero-iteration loops, one-sided branches, and later
 * outlining safe without a path-exponential analysis.
 *
 * ``AccStorePhaseValid`` runs immediately after InlineFunctions, so a final
 * producer returned by an Inline helper is checked in the same function and
 * region as its consuming store. The visitor handles both the resulting
 * pre-SSA form (including rebindings and an immediately nested
 * ``store(gemv(..., final), ..., final)``) and the normalized three-address form
 * seen by later verification instrumentation.
 * Logical storage provenance is indexed once before token checking. Overall
 * complexity is O(N log N), where N is the IR size.
 */

#include <cstddef>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/phase.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/type.h"
#include "pypto/ir/verifier/verifier.h"

namespace pypto {
namespace ir {

namespace {

using TokenId = size_t;
using StorageId = size_t;

/// Logical may-alias provenance, independent of Final's exact-value token.
/// An in-place Final updates its input storage, including aliases made before
/// the Final call. Record call operands at their lexical definition so a fresh
/// pre-SSA rebind does not accidentally change older aliases. Registered reuse
/// and view contracts are the authority; physical allocation/slot overlap is
/// deliberately left to the memory planner.
///
/// Complete this index before checking tokens: loop init/yield/return values
/// may alias on a later iteration even when their initial values are distinct.
/// Loop-carried storage identities and possible branch sources use separate
/// union-by-rank forests. A choice connects its candidates without identifying
/// the candidates with each other; concrete Acc buffers stay independently
/// usable. Both forests have O(N log N) total work and O(N) space.
class AccumulatorStorageLineage : public IRVisitor {
 public:
  void Build(const StmtPtr& body) { VisitStmt(body); }
  bool HasUnresolvedTupleProvenance() const { return unresolved_tuple_; }

  StorageId AliasGroup(StorageId storage) { return Find(group_parents_, storage); }
  bool IsAmbiguous(StorageId storage) { return ambiguous_[Find(parents_, storage)]; }

  std::optional<StorageId> Argument(const CallPtr& call, size_t index) {
    const auto found = calls_.find(call.get());
    if (found == calls_.end() || index >= found->second.args.size()) return std::nullopt;
    return Root(found->second.args[index]);
  }

  std::optional<StorageId> Result(const CallPtr& call) {
    const auto found = calls_.find(call.get());
    return found == calls_.end() ? std::nullopt : Root(found->second.result);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const IterArgPtr&) override {}
  void VisitExpr_(const CallPtr& call) override { static_cast<void>(Value(call)); }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (As<TupleType>(assign->var_->GetType())) {
      BindTuple(assign->var_, Tuple(assign->value_));
    } else {
      Bind(assign->var_, Value(assign->value_));
    }
  }

  void VisitStmt_(const IfStmtPtr& statement) override {
    static_cast<void>(Value(statement->condition_));
    const auto then_values = Region(statement->then_body_);
    const auto else_values =
        statement->else_body_ ? Region(*statement->else_body_) : std::vector<std::optional<StorageId>>{};
    for (size_t i = 0; i < statement->return_vars_.size(); ++i) {
      if (As<TupleType>(statement->return_vars_[i]->GetType())) {
        MarkUnresolvedTuple(statement->return_vars_[i]->GetType());
        continue;
      }
      auto storage = i < then_values.size() ? then_values[i] : std::nullopt;
      if (i < else_values.size()) storage = Choice(storage, else_values[i]);
      if (!statement->else_body_) storage = Choice(storage, Value(statement->return_vars_[i]));
      Bind(statement->return_vars_[i], storage);
    }
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    static_cast<void>(Value(loop->start_));
    static_cast<void>(Value(loop->stop_));
    static_cast<void>(Value(loop->step_));
    Loop(loop->iter_args_, loop->return_vars_, loop->body_);
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    static_cast<void>(Value(loop->condition_));
    Loop(loop->iter_args_, loop->return_vars_, loop->body_);
  }

  void VisitStmt_(const YieldStmtPtr& statement) override {
    for (size_t i = 0; i < statement->value_.size(); ++i) {
      const auto storage = Value(statement->value_[i]);
      if (yields_.empty()) continue;
      auto& values = yields_.back();
      if (values.size() <= i) values.resize(i + 1);
      values[i] = Choice(values[i], storage);
    }
  }

 private:
  struct CallStorage {
    std::vector<std::optional<StorageId>> args;
    std::optional<StorageId> result;
  };

  /// Persistent tuple provenance: aliases/projections share children instead
  /// of copying a nested tuple tree. Each explicit tuple contributes only its
  /// own edges, keeping repeated forwarding and nested projections linear.
  struct TupleStorage {
    std::vector<std::shared_ptr<const TupleStorage>> elements;
    std::optional<StorageId> storage;
    bool may_contain_accumulator = false;
  };
  using TupleStoragePtr = std::shared_ptr<const TupleStorage>;

  bool MayContainAccumulator(const TypePtr& type) {
    const auto cached = accumulator_types_.find(type.get());
    if (cached != accumulator_types_.end()) return cached->second;
    const auto tile = As<TileType>(type);
    // Known Mat/Vec/operand tiles cannot alias an Acc buffer. An unresolved
    // memory space remains conservative until memory inference establishes it.
    bool contains = tile && (!tile->GetMemorySpace() || tile->GetMemorySpace() == MemorySpace::Acc);
    if (const auto tuple = As<TupleType>(type)) {
      for (const auto& element : tuple->types_) contains |= MayContainAccumulator(element);
    }
    accumulator_types_.emplace(type.get(), contains);
    return contains;
  }

  void MarkUnresolvedTuple(const TypePtr& type) { unresolved_tuple_ |= MayContainAccumulator(type); }

  TupleStoragePtr UnknownTuple(const TypePtr& type) {
    MarkUnresolvedTuple(type);
    return std::make_shared<TupleStorage>(TupleStorage{{}, std::nullopt, MayContainAccumulator(type)});
  }

  TupleStoragePtr FreshTuple(const TypePtr& type) {
    TupleStorage value;
    if (As<TileType>(type)) {
      value.storage = Fresh();
      value.may_contain_accumulator = MayContainAccumulator(type);
    } else if (const auto tuple = As<TupleType>(type)) {
      for (const auto& element : tuple->types_) {
        auto child = FreshTuple(element);
        value.may_contain_accumulator |= child->may_contain_accumulator;
        value.elements.push_back(std::move(child));
      }
    }
    return std::make_shared<TupleStorage>(std::move(value));
  }

  void BindTuple(const VarPtr& var, const TupleStoragePtr& tuple) {
    const auto old = tuple_variables_.find(var.get());
    if (!yields_.empty() && old != tuple_variables_.end() && old->second != tuple &&
        (old->second->may_contain_accumulator || tuple->may_contain_accumulator)) {
      // Joining different tuple trees would require a separate tuple CFG
      // analysis. Reject this case only if a Final protocol is actually used.
      unresolved_tuple_ = true;
    }
    tuple_variables_[var.get()] = tuple;
  }

  TupleStoragePtr Project(const std::shared_ptr<const TupleGetItemExpr>& projection) {
    const auto tuple = Tuple(projection->tuple_);
    if (projection->index_ < 0 || static_cast<size_t>(projection->index_) >= tuple->elements.size()) {
      return UnknownTuple(projection->GetType());
    }
    return tuple->elements[static_cast<size_t>(projection->index_)];
  }

  TupleStoragePtr Tuple(const ExprPtr& expr) {
    if (const auto var = AsVarLike(expr)) {
      const auto found = tuple_variables_.find(var.get());
      if (found != tuple_variables_.end()) return found->second;
      auto tuple = UnknownTuple(var->GetType());
      tuple_variables_.emplace(var.get(), tuple);
      return tuple;
    }
    if (const auto projection = As<TupleGetItemExpr>(expr)) return Project(projection);
    if (const auto tuple = As<MakeTuple>(expr)) {
      TupleStorage result;
      for (const auto& element : tuple->elements_) {
        TupleStoragePtr provenance;
        if (As<TupleType>(element->GetType())) {
          provenance = Tuple(element);
        } else {
          const auto storage = Value(element);
          provenance = std::make_shared<TupleStorage>(
              TupleStorage{{}, storage, MayContainAccumulator(element->GetType())});
        }
        result.may_contain_accumulator |= provenance->may_contain_accumulator;
        result.elements.push_back(std::move(provenance));
      }
      return std::make_shared<TupleStorage>(std::move(result));
    }
    if (const auto call = As<Call>(expr)) {
      static_cast<void>(Value(call));
      const auto* entry = LookupOpEntry(call->op_);
      // Registered value-returning multi-output ops own their outputs; the
      // registry forbids them from declaring a reused destination argument.
      // Quantization/gather tuples therefore do not become an unrelated
      // restriction on a Final accumulator elsewhere in the same function.
      if (entry && entry->GetOutputArity() > 1 &&
          !op_predicates::OutputInheritsSourceBuffer(call->op_->name_)) {
        const auto found = call_tuples_.find(call.get());
        if (found != call_tuples_.end()) return found->second;
        auto result = FreshTuple(call->GetType());
        call_tuples_.emplace(call.get(), result);
        return result;
      }
    } else {
      VisitExpr(expr);
    }
    return UnknownTuple(expr->GetType());
  }

  StorageId Fresh() {
    const auto id = parents_.size();
    parents_.push_back(id);
    ranks_.push_back(0);
    group_parents_.push_back(id);
    group_ranks_.push_back(0);
    ambiguous_.push_back(false);
    return id;
  }

  static StorageId Find(std::vector<StorageId>& parents, StorageId id) {
    while (parents[id] != id) {
      parents[id] = parents[parents[id]];
      id = parents[id];
    }
    return id;
  }

  std::optional<StorageId> Root(std::optional<StorageId> storage) {
    return storage ? std::make_optional(Find(parents_, *storage)) : std::nullopt;
  }

  static StorageId Unite(std::vector<StorageId>& parents, std::vector<size_t>& ranks, StorageId lhs,
                         StorageId rhs) {
    lhs = Find(parents, lhs);
    rhs = Find(parents, rhs);
    if (lhs == rhs) return lhs;
    if (ranks[lhs] < ranks[rhs]) std::swap(lhs, rhs);
    parents[rhs] = lhs;
    if (ranks[lhs] == ranks[rhs]) ++ranks[lhs];
    return lhs;
  }

  void ConnectChoices(StorageId lhs, StorageId rhs) {
    static_cast<void>(Unite(group_parents_, group_ranks_, lhs, rhs));
  }

  std::optional<StorageId> Choice(std::optional<StorageId> lhs, std::optional<StorageId> rhs) {
    lhs = Root(lhs);
    rhs = Root(rhs);
    if (!lhs) return rhs;
    if (!rhs || lhs == rhs) return lhs;
    const auto choice = Fresh();
    ambiguous_[choice] = true;
    ConnectChoices(choice, *lhs);
    ConnectChoices(choice, *rhs);
    return choice;
  }

  std::optional<StorageId> Merge(std::optional<StorageId> lhs, std::optional<StorageId> rhs) {
    lhs = Root(lhs);
    rhs = Root(rhs);
    if (!lhs) return rhs;
    if (!rhs || lhs == rhs) return lhs;
    // Loop carries can name the yielded storage on a later iteration. Update
    // earlier call operands through the identity forest before token checking.
    ConnectChoices(*lhs, *rhs);
    const bool ambiguous = ambiguous_[*lhs] || ambiguous_[*rhs];
    const auto root = Unite(parents_, ranks_, *lhs, *rhs);
    ambiguous_[root] = ambiguous;
    return root;
  }

  void Bind(const VarPtr& var, std::optional<StorageId> storage) {
    if (!var || !As<TileType>(var->GetType()) || !storage) return;
    const auto old = variables_.find(var.get());
    // A rebind changes this name's candidates without changing earlier aliases
    // of either source. Loop init/yield identities are joined separately.
    if (!yields_.empty() && old != variables_.end()) storage = Choice(old->second, storage);
    if (storage.has_value()) {
      variables_[var.get()] = *storage;
    }
  }

  std::optional<StorageId> Value(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    if (auto var = AsVarLike(expr)) {
      if (As<TupleType>(var->GetType())) {
        static_cast<void>(Tuple(var));
        return std::nullopt;
      }
      if (!As<TileType>(var->GetType())) return std::nullopt;
      const auto found = variables_.find(var.get());
      if (found != variables_.end()) return found->second;
      const auto storage = Fresh();
      variables_.emplace(var.get(), storage);
      return storage;
    }
    if (auto projection = As<TupleGetItemExpr>(expr)) return Project(projection)->storage;
    if (As<MakeTuple>(expr)) {
      static_cast<void>(Tuple(expr));
      return std::nullopt;
    }
    if (auto call = As<Call>(expr)) {
      CallStorage storage;
      for (const auto& arg : call->args_) storage.args.push_back(Value(arg));
      if (As<TileType>(call->GetType())) {
        const auto reused = op_predicates::BuiltinWritebackArgIndex(call->op_, call->args_.size());
        if (reused) {
          storage.result = storage.args[*reused];
        } else if (call->op_ && op_predicates::IsBufferAliasingViewOp(call->op_->name_)) {
          for (const auto& argument : storage.args) {
            if (argument) {
              storage.result = argument;
              break;
            }
          }
        }
        if (!storage.result) storage.result = Fresh();
      }
      const auto [position, inserted] = calls_.emplace(call.get(), storage);
      if (!inserted) {
        // A shared expression can occur more than once in hand-built IR.
        // Conservatively retain every operand/storage interpretation.
        for (size_t i = 0; i < storage.args.size(); ++i) {
          position->second.args[i] = Choice(position->second.args[i], storage.args[i]);
        }
        position->second.result = Choice(position->second.result, storage.result);
      }
      return position->second.result;
    }
    VisitExpr(expr);
    return std::nullopt;
  }

  std::vector<std::optional<StorageId>> Region(const StmtPtr& body) {
    yields_.emplace_back();
    if (body) VisitStmt(body);
    auto values = std::move(yields_.back());
    yields_.pop_back();
    return values;
  }

  void Loop(const std::vector<IterArgPtr>& arguments, const std::vector<VarPtr>& results,
            const StmtPtr& body) {
    std::vector<std::optional<StorageId>> initial;
    for (const auto& argument : arguments) {
      if (argument && As<TupleType>(argument->GetType())) {
        BindTuple(argument, Tuple(argument->initValue_));
        MarkUnresolvedTuple(argument->GetType());
        initial.emplace_back(std::nullopt);
        continue;
      }
      initial.push_back(argument ? Value(argument->initValue_) : std::nullopt);
      Bind(argument, initial.back());
    }
    const auto yielded = Region(body);
    for (size_t i = 0; i < arguments.size(); ++i) {
      auto storage = i < yielded.size() ? Merge(initial[i], yielded[i]) : initial[i];
      if (i < results.size()) Bind(results[i], storage);
    }
  }

  std::vector<StorageId> parents_;
  std::vector<size_t> ranks_;
  std::vector<StorageId> group_parents_;
  std::vector<size_t> group_ranks_;
  std::vector<bool> ambiguous_;
  std::unordered_map<const Var*, StorageId> variables_;
  std::unordered_map<const Call*, CallStorage> calls_;
  std::vector<std::vector<std::optional<StorageId>>> yields_;
  std::unordered_map<const Var*, TupleStoragePtr> tuple_variables_;
  std::unordered_map<const Call*, TupleStoragePtr> call_tuples_;
  std::unordered_map<const Type*, bool> accumulator_types_;
  bool unresolved_tuple_ = false;
};

struct FinalProducer {
  std::string op_name;
  Span span;
  std::optional<StorageId> storage;
};

bool IsPhasedAccumulatorProducer(const CallPtr& call) {
  return IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.gemv") ||
         IsOp(call, "tile.gemv_acc") || IsOp(call, "tile.gemv_bias");
}

bool IsFinalAccumulatorProducer(const CallPtr& call) {
  return IsPhasedAccumulatorProducer(call) &&
         call->GetKwarg<int>("acc_phase", static_cast<int>(AccPhase::kUnspecified)) ==
             static_cast<int>(AccPhase::kFinal);
}

/**
 * @brief Straight-line typestate checker for one function.
 *
 * ``pending_`` contains producer obligations that have been set but not yet
 * cleared. ``var_tokens_`` carries an obligation through a bare SSA alias.
 * Consuming a token removes it from ``pending_`` but deliberately leaves the
 * Var mapping in place: a second final store of the same value can then be
 * diagnosed as a double-consume instead of as an unrelated store.
 */
class AccStorePhaseVisitor : public IRVisitor {
 public:
  AccStorePhaseVisitor(std::vector<Diagnostic>& diagnostics, std::string func_name)
      : diagnostics_(diagnostics), func_name_(std::move(func_name)) {}

  void Run(const StmtPtr& body) {
    body_ = body;
    if (body) VisitStmt(body);
    FlushPending("before leaving function '" + func_name_ + "'");
  }

 protected:
  void VisitExpr_(const CallPtr& op) override { static_cast<void>(ProcessCall(op)); }

  void VisitStmt_(const AssignStmtPtr& op) override {
    if (!op || !op->var_) return;
    auto token = ProcessValue(op->value_);
    if (token.has_value()) {
      var_tokens_[op->var_.get()] = *token;
    } else {
      // The pre-SSA form may reuse one Var object for a later assignment. Do
      // not let an old final-producer obligation leak through that rebind.
      var_tokens_.erase(op->var_.get());
    }
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    if (op && op->expr_) static_cast<void>(ProcessValue(op->expr_));
  }

  void VisitStmt_(const SeqStmtsPtr& op) override {
    if (!op) return;
    // SeqStmts is a transparent container, so nested sequences do not create a
    // control-flow boundary and an obligation may cross them.
    for (const auto& stmt : op->stmts_) {
      if (stmt) VisitStmt(stmt);
    }
  }

  void VisitStmt_(const IfStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->condition_);
    BreakRegion("before entering an if statement");
    AnalyzeIsolatedRegion(op->then_body_, "before leaving an if branch");
    if (op->else_body_.has_value()) {
      AnalyzeIsolatedRegion(*op->else_body_, "before leaving an else branch");
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->start_);
    ScanExpr(op->stop_);
    ScanExpr(op->step_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) ScanExpr(iter_arg->initValue_);
    }
    BreakRegion("before entering a loop");
    AnalyzeIsolatedRegion(op->body_, "before completing a loop iteration");
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) var_tokens_.erase(iter_arg.get());
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    if (!op) return;
    ScanExpr(op->condition_);
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) ScanExpr(iter_arg->initValue_);
    }
    BreakRegion("before entering a while loop");
    AnalyzeIsolatedRegion(op->body_, "before completing a while-loop iteration");
    for (const auto& iter_arg : op->iter_args_) {
      if (iter_arg) var_tokens_.erase(iter_arg.get());
    }
    for (const auto& var : op->return_vars_) {
      if (var) var_tokens_.erase(var.get());
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override {
    if (op) ScanExpr(op->core_num_);
    AnalyzeScope(op);
  }
  void VisitStmt_(const SplitAivScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const RuntimeScopeStmtPtr& op) override { AnalyzeScope(op); }
  void VisitStmt_(const CommDomainScopeStmtPtr& op) override { AnalyzeScope(op); }

  void VisitStmt_(const YieldStmtPtr& op) override {
    if (op) {
      for (const auto& value : op->value_) ScanExpr(value);
    }
    FlushPending("before yielding from a control-flow region");
  }

  void VisitStmt_(const ReturnStmtPtr& op) override {
    if (op) {
      for (const auto& value : op->value_) ScanExpr(value);
    }
    FlushPending("before returning from function '" + func_name_ + "'");
  }

  void VisitStmt_(const BreakStmtPtr& /*op*/) override { FlushPending("before breaking out of a loop"); }

  void VisitStmt_(const ContinueStmtPtr& /*op*/) override { FlushPending("before continuing a loop"); }

 private:
  std::optional<TokenId> ProcessValue(const ExprPtr& expr) {
    if (!expr) return std::nullopt;
    if (auto var = AsVarLike(expr)) {
      auto it = var_tokens_.find(var.get());
      return it == var_tokens_.end() ? std::nullopt : std::make_optional(it->second);
    }
    if (auto call = As<Call>(expr)) return ProcessCall(call);
    // Any final producer buried in another expression is still recorded and
    // will be reported as unmatched. Its containing expression is not a plain
    // alias, so it cannot carry the token to a store.
    VisitExpr(expr);
    return std::nullopt;
  }

  std::optional<TokenId> ProcessCall(const CallPtr& call) {
    if (!call) return std::nullopt;

    if (IsOp(call, "tile.store")) {
      std::optional<TokenId> source_token;
      if (!call->args_.empty()) source_token = ProcessValue(call->args_[0]);
      for (size_t i = 1; i < call->args_.size(); ++i) ScanExpr(call->args_[i]);
      // A pre-existing alias of an in-place Final has no exact-value token,
      // but cannot be read by a plain store before the matching Final drains.
      // A final store of that alias is already rejected by exact-value checks.
      if (!source_token && call->GetKwarg<int>("st_phase", static_cast<int>(STPhase::kUnspecified)) ==
                               static_cast<int>(STPhase::kUnspecified)) {
        CheckUndrainedStorage(call, storage_.Argument(call, 0));
      }
      CheckStore(call, source_token);
      return std::nullopt;
    }

    const bool is_final = IsFinalAccumulatorProducer(call);
    if (is_final && !storage_built_) {
      // Build the whole index once, including aliases before this producer and
      // later loop yields. Unphased functions never allocate this analysis.
      storage_.Build(body_);
      storage_built_ = true;
    }

    for (size_t i = 0; i < call->args_.size(); ++i) {
      static_cast<void>(ProcessValue(call->args_[i]));
      // Only a bare alias may carry an undrained Final value to its store.
      // Reject other uses conservatively: an accumulating MAD writes in place,
      // and a slice/view can hide that same storage from a later MAD. Checking
      // only the supported accumulating op names would miss that alias escape.
      CheckUndrainedStorage(call, storage_.Argument(call, i));
    }
    if (!is_final) return std::nullopt;

    if (storage_.HasUnresolvedTupleProvenance() && !tuple_provenance_reported_) {
      tuple_provenance_reported_ = true;
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/7,
          "Final accumulator protocol in function '" + func_name_ +
              "' has unresolved tile-bearing tuple provenance that may contain Acc storage across a "
              "control-flow join, loop carry, "
              "or opaque tuple value. Unpack the tiles into separate aliases or tile iter_args so "
              "the compiler can verify that every accumulator is drained before storage reuse.",
          call->span_);
    }

    const TokenId token = producers_.size();
    const auto storage = storage_.Result(call);
    producers_.push_back(FinalProducer{call->op_->name_, call->span_, storage});
    pending_.insert(token);
    if (storage) {
      auto& pending = pending_storage_[storage_.AliasGroup(*storage)];
      pending.all.insert(token);
      if (storage_.IsAmbiguous(*storage)) {
        pending.ambiguous.insert(token);
      } else {
        pending.concrete[*storage].insert(token);
      }
    }
    return token;
  }

  void ScanExpr(const ExprPtr& expr) {
    if (expr) static_cast<void>(ProcessValue(expr));
  }

  void CheckUndrainedStorage(const CallPtr& call, std::optional<StorageId> storage) {
    if (!storage) return;
    const auto pending = pending_storage_.find(storage_.AliasGroup(*storage));
    if (pending == pending_storage_.end()) return;
    const auto& aliases = pending->second;
    const auto concrete = aliases.concrete.find(*storage);
    const auto* tokens = &aliases.ambiguous;
    if (storage_.IsAmbiguous(*storage)) {
      tokens = &aliases.all;
    } else if (concrete != aliases.concrete.end()) {
      tokens = &concrete->second;
    }
    if (tokens->empty() || !unsafe_use_reported_.insert(*tokens->begin()).second) return;
    // Diagnose the first unsafe use; aliases of that use must not create a
    // cascade. Keep the obligation live so a later missing drain is still seen.
    diagnostics_.emplace_back(
        DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/6,
        "input to '" + call->op_->name_ + "' in function '" + func_name_ +
            "' has an undrained final accumulator producer. Store that exact value with "
            "st_phase=pl.STPhase.Final before using it in another operation. Only a bare alias "
            "may carry an undrained Final value; a view or in-place update can hide accumulator "
            "reuse before its unit flag has been cleared, including through an alias created "
            "before the Final operation.",
        call->span_);
  }

  bool ConsumePending(TokenId token) {
    if (pending_.erase(token) == 0) return false;
    if (const auto storage = producers_[token].storage) {
      const auto pending = pending_storage_.find(storage_.AliasGroup(*storage));
      INTERNAL_CHECK_SPAN(pending != pending_storage_.end(), producers_[token].span)
          << "Internal error: live Final token has no pending storage entry";
      auto& aliases = pending->second;
      aliases.all.erase(token);
      if (storage_.IsAmbiguous(*storage)) {
        aliases.ambiguous.erase(token);
      } else {
        auto concrete = aliases.concrete.find(*storage);
        INTERNAL_CHECK_SPAN(concrete != aliases.concrete.end(), producers_[token].span)
            << "Internal error: concrete Final token has no storage entry";
        concrete->second.erase(token);
        if (concrete->second.empty()) aliases.concrete.erase(concrete);
      }
      if (aliases.all.empty()) pending_storage_.erase(pending);
    }
    return true;
  }

  void CheckStore(const CallPtr& store, const std::optional<TokenId>& source_token) {
    const int st_phase = store->GetKwarg<int>("st_phase", static_cast<int>(STPhase::kUnspecified));
    if (!IsValidSTPhase(st_phase)) {
      if (source_token.has_value() && ConsumePending(*source_token)) {
        reported_.insert(*source_token);
      }
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/5,
          "tile.store st_phase in function '" + func_name_ +
              "' must encode pl.STPhase.Unspecified (0) or pl.STPhase.Final (3), but got " +
              std::to_string(st_phase) +
              ". A final accumulator producer must be consumed with pl.STPhase.Final so its unit flag is "
              "cleared.",
          store->span_);
      return;
    }
    if (st_phase == static_cast<int>(STPhase::kFinal)) {
      if (!source_token.has_value()) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/1,
            "tile.store(..., st_phase=pl.STPhase.Final) in function '" + func_name_ +
                "' must consume the exact SSA value produced by tile.matmul, tile.matmul_acc, "
                "tile.gemv, tile.gemv_acc, or tile.gemv_bias with acc_phase=pl.AccPhase.Final in the "
                "same straight-line region. A final "
                "store "
                "without a matching producer waits on a unit flag that was never set and can stall device "
                "execution.",
            store->span_);
        return;
      }
      if (!ConsumePending(*source_token)) {
        diagnostics_.emplace_back(
            DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/2,
            "tile.store(..., st_phase=pl.STPhase.Final) in function '" + func_name_ +
                "' does not have a live matching final accumulator producer in this straight-line region. "
                "The producer may already have been consumed, or the value crossed a control-flow boundary; "
                "either case makes the unit-flag clear ambiguous and can stall device execution.",
            store->span_);
      }
      return;
    }

    if (source_token.has_value() && ConsumePending(*source_token)) {
      const auto& producer = producers_[*source_token];
      reported_.insert(*source_token);
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/3,
          "final accumulator producer '" + producer.op_name + "' in function '" + func_name_ +
              "' is consumed by tile.store with st_phase=pl.STPhase." +
              STPhaseToString(static_cast<STPhase>(st_phase)) +
              ". Use st_phase=pl.STPhase.Final so the store performs check-and-clear; otherwise the "
              "producer's "
              "unit flag remains set and a later accumulator operation can stall.",
          store->span_);
    }
  }

  void AnalyzeScope(const ScopeStmtPtr& scope) {
    if (!scope) return;
    BreakRegion("before entering scope '" + scope->name_hint_ + "'");
    AnalyzeIsolatedRegion(scope->body_, "before leaving scope '" + scope->name_hint_ + "'");
  }

  void AnalyzeIsolatedRegion(const StmtPtr& body, const std::string& exit_description) {
    auto outer_pending = std::move(pending_);
    auto outer_storage = std::move(pending_storage_);
    pending_.clear();
    pending_storage_.clear();
    if (body) VisitStmt(body);
    FlushPending(exit_description);
    pending_ = std::move(outer_pending);
    pending_storage_ = std::move(outer_storage);
  }

  void BreakRegion(const std::string& description) { FlushPending(description); }

  void FlushPending(const std::string& description) {
    for (TokenId token : pending_) {
      if (!reported_.insert(token).second) continue;
      const auto& producer = producers_[token];
      diagnostics_.emplace_back(
          DiagnosticSeverity::Error, "AccStorePhaseValid", /*error_code=*/4,
          "final accumulator producer '" + producer.op_name + "' in function '" + func_name_ +
              "' must be consumed exactly once by tile.store(..., st_phase=pl.STPhase.Final) using that SSA "
              "value " +
              description +
              ". acc_phase=pl.AccPhase.Final performs check-and-set; leaving it uncleared can stall a later "
              "device "
              "operation.",
          producer.span);
    }
    pending_.clear();
    pending_storage_.clear();
  }

  std::vector<Diagnostic>& diagnostics_;
  std::string func_name_;
  std::vector<FinalProducer> producers_;
  std::set<TokenId> pending_;
  std::set<TokenId> reported_;
  std::unordered_map<const Var*, TokenId> var_tokens_;
  StmtPtr body_;
  bool storage_built_ = false;
  AccumulatorStorageLineage storage_;
  struct PendingStorage {
    // An ambiguous access checks the whole candidate group. A concrete access
    // checks its own storage plus ambiguous producers, preserving independence
    // of concrete Acc buffers that merely participate in the same selection.
    std::set<TokenId> all;
    std::set<TokenId> ambiguous;
    std::unordered_map<StorageId, std::set<TokenId>> concrete;
  };
  std::unordered_map<StorageId, PendingStorage> pending_storage_;
  std::set<TokenId> unsafe_use_reported_;
  bool tuple_provenance_reported_ = false;
};

}  // namespace

class AccStorePhaseValidPropertyVerifierImpl : public PropertyVerifier {
 public:
  [[nodiscard]] std::string GetName() const override { return "AccStorePhaseValid"; }

  void Verify(const ProgramPtr& program, std::vector<Diagnostic>& diagnostics) override {
    if (!program) return;
    for (const auto& [global_var, func] : program->functions_) {
      static_cast<void>(global_var);
      if (!func || !func->body_) continue;
      AccStorePhaseVisitor visitor(diagnostics, func->name_);
      visitor.Run(func->body_);
    }
  }
};

PropertyVerifierPtr CreateAccStorePhaseValidPropertyVerifier() {
  return std::make_shared<AccStorePhaseValidPropertyVerifierImpl>();
}

}  // namespace ir
}  // namespace pypto
