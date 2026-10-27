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

#include "pypto/ir/transforms/utils/lifetime_analysis.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/memory_reuse/internal.h"

namespace pypto {
namespace ir {
namespace {

/**
 * @brief Collect all Var nodes referenced in an expression.
 */
class VarUseCollector : public IRVisitor {
 public:
  std::set<VarPtr> used_vars;

  void VisitVarLike_(const VarPtr& var) override {
    used_vars.insert(var);
    IRVisitor::VisitVarLike_(var);
  }

  // A use of a bound carry reads its current value, not its initializer.
  // The completed source graph below supplies the appropriate phi edges.
  void VisitExpr_(const IterArgPtr& argument) override { VisitVarLike_(argument); }

  void VisitExpr_(const CallPtr& call) override {
    const auto* entry = LookupOpEntry(call->op_);
    for (size_t i = 0; i < call->args_.size(); ++i) {
      auto variable = AsVarLike(call->args_[i]);
      if (variable && As<TileType>(variable->GetType()) && entry &&
          (entry->GetExecutionMemoryAccessEvidence() == ExecutionMemoryAccessEvidence::NoAccess ||
           (entry->GetIRStage() == OpIRStage::Functional &&
            !ArgEffectReads(entry->GetArgEffect(i, call->kwargs_))))) {
        continue;
      }
      VisitExpr(call->args_[i]);
    }
    for (const auto& [key, value] : call->attrs_) {
      (void)key;
      ForEachAttrExpr(value, [this](const ExprPtr& expression) { VisitExpr(expression); });
    }
  }
};

/**
 * @brief Full IR tree walker for lifetime analysis.
 *
 * Walks the entire IR tree (including nested control flow bodies) to
 * assign sequential order to all leaf statements and collect variable
 * definitions and uses.
 *
 * Three-phase design:
 *   Phase 1: Walk tree, collect raw uses and complete the phi/view source graph.
 *   Phase 2: Propagate maximum uses, including cycles through native loop phis.
 *   Phase 3 (ComputeEffectiveLastUse): For each var, extend lifetime to the end of any
 *           enclosing loop where the var is defined before the loop.
 */
class LifetimeAnalyzer : public IRVisitor {
 public:
  struct Result {
    std::vector<VarPtr> ordered_defs;        // TileType vars in definition order
    std::map<VarPtr, int> var_def_order;     // var -> def order
    std::map<VarPtr, int> var_last_use;      // var -> effective last use (with loop extension)
    std::map<VarPtr, StmtPtr> var_def_stmt;  // var -> defining AssignStmt (for op name extraction)
    // Phi families, keyed per (scf.if, return-slot): each family id groups one
    // phi return_var with its branch-local yield-sources (the same logical value
    // across mutually-exclusive paths), so they may always share a buffer.  A
    // var may belong to several families (e.g. a source yielded at >1 slot)
    // without those families merging, so we store a *set* of ids per var.
    std::map<const Var*, std::set<int>> phi_family_ids;
  };

  Result Analyze(const StmtPtr& func_body, const std::vector<VarPtr>& entry_vars = {}) {
    // Function parameters have no defining AssignStmt. Seed Tile parameters at
    // the entry point so function-wide allocation planning includes their
    // MemRefs and body uses extend their lifetimes normally.
    for (const VarPtr& var : entry_vars) {
      const auto tile_type = As<TileType>(var->GetType());
      if (!tile_type || !tile_type->memref_.has_value()) continue;
      ordered_defs_.push_back(var);
      var_def_order_[var] = current_order_;
    }

    // Phase 1: Walk IR tree
    if (func_body) {
      VisitStmt(func_body);
    }

    // Loop backedges can name sources not visited when an IterArg is read.
    // Finish every edge before propagating those reads.
    PropagateUses();

    // Apply loop-aware lifetime extension.
    auto effective_last_use = ComputeEffectiveLastUse();

    return {std::move(ordered_defs_), std::move(var_def_order_), std::move(effective_last_use),
            std::move(var_def_stmt_), std::move(var_family_ids_)};
  }

 protected:
  // Container statements: recurse into children
  void VisitStmt_(const SeqStmtsPtr& op) override {
    for (const auto& stmt : op->stmts_) {
      VisitStmt(stmt);
    }
  }

  void VisitStmt_(const InCoreScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const ClusterScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const HierarchyScopeStmtPtr& op) override { VisitStmt(op->body_); }
  void VisitStmt_(const SpmdScopeStmtPtr& op) override { VisitStmt(op->body_); }

  // Leaf statements: assign order + collect defs/uses
  void VisitStmt_(const AssignStmtPtr& op) override {
    int order = current_order_++;

    // Record TileType variable definitions
    auto tile_type = As<TileType>(op->var_->GetType());
    if (tile_type) {
      ordered_defs_.push_back(op->var_);
      var_def_order_[op->var_] = order;
      var_def_stmt_[op->var_] = op;
      if (tile_type->memref_) {
        aliases_by_base_[(*tile_type->memref_)->base_.get()].emplace_back(order, op->var_);
      }
      // Propagate demanded payloads through physical aliases as well as native
      // handles. Otherwise a branch-local alias could gain phi exclusivity
      // while hiding its independently live source outside the branch.
      auto source = AsVarLike(op->value_);
      if (auto call = As<Call>(op->value_);
          call && op_predicates::IsBufferAliasingViewOp(call->op_->name_) && !call->args_.empty()) {
        source = AsVarLike(call->args_[0]);
      }
      if (source) phi_sources_[op->var_.get()] = {{source, -1}};
    }

    // A bare Tile rename only binds a handle. Its sharing group or source edge
    // forwards later payload observations without turning the rename into one.
    if (!tile_type || !AsVarLike(op->value_)) CollectUsesFromExpr(op->value_, order);
  }

  void VisitStmt_(const EvalStmtPtr& op) override {
    int order = current_order_++;
    CollectUsesFromExpr(op->expr_, order);
  }

  void VisitStmt_(const YieldStmtPtr& op) override {
    const int order = current_order_++;
    for (size_t i = 0; i < op->value_.size(); ++i) {
      if (!region_results_.empty() && i < region_results_.back()->size()) {
        const auto& result = (*region_results_.back())[i];
        auto type = As<TileType>(result->GetType());
        // Native results bind storage. Canonical results retain their payload
        // reads because yield repair can copy into their fixed destination.
        if (type && !type->memref_ && AsVarLike(op->value_[i])) continue;
      }
      CollectUsesFromExpr(op->value_[i], order);
    }
  }

  void VisitStmt_(const ReturnStmtPtr& op) override { CollectUsesFromValues(op->value_); }

  // Control flow: recurse into bodies with scope tracking
  void VisitStmt_(const IfStmtPtr& op) override {
    // Visit condition uses at the current order point (before branches)
    CollectUsesFromExpr(op->condition_, current_order_);

    // Visit then body first, else body second (sequential ordering), recording
    // each branch's order range so we can tell branch-local yield-sources apart.
    region_results_.push_back(&op->return_vars_);
    int then_start = current_order_;
    VisitStmt(op->then_body_);
    int then_end = current_order_ - 1;
    int else_start = -1;
    int else_end = -1;
    if (op->else_body_.has_value()) {
      else_start = current_order_;
      VisitStmt(*op->else_body_);
      else_end = current_order_ - 1;
    }
    region_results_.pop_back();

    // Canonical results protect their destination and branch-local sources
    // (#1821). Native handle results instead retain every incoming allocation.
    // Only branch-local definitions join the mutually exclusive family: an
    // outside seed may still be observed independently after the join.
    //
    // Extending both branch sources to the phi's last read makes the two
    // mutually-exclusive sources overlap, so we also tag them with a fresh
    // per-slot family id; the reuse packer exempts intra-family pairs from the
    // interference check.  Ids are *per return slot* (not a transitive union): a
    // source shared across slots joins several families without merging them, so
    // two distinct phi results are never exempted.
    auto then_yield = transform_utils::GetLastYieldStmt(op->then_body_);
    auto else_yield =
        op->else_body_.has_value() ? transform_utils::GetLastYieldStmt(*op->else_body_) : nullptr;
    for (size_t i = 0; i < op->return_vars_.size(); ++i) {
      const auto& rv = op->return_vars_[i];
      auto tile_type = As<TileType>(rv->GetType());
      if (!tile_type) continue;
      int family_id = next_family_id_++;
      std::vector<VarPtr> local_sources;
      std::vector<VarPtr> incoming;
      std::set<const Var*> seen_phi;
      std::set<std::tuple<const Var*, int, int>> expanded_bases;
      std::set<const Var*> seen_sources;
      // Flatten a nested-if phi source down to its terminal tracked tiles, and
      // tag THOSE with this (outer) slot's family id -- otherwise the terminals
      // only carry the inner slot's id and `overlap_blocks_sharing` would reject safe nested
      // mutually-exclusive sharing.
      std::function<void(const VarPtr&, int, int)> add_source_var = [&](const VarPtr& src, int start,
                                                                        int end) {
        auto pit = phi_sources_.find(src.get());
        // An incoming loop carry can name an earlier execution of a producer;
        // it is not a branch-local alternative from this execution of the If.
        if (pit != phi_sources_.end() && !As<IterArg>(src)) {
          if (!seen_phi.insert(src.get()).second) return;
          for (const auto& [nested, binding_order] : pit->second) {
            (void)binding_order;
            add_source_var(nested, start, end);
          }
          return;
        }
        auto tile = GetTileTypeWithMemRef(src->GetType());
        if (!tile || start < 0) return;
        // A loop return or final writeback name is only one alias of this
        // allocation. Tag all branch-local definitions, never an outside seed.
        const auto* base = GetDefinedMemRef(tile)->base_.get();
        // Nested phis can forward several aliases of one allocation. Expand
        // each branch range once and retain one edge per source, so later phi
        // reads do not revisit duplicate aliases. Families remain per slot.
        if (!expanded_bases.emplace(base, start, end).second) return;
        const auto& aliases = aliases_by_base_[base];
        auto first = std::lower_bound(aliases.begin(), aliases.end(), start,
                                      [](const auto& item, int point) { return item.first < point; });
        for (; first != aliases.end() && first->first <= end; ++first) {
          if (!seen_sources.insert(first->second.get()).second) continue;
          local_sources.push_back(first->second);
          var_family_ids_[first->second.get()].insert(family_id);
        }
      };
      auto add_source = [&](const ExprPtr& val, int start, int end) {
        if (auto src = AsVarLike(val)) {
          incoming.push_back(src);
          add_source_var(src, start, end);
        }
      };
      if (then_yield && i < then_yield->value_.size()) {
        add_source(then_yield->value_[i], then_start, then_end);
      }
      if (else_yield && i < else_yield->value_.size()) {
        add_source(else_yield->value_[i], else_start, else_end);
      }
      if (!local_sources.empty()) {
        var_family_ids_[rv.get()].insert(family_id);
        auto& sources = phi_sources_[rv.get()];
        for (const auto& source : local_sources) sources.emplace_back(source, -1);
      }
      if (!tile_type->memref_) {
        // A native handle phi has no canonical destination: all incoming
        // allocations, including values defined before the If, remain possible.
        // Keep direct edges; use propagation memoizes the completed graph.
        auto& sources = phi_sources_[rv.get()];
        sources.clear();
        for (const auto& source : incoming) sources.emplace_back(source, -1);
      }
    }
    RecordRegionResults(op->return_vars_, op);
  }

  void VisitStmt_(const ForStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->return_vars_, op->body_);
    RegisterLoopSources(op);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
    RecordRegionResults(op->return_vars_, op);
  }

  void VisitStmt_(const WhileStmtPtr& op) override {
    ProcessLoopBody(op->iter_args_, op->return_vars_, op->body_, op->condition_);
    RegisterLoopSources(op);
    RegisterReturnVars(op->iter_args_, op->return_vars_);
    RecordRegionResults(op->return_vars_, op);
  }

 private:
  void RecordRegionResults(const std::vector<VarPtr>& results, const StmtPtr& statement) {
    for (const auto& value : results) {
      auto tile = GetTileTypeWithMemRef(value->GetType());
      if (!tile) continue;
      const int point = current_order_++;
      ordered_defs_.push_back(value);
      var_def_order_[value] = point;
      var_def_stmt_[value] = statement;
      aliases_by_base_[GetDefinedMemRef(tile)->base_.get()].emplace_back(point, value);
    }
  }
  std::map<const Var*, std::vector<std::pair<int, VarPtr>>> aliases_by_base_;
  int current_order_ = 0;

  // Loop boundaries recorded during Phase 1
  struct LoopBoundary {
    int start_order;
    int end_order;
  };
  std::vector<LoopBoundary> loop_scopes_;

  // Variable tracking (Phase 1)
  std::vector<VarPtr> ordered_defs_;
  std::map<VarPtr, int> var_def_order_;
  std::map<VarPtr, int> var_raw_last_use_;  // Raw last use (without loop extension)
  std::map<const Var*, int> propagated_use_order_;
  std::map<VarPtr, StmtPtr> var_def_stmt_;

  // Canonical For/While results retain their initializer's fixed destination.
  // Native results instead use the init/yield source edges below.
  std::map<VarPtr, VarPtr> return_var_to_init_var_;

  // Per-slot phi families: var -> set of family ids (see VisitStmt_(IfStmtPtr)).
  std::map<const Var*, std::set<int>> var_family_ids_;
  int next_family_id_ = 0;

  // Phi/view -> (source, minimum use order). Ordinary forwarding has no lower
  // bound (-1). An observed backedge must retain its source through the yield,
  // even when the next iteration's payload read is lexically before that yield.
  // Canonical If phis only forward branch-local sources; native phis can also
  // retain outside allocations.
  std::map<const Var*, std::vector<std::pair<VarPtr, int>>> phi_sources_;
  std::vector<const std::vector<VarPtr>*> region_results_;

  void CollectUsesFromExpr(const ExprPtr& expr, int use_order) {
    if (!expr) return;
    VarUseCollector collector;
    collector.VisitExpr(expr);
    for (const auto& var : collector.used_vars) {
      RecordRawUse(var, use_order);
    }
  }

  void CollectUsesFromValues(const std::vector<ExprPtr>& values) {
    int order = current_order_++;
    for (const auto& val : values) {
      CollectUsesFromExpr(val, order);
    }
  }

  void ProcessLoopBody(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& results,
                       const StmtPtr& body, const ExprPtr& condition = nullptr) {
    for (const auto& iter_arg : iter_args) {
      if (iter_arg->initValue_) {
        CollectUsesFromExpr(iter_arg->initValue_, current_order_);
      }
      // Logical carries need use intervals for producer-version planning, but
      // own no allocation and therefore stay out of ordered_defs_.
      if (As<TileType>(iter_arg->GetType())) var_def_order_[iter_arg] = current_order_;
    }
    int loop_start = current_order_;
    if (condition) CollectUsesFromExpr(condition, current_order_++);
    region_results_.push_back(&results);
    VisitStmt(body);
    region_results_.pop_back();
    int loop_end = current_order_ - 1;
    loop_scopes_.push_back({loop_start, loop_end});
  }

  template <typename LoopPtr>
  void RegisterLoopSources(const LoopPtr& loop) {
    const auto yield = transform_utils::GetLastYieldStmt(loop->body_);
    for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      auto tile = As<TileType>(argument->GetType());
      if (!tile) continue;
      const auto initial = AsVarLike(argument->initValue_);
      const auto backedge = yield && i < yield->value_.size() ? AsVarLike(yield->value_[i]) : nullptr;
      auto& sources = phi_sources_[argument.get()];
      if (initial) sources.emplace_back(initial, -1);
      if (!tile->memref_ && backedge && memory_reuse::LoopMayExecute(loop, true)) {
        sources.emplace_back(backedge, current_order_ - 1);
      }
      if (i >= loop->return_vars_.size()) continue;
      const auto& result = loop->return_vars_[i];
      auto result_tile = As<TileType>(result->GetType());
      if (!result_tile || result_tile->memref_) continue;
      // Include the initializer for a possible zero-trip exit. Native results
      // select a source rather than aliasing one canonical initializer buffer.
      auto& exit_sources = phi_sources_[result.get()];
      if (initial) exit_sources.emplace_back(initial, -1);
      if (backedge && memory_reuse::LoopMayExecute(loop, false)) exit_sources.emplace_back(backedge, -1);
    }
  }

  /// Resolve a loop carry's init to the variable that actually owns the buffer.
  ///
  /// A nested loop seeds its carry from the *enclosing* loop's IterArg, and a
  /// canonical IterArg is an alias of its own init rather than an owning
  /// definition; iter-args are absent from `ordered_defs_`.
  /// Walking the carry chain to the first non-IterArg therefore lands on the
  /// AssignStmt-defined tile whose buffer the whole loop nest shares, which is
  /// the variable a use of any return_var in that nest has to keep alive.
  ///
  /// Returns nullptr when the chain does not end at a tracked carrier; the
  /// caller then records no mapping, exactly as before.
  [[nodiscard]] VarPtr ResolveCarryInitCarrier(const ExprPtr& init_value) const {
    auto var = AsVarLike(init_value);
    // An init is always defined before its own loop, so the chain is acyclic in
    // well-formed IR; the visited set bounds it anyway rather than hanging.
    std::set<const Var*> seen;
    while (var) {
      auto iter_arg = As<IterArg>(var);
      if (!iter_arg) return var;
      if (!seen.insert(var.get()).second) return nullptr;
      var = AsVarLike(iter_arg->initValue_);
    }
    return nullptr;
  }

  void RegisterReturnVars(const std::vector<IterArgPtr>& iter_args, const std::vector<VarPtr>& return_vars) {
    size_t count = std::min(return_vars.size(), iter_args.size());
    for (size_t i = 0; i < count; ++i) {
      const auto& rv = return_vars[i];
      auto tile = As<TileType>(rv->GetType());
      if (!tile || !tile->memref_) continue;

      // Map return_var -> the init's buffer carrier so that post-loop uses of the
      // return_var also extend that carrier's lifetime. RecordRegionResults
      // separately protects the result's canonical post-loop destination.
      auto init_var = ResolveCarryInitCarrier(iter_args[i]->initValue_);
      if (init_var && var_def_order_.count(init_var)) {
        return_var_to_init_var_[rv] = init_var;
      }
    }
  }

  void RecordRawUse(const VarPtr& var, int use_order) {
    auto [use, inserted] = var_raw_last_use_.try_emplace(var, use_order);
    if (!inserted) use->second = std::max(use->second, use_order);
  }

  [[nodiscard]] bool IsCanonicalIfResult(const VarPtr& var) const {
    auto definition = var_def_stmt_.find(var);
    return GetTileTypeWithMemRef(var->GetType()) && definition != var_def_stmt_.end() &&
           As<IfStmt>(definition->second);
  }

  void PropagateUses() {
    // Activate demanded bindings once before propagating orders. In particular,
    // unused native yields never become payload reads. Seeding an activated
    // backedge's binding order here keeps the subsequent maximum-use traversal
    // descending, even though a backedge can raise an earlier lexical read.
    std::vector<VarPtr> demanded;
    for (const auto& [var, order] : var_raw_last_use_) {
      (void)order;
      demanded.push_back(var);
    }
    while (!demanded.empty()) {
      auto var = demanded.back();
      demanded.pop_back();
      if (!propagated_use_order_.try_emplace(var.get(), -1).second) continue;
      // Canonical arm copies already read their incoming aliases at yield.
      // A later result read protects the destination, not the old payload of
      // an outside source reached through a branch-local alias.
      if (IsCanonicalIfResult(var)) continue;
      if (auto sources = phi_sources_.find(var.get()); sources != phi_sources_.end()) {
        for (const auto& [source, binding_order] : sources->second) {
          if (binding_order >= 0) RecordRawUse(source, binding_order);
          demanded.push_back(source);
        }
      }
    }
    propagated_use_order_.clear();

    // A descending maximum-use worklist visits each node and source edge once:
    // a node's first pop already has the largest use that can reach it. This
    // handles both deep phi DAGs and native-loop cycles in O((V + E) log V),
    // without revisiting an unfinished forward edge or recursively growing the
    // C++ stack. Raw last-use and the propagation memo remain distinct because
    // canonical loop destinations only extend their own initializer buffer.
    std::priority_queue<std::pair<int, VarPtr>> pending;
    for (const auto& [var, order] : var_raw_last_use_) pending.emplace(order, var);
    while (!pending.empty()) {
      const auto [order, var] = pending.top();
      pending.pop();
      auto [seen, inserted] = propagated_use_order_.try_emplace(var.get(), order);
      if (!inserted && seen->second >= order) continue;
      seen->second = order;
      RecordRawUse(var, order);
      // Canonical copies consume incoming payloads at arm yield; the result's
      // own allocation interval protects its destination after the join.
      if (IsCanonicalIfResult(var)) continue;
      if (auto sources = phi_sources_.find(var.get()); sources != phi_sources_.end()) {
        for (const auto& [source, binding_order] : sources->second) {
          (void)binding_order;
          pending.emplace(order, source);
        }
      } else if (auto initial = return_var_to_init_var_.find(var);
                 initial != return_var_to_init_var_.end() && var_def_order_.count(initial->second)) {
        RecordRawUse(initial->second, order);
      }
    }
  }

  /**
   * @brief Compute effective last use with loop-aware extension.
   *
   * For each variable, if it's used inside a loop body and defined before
   * that loop, extend its lifetime to the end of the loop. This handles
   * the case where the loop re-executes and the variable is needed again.
   */
  [[nodiscard]] std::map<VarPtr, int> ComputeEffectiveLastUse() const {
    std::map<VarPtr, int> effective;

    for (const auto& [var, def_order] : var_def_order_) {
      int raw_last = var_raw_last_use_.count(var) ? var_raw_last_use_.at(var) : def_order;
      int extended = raw_last;

      // Check all loop scopes: if var is defined before the loop and
      // has any use inside the loop [start, end], extend to loop end.
      for (const auto& loop : loop_scopes_) {
        if (def_order < loop.start_order && raw_last >= loop.start_order) {
          // Variable is defined before loop and used inside it
          extended = std::max(extended, loop.end_order);
        }
      }

      effective[var] = extended;
    }

    return effective;
  }
};

/**
 * @brief Compute lifetime intervals by walking the full IR tree.
 *
 * Walks ALL statements
 * including those inside nested control flow (IfStmt/ForStmt/WhileStmt bodies).
 */
LifetimeAnalysisResult AnalyzeAllocationLifetimesImpl(const StmtPtr& func_body,
                                                      const std::vector<VarPtr>& entry_vars = {}) {
  std::vector<LifetimeInterval> lifetimes;

  // Step 1: Walk full IR tree to collect variable defs, uses, and ordering
  LifetimeAnalyzer analyzer;
  auto result = analyzer.Analyze(func_body, entry_vars);

  if (result.ordered_defs.empty()) {
    return {lifetimes, {}};
  }

  // Step 2: Build MemRef sharing groups (keyed by base_ Ptr identity)
  std::map<const Var*, std::vector<VarPtr>> memref_groups;
  for (const auto& var : result.ordered_defs) {
    auto tile_type = As<TileType>(var->GetType());
    if (tile_type && tile_type->memref_.has_value()) {
      const Var* base_ptr = tile_type->memref_.value()->base_.get();
      memref_groups[base_ptr].push_back(var);
    }
  }

  std::map<VarPtr, std::vector<VarPtr>> var_sharing_groups;
  for (const auto& [base_ptr, vars] : memref_groups) {
    if (vars.size() > 1) {
      for (const auto& var : vars) {
        var_sharing_groups[var] = vars;
      }
      LOG_DEBUG << "MemRef sharing group: " << vars.size() << " variables share same MemRef";
    }
  }

  // Step 3: Compute lifetime intervals (with MemRef sharing group merging)
  std::set<VarPtr> processed_vars;

  // Pipeline-stage membership per interval representative (parsed once from the
  // defining Call's pipeline_membership attr, set by LowerPipelineLoops), plus
  // the subset whose defining op is a load (vs compute).
  std::map<const Var*, std::vector<std::pair<int32_t, int32_t>>> pipeline_membership;
  std::set<const Var*> pipeline_load_tiles;
  std::set<const Var*> subrange_unsafe_groups;

  for (const auto& var : result.ordered_defs) {
    if (processed_vars.count(var)) {
      continue;
    }

    auto tile_type = As<TileType>(var->GetType());
    if (!tile_type || !tile_type->memref_.has_value()) {
      continue;
    }

    std::vector<VarPtr> sharing_group;
    if (var_sharing_groups.count(var)) {
      sharing_group = var_sharing_groups[var];
    } else {
      sharing_group = {var};
    }

    // Compute MERGED lifetime for all variables in the sharing group
    int min_def_point = INT_MAX;
    int max_last_use = INT_MIN;

    for (const auto& group_var : sharing_group) {
      int def_point = result.var_def_order.count(group_var) ? result.var_def_order.at(group_var) : 0;
      int last_use = result.var_last_use.count(group_var) ? result.var_last_use.at(group_var) : def_point;

      LOG_DEBUG << "Variable " << group_var->name_hint_ << " def=" << def_point << " last_use=" << last_use;

      min_def_point = std::min(min_def_point, def_point);
      max_last_use = std::max(max_last_use, last_use);
    }

    LifetimeInterval interval;
    interval.variable = sharing_group[0];
    interval.def_point = min_def_point;
    interval.last_use_point = max_last_use;
    auto representative_tile_type = As<TileType>(sharing_group[0]->GetType());
    INTERNAL_CHECK_SPAN(representative_tile_type != nullptr, sharing_group[0]->span_)
        << "Expected TileType for reuse interval";
    auto memory_space = representative_tile_type->GetMemorySpace();
    INTERNAL_CHECK_SPAN(memory_space.has_value(), sharing_group[0]->span_)
        << "TileType with MemRef must have memory_space for reuse analysis";
    interval.memory_space = *memory_space;
    interval.size = GetDefinedMemRef(representative_tile_type)->size_;

    // A direct dynamic tile.slice is lowered through pto.subview, which rebuilds
    // its runtime address from the source tile and slice operands. Every other
    // dynamic-offset member is unsafe for capacity-triggered subdivision:
    // reshape/reinterpret/transpose views have no offset operand, and an unknown
    // producer must fail closed. Mark the complete sharing group so the packer
    // can keep its whole memory space on the legacy layout. Restricting only
    // this interval is insufficient — subdividing other roots can make the
    // space fit and let this pre-existing dynamic address loss reach codegen.
    for (const auto& group_var : sharing_group) {
      auto member_memref = GetTypeMemRef(group_var->GetType());
      if (!member_memref.has_value() || !*member_memref || As<ConstInt>((*member_memref)->byte_offset_)) {
        continue;
      }
      auto dit = result.var_def_stmt.find(group_var);
      auto assign = dit != result.var_def_stmt.end() ? As<AssignStmt>(dit->second) : nullptr;
      auto call = assign ? As<Call>(assign->value_) : nullptr;
      if (!call || !IsOp(call, "tile.slice")) {
        subrange_unsafe_groups.insert(interval.variable.get());
        break;
      }
    }

    lifetimes.push_back(interval);

    // Record pipeline-stage membership for this interval's representative so the
    // reuse packer can forbid cross-stage coalescing. Every sharing-group member
    // is a tile of one pipeline clone and carries the same membership; take the
    // first non-empty one (views with no Call def simply contribute nothing).
    for (const auto& group_var : sharing_group) {
      auto dit = result.var_def_stmt.find(group_var);
      if (dit == result.var_def_stmt.end()) continue;
      auto assign = As<AssignStmt>(dit->second);
      if (!assign) continue;
      auto call = As<Call>(assign->value_);
      if (!call) continue;
      auto packed = call->GetAttr<std::string>(kPipelineMembershipAttr, std::string());
      if (!packed.empty()) {
        // Parse the membership string once here; the packer compares pre-parsed
        // vectors so it never re-parses in its O(N²) pairwise loop.
        pipeline_membership[interval.variable.get()] = ParsePipelineMembership(packed);
        if (IsOp(call, "tile.load") || IsOp(call, "tile.read")) {
          pipeline_load_tiles.insert(interval.variable.get());
        }
        break;
      }
    }

    for (const auto& group_var : sharing_group) {
      processed_vars.insert(group_var);
    }

    LOG_DEBUG << "Lifetime for sharing group (representative: " << sharing_group[0]->name_hint_
              << ", size: " << sharing_group.size() << "): [" << min_def_point << ", " << max_last_use << "]"
              << " space=" << static_cast<int>(interval.memory_space) << " size=" << interval.size;
  }

  // Per-var individual liveness (def + effective last use, including phi/loop
  // extension) for the packer's precise pairwise interference check.
  std::map<const Var*, std::pair<int, int>> var_liveness;
  for (const auto& [var, def_point] : result.var_def_order) {
    int last_use = result.var_last_use.count(var) ? result.var_last_use.at(var) : def_point;
    var_liveness[var.get()] = {def_point, last_use};
  }

  return {lifetimes,
          var_sharing_groups,
          std::move(result.phi_family_ids),
          std::move(var_liveness),
          std::move(pipeline_membership),
          std::move(pipeline_load_tiles),
          std::move(subrange_unsafe_groups)};
}

}  // namespace

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const StmtPtr& func_body) {
  return AnalyzeAllocationLifetimesImpl(func_body);
}

LifetimeAnalysisResult AnalyzeAllocationLifetimes(const FunctionPtr& func) {
  INTERNAL_CHECK(func != nullptr) << "Cannot analyze allocation lifetimes for a null function";
  return AnalyzeAllocationLifetimesImpl(func->body_, func->params_);
}

}  // namespace ir
}  // namespace pypto
