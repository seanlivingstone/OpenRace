/* Copyright 2021 Coderrect Inc. All Rights Reserved.
Licensed under the GNU Affero General Public License, version 3 or later (“AGPL”), as published by the Free Software
Foundation. You may not use this file except in compliance with the License. You may obtain a copy of the License at
https://www.gnu.org/licenses/agpl-3.0.en.html
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//
// Created by peiming on 10/30/19.
//
#ifndef PTA_CTXFUNCTION_H
#define PTA_CTXFUNCTION_H

#include <llvm/Support/CommandLine.h>

#include "PointerAnalysis/Program/CallSite.h"

namespace pta {

template <typename ctx>
class CallGraphNode;
template <typename ctx>
class CtxModule;

// wrapper for llvm::Function, but it has context attached to it
template <typename ctx>
class CtxFunction {
 private:
  const ctx *context;
  bool isExt = false;
  const llvm::Function *function;
  CallGraphNode<ctx> *const callNode;  // the corresponding call node

  // FIXME: remove callSite, there is a set of callsite for a function because
  // the context might be merged!!!
  const llvm::Instruction *callSite;

 public:
  CtxFunction(const ctx *C, const llvm::Function *F, const llvm::Instruction *I, CallGraphNode<ctx> *N)
      : context(C), function(F), callSite(I), callNode(N) {}
  CtxFunction(const CtxFunction<ctx> &&cf) noexcept
      : context(cf.context), function(cf.function), callNode(cf.callNode) {}

  CtxFunction(const CtxFunction<ctx> &) = delete;
  CtxFunction<ctx> &operator=(const CtxFunction<ctx> &) = delete;

  [[nodiscard]] inline const ctx *getContext() const { return this->context; }

  inline void markAsExtFunction() { this->isExt = true; }

  [[nodiscard]] inline const llvm::Function *getFunction() const { return this->function; }

  [[deprecated("The Callsite returned could be wrong")]] inline const llvm::Instruction *getCallSite() const {
    return this->callSite;
  }

  [[nodiscard]] inline CallGraphNode<ctx> *getCallNode() const { return this->callNode; }
  [[nodiscard]] inline bool isExtFunction() const { return this->isExt; }

  [[nodiscard]] inline auto begin() const -> decltype(function->begin()) { return function->begin(); }

  [[nodiscard]] inline auto end() const -> decltype(function->end()) { return function->end(); }

  [[nodiscard]] inline auto arg_begin() const -> decltype(function->arg_begin()) { return function->arg_begin(); }

  [[nodiscard]] inline auto arg_end() const -> decltype(function->arg_end()) { return function->arg_end(); }

  [[nodiscard]] inline llvm::StringRef getName() const { return function->getName(); }

  friend CtxModule<ctx>;
};

// a indirect function call invoked in ctx
template <typename ctx>
class InDirectCallSite {
 private:
  const ctx *context;
  const pta::CallSite callSite;
  const llvm::Value *const funPtr;

  std::set<const llvm::Function *> targets;           // possible targets
  std::set<const CallGraphNode<ctx> *> resolvedNode;  // the corresponding call node

  CallGraphNode<ctx> *const callNode;  // the corresponding indirect call node

 public:
  InDirectCallSite(const ctx *C, const llvm::Instruction *CS, const llvm::Value *V, CallGraphNode<ctx> *N)
      : context(C), callSite(CS), funPtr(V), callNode(N) {
    assert(callSite.isCallOrInvoke());
  }

  InDirectCallSite(const ctx *C, const llvm::Instruction *CS, CallGraphNode<ctx> *N)
      : context(C), callSite(CS), funPtr(nullptr), callNode(N) {
    assert(callSite.isCallOrInvoke());
  }

  InDirectCallSite(const InDirectCallSite<ctx> &&cf) noexcept
      : context(cf.context), callSite(cf.callSite), funPtr(cf.funPtr), callNode(cf.callNode) {}

  InDirectCallSite(const InDirectCallSite<ctx> &) = delete;
  InDirectCallSite<ctx> &operator=(const InDirectCallSite<ctx> &) = delete;

  [[nodiscard]] inline CallGraphNode<ctx> *getCallNode() const { return this->callNode; }
  [[nodiscard]] inline bool isInterceptedCallSite() const {
    return this->funPtr != nullptr;  // the callsite is changed by language model
  }

  [[nodiscard]] inline const ctx *getContext() const { return this->context; }

  [[nodiscard]] inline const llvm::Value *getValue() const {
    if (funPtr != nullptr) {
      // overriden by language model.
      return funPtr;
    }
    return callSite.getCalledValue();
  }

  [[nodiscard]] inline const llvm::Type *getCalledType() const { return this->getValue()->getType(); }

  [[nodiscard]] inline const llvm::Instruction *getCallSite() const { return this->callSite.getInstruction(); }

  [[nodiscard]] inline bool resolvedTo(const llvm::Function *fun, bool applyLimit, size_t maxIndirectTarget = 999) {
    if (applyLimit && this->targets.size() >= maxIndirectTarget) {
      return false;
    }
    return this->targets.insert(fun).second;
  }

  inline void resolvedToNode(const CallGraphNode<ctx> *node) {
    auto result = this->resolvedNode.insert(node).second;
    assert(result);
  }

  [[nodiscard]] inline const std::set<const CallGraphNode<ctx> *> &getResolvedNode() const {
    return this->resolvedNode;
  }

  [[nodiscard]] inline const std::set<const llvm::Function *> &getResolvedTarget() const { return this->targets; }

  friend CtxModule<ctx>;
};

}  // namespace pta

#endif