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

#pragma once

#include <llvm/IR/Instructions.h>

#include "IR/IR.h"

namespace race {

// ==================================================================
// ================== ReadIR Implementations ========================
// ==================================================================

class Load : public ReadIR {
  const llvm::LoadInst *inst;

 public:
  explicit Load(const llvm::LoadInst *load) : ReadIR(Type::Load), inst(load) {}

  [[nodiscard]] inline const llvm::LoadInst *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getPointerOperand(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::Load; }
};

class APIRead : public ReadIR {
  // Operand that this API call reads
  unsigned int operandOffset;

  const llvm::CallBase *inst;

 public:
  // API call that reads one of it's operands, specified by 'operandOffset'
  APIRead(const llvm::CallBase *inst, unsigned int operandOffset)
      : ReadIR(Type::APIRead), operandOffset(operandOffset), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getOperand(operandOffset); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::APIRead; }
};

// ==================================================================
// ================= WriteIR Implementations ========================
// ==================================================================

class Store : public WriteIR {
  const llvm::StoreInst *inst;

 public:
  explicit Store(const llvm::StoreInst *store) : WriteIR(Type::Store), inst(store) {}

  [[nodiscard]] inline const llvm::StoreInst *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override { return inst->getPointerOperand(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::Store; }
};

class APIWrite : public WriteIR {
  // Operand that this API call reads
  unsigned int operandOffset;

  const llvm::CallBase *inst;

 public:
  // API call that write to one of it's operands, specified by 'operandOffset'
  APIWrite(const llvm::CallBase *inst, unsigned int operandOffset)
      : WriteIR(Type::APIWrite), operandOffset(operandOffset), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] inline const llvm::Value *getAccessedValue() const override {
    return getInst()->getOperand(operandOffset);
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::APIWrite; }
};

// ==================================================================
// ================== ForkIR Implementations ========================
// ==================================================================

class PthreadCreate : public ForkIR {
  constexpr static unsigned int threadHandleOffset = 0;
  constexpr static unsigned int threadEntryOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit PthreadCreate(const llvm::CallBase *inst) : ForkIR(Type::PthreadCreate), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    return inst->getArgOperand(threadEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::PthreadCreate; }
};

class OpenMPFork : public ForkIR {
  // https://github.com/llvm/llvm-project/blob/ef32c611aa214dea855364efd7ba451ec5ec3f74/openmp/runtime/src/kmp_csupport.cpp#L262
  // @param loc  source location information
  // @param argc  total number of arguments in the ellipsis
  // @param microtask  pointer to callback routine consisting of outlined parallel
  // construct
  // @param ...  pointers to shared variables that aren't global
  constexpr static unsigned int threadHandleOffset = 0;
  constexpr static unsigned int threadEntryOffset = 2;
  const llvm::CallBase *inst;

 public:
  explicit OpenMPFork(const llvm::CallBase *inst) : ForkIR(Type::OpenMPFork), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  [[nodiscard]] const llvm::Value *getThreadEntry() const override {
    return inst->getArgOperand(threadEntryOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPFork; }
};

// ==================================================================
// ================== JoinIR Implementations ========================
// ==================================================================

class PthreadJoin : public JoinIR {
  const unsigned int threadHandleOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit PthreadJoin(const llvm::CallBase *inst) : JoinIR(Type::PthreadJoin), inst(inst) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override {
    return inst->getArgOperand(threadHandleOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::PthreadJoin; }
};

// This actually corresponds to a OpenMP fork instruction, as the fork call acts as both a fork and join in one call
class OpenMPJoin : public JoinIR {
  std::shared_ptr<OpenMPFork> fork;

 public:
  explicit OpenMPJoin(const std::shared_ptr<OpenMPFork> fork) : JoinIR(Type::OpenMPJoin), fork(fork) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return fork->getInst(); }

  [[nodiscard]] const llvm::Value *getThreadHandle() const override { return fork->getThreadHandle(); }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == Type::OpenMPJoin; }
};

// ==================================================================
// ================== LockIR Implementations ========================
// ==================================================================

// LockIRImpl should not be used directly. Instead define a using alias.
// See PthreadMutexLock below as an example.
template <IR::Type T>
class LockIRImpl : public LockIR {
  const unsigned int lockObjectOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit LockIRImpl(const llvm::CallBase *call) : LockIR(T), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(lockObjectOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

// NOTE: if a specific API semantic is the same as default impl,
// create a type alias.
using PthreadMutexLock = LockIRImpl<IR::Type::PthreadMutexLock>;
using PthreadSpinLock = LockIRImpl<IR::Type::PthreadSpinLock>;

// ==================================================================
// ================= UnlockIR Implementations =======================
// ==================================================================

// UnlockIRImpl should not be used directly. Instead define using alias.
// See PthreadMutexUnlock below as an example.
template <IR::Type T>
class UnlockIRImpl : public UnlockIR {
  const unsigned int lockObjectOffset = 0;
  const llvm::CallBase *inst;

 public:
  explicit UnlockIRImpl(const llvm::CallBase *call) : UnlockIR(T), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }

  [[nodiscard]] const llvm::Value *getLockValue() const override {
    return inst->getArgOperand(lockObjectOffset)->stripPointerCasts();
  }

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

// NOTE: if a specific API semantic is the same as default impl,
// create a type alias.
using PthreadMutexUnlock = UnlockIRImpl<IR::Type::PthreadMutexUnlock>;
using PthreadSpinUnlock = UnlockIRImpl<IR::Type::PthreadSpinUnlock>;

// =================================================================
// ================= Barrier Implementations =======================
// =================================================================

// https://github.com/llvm/llvm-project/blob/d32170dbd5b0d54436537b6b75beaf44324e0c28/openmp/runtime/src/kmp_csupport.cpp#L713
class OpenMPBarrier : public BarrierIR {
  const llvm::CallBase *inst;

 public:
  explicit OpenMPBarrier(const llvm::CallBase *call) : BarrierIR(Type::OpenMPBarrier), inst(call) {}

  [[nodiscard]] inline const llvm::CallBase *getInst() const override { return inst; }
};

// =================================================================
// ================= Other Implementations =========================
// =================================================================

// CallIRImpl should not be used directly. Instead define using alias.
// See OmpForInit below as an example.
template <const IR::Type T>
class CallIRImpl : public CallIR {
 public:
  CallIRImpl(const llvm::CallBase *inst) : CallIR(inst, T) {}

  // Used for llvm style RTTI (isa, dyn_cast, etc.)
  static inline bool classof(const IR *e) { return e->type == T; }
};

using OmpForInit = CallIRImpl<IR::Type::OpenMPForInit>;
using OmpForFini = CallIRImpl<IR::Type::OpenMPForFini>;

using OpenMPSingleStart = CallIRImpl<IR::Type::OpenMPSingleStart>;
using OpenMPSingleEnd = CallIRImpl<IR::Type::OpenMPSingleEnd>;

using OpenMPReduce = CallIRImpl<IR::Type::OpenMPReduce>;

}  // namespace race