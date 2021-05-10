#include "Analysis/OpenMPAnalysis.h"

#include "LanguageModel/OpenMP.h"
#include "Trace/Event.h"
#include "Trace/ThreadTrace.h"

using namespace race;
using namespace llvm;

#define KMPC_STATIC_INIT_PREFIX "__kmpc_for_static_init"
#define KMPC_DISPATCH_INIT_PREFIX "__kmpc_dispatch_init"

namespace {

const llvm::GetElementPtrInst* getArrayAccess(const MemAccessEvent* event) {
  return llvm::dyn_cast<llvm::GetElementPtrInst>(event->getIRInst()->getAccessedValue()->stripPointerCasts());
}

// move add operation out the (sext ) SCEV
class BitExtSCEVRewriter : public llvm::SCEVRewriteVisitor<BitExtSCEVRewriter> {
 private:
  const SCEV *rewriteCastExpr(const SCEVCastExpr *Expr);

 public:
  using super = SCEVRewriteVisitor<BitExtSCEVRewriter>;
  explicit BitExtSCEVRewriter(llvm::ScalarEvolution &SE) : super(SE) {}

  const SCEV *visit(const SCEV *S);

  inline const SCEV *visitZeroExtendExpr(const SCEVZeroExtendExpr *Expr) {
    return rewriteCastExpr(Expr);
  };

  inline const SCEV *visitSignExtendExpr(const SCEVSignExtendExpr *Expr) {
    return rewriteCastExpr(Expr);
  }
};

class OpenMPLoopManager {
 private:
  Function *F;

  // dependent pass from LLVM
  //LoopInfo *LI;
  DominatorTree *DT;

  // cached result. TODO: use const pointer
  SmallDenseMap<BasicBlock *, CallBase *, 4> ompStaticInitBlocks;
  SmallDenseMap<BasicBlock *, CallBase *, 4> ompDispatchInitBlocks;

  void init();

  int64_t resolveBoundValue(const AllocaInst *V, const CallBase *initCall) const;
 public:
  // constructor
  OpenMPLoopManager(AnalysisManager<Function> &FAM, Function &fun)
      : F(&fun),
        //LI(&FAM.getResult<LoopAnalysis>(fun)),
        DT(&FAM.getResult<DominatorTreeAnalysis>(fun)) { init(); }

  void rebuildWith(AnalysisManager<Function> &FAM, Function &F);
  // getter
  [[nodiscard]] inline Function *getTargetFunction() const { return F; }

  // query.
  // TODO: handle dynamic dispatch calls.
  inline CallBase *getStaticInitCallIfExist(const BasicBlock *block) const {
    auto it = ompStaticInitBlocks.find(block);
    return it == ompStaticInitBlocks.end() ? nullptr : it->second;
  }

  inline CallBase *getStaticInitCallIfExist(const Loop *L) const {
    if (L->getLoopPreheader() == nullptr) {
      return nullptr;
    }

    auto initBlock = L->getLoopPreheader()->getUniquePredecessor();
    return getStaticInitCallIfExist(initBlock);
  }

  std::pair<int64_t, int64_t> resolveOMPLoopBound(const CallBase *initForCall) const;

  static constexpr int64_t InvalidLoopBound () {
    return std::numeric_limits<int64_t>::min();
  }
};

}  // namespace

static CallBase *getOMPStaticInitCall(const Loop *L, const OpenMPLoopManager &ompManger) {
  if (L->getLoopPreheader() != nullptr &&
      L->getLoopPreheader()->getUniquePredecessor() != nullptr) {
    // find the corresponding static init call
    return ompManger.getStaticInitCallIfExist(L->getLoopPreheader()->getUniquePredecessor());
  }
  return nullptr;
}

// TODO: handle dynamic dispatch for loop
static inline bool isOMPForLoop(const Loop *L, const OpenMPLoopManager &ompManager) {
  return getOMPStaticInitCall(L, ompManager) != nullptr;
}

template <typename PredTy>
static const SCEV *FindSCEVExpr(const llvm::SCEV *Root, PredTy Pred) {
  struct FindClosure {
    const SCEV *Found = nullptr;
    PredTy Pred;

    FindClosure(PredTy Pred) : Pred(Pred) {}

    bool follow(const llvm::SCEV *S) {
      if (!Pred(S)) return true;

      Found = S;
      return false;
    }

    bool isDone() const { return Found != nullptr; }
  };

  FindClosure FC(Pred);
  visitAll(Root, FC);
  return FC.Found;
}

static inline const SCEV *stripSCEVBaseAddr(const SCEV *root) {
  return FindSCEVExpr(root, [](const llvm::SCEV *S) -> bool {
    return isa<llvm::SCEVAddRecExpr>(S);
  });
}

static const SCEVAddRecExpr *getOMPLoopSCEV(const llvm::SCEV *root, const OpenMPLoopManager &ompManager) {
  // get the outter-most loop (omp loop should always be the outter-most
  // loop
  auto omp = FindSCEVExpr(root, [&](const llvm::SCEV *S) -> bool {
    if (auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(S)) {
      if (isOMPForLoop(addRec->getLoop(), ompManager)) {
        return true;
      }
    }
    return false;
  });

  return llvm::dyn_cast_or_null<llvm::SCEVAddRecExpr>(omp);
}

const SCEV * BitExtSCEVRewriter::visit(const SCEV *S) {
  auto result = super::visit(S);
  // recursively into the sub expression
  while (result != S) {
    S = result;
    result = super::visit(S);
  }
  return result;
}

const SCEV * BitExtSCEVRewriter::rewriteCastExpr(const SCEVCastExpr *Expr) {
  auto buildCastExpr = [&] (const SCEV *op, Type *type) -> const SCEV *{
    switch (Expr->getSCEVType()) {
      case scSignExtend:
        return SE.getSignExtendExpr(op, type);
      case scZeroExtend:
        return SE.getZeroExtendExpr(op, type);
      default:
        llvm_unreachable("unhandled type of scev cast expression");
    }
  };

  const llvm::SCEV *Operand = super::visit(Expr->getOperand());
  if (auto add = llvm::dyn_cast<llvm::SCEVNAryExpr>(Operand)) {
    llvm::SmallVector<const llvm::SCEV *, 2> Operands;
    for (auto op : add->operands()) {
      Operands.push_back(buildCastExpr(op, Expr->getType()));
    }
    switch (add->getSCEVType()) {
      case llvm::scMulExpr:
        return SE.getMulExpr(Operands);
      case llvm::scAddExpr:
        return SE.getAddExpr(Operands);
      case llvm::scAddRecExpr:
        auto addRec = llvm::dyn_cast<llvm::SCEVAddRecExpr>(add);
        return SE.getAddRecExpr(Operands, addRec->getLoop(), addRec->getNoWrapFlags());
    }
  }
  return Operand == Expr->getOperand() ? Expr : buildCastExpr(Operand, Expr->getType());
}

void OpenMPLoopManager::rebuildWith(AnalysisManager<Function> &FAM, Function &fun) {
  this->F = &fun;
  //this->LI = &FAM.getResult<LoopAnalysis>(fun);
  this->DT = &FAM.getResult<DominatorTreeAnalysis>(fun);

  ompDispatchInitBlocks.clear();
  ompStaticInitBlocks.clear();

  init();
}

void OpenMPLoopManager::init() {
  // initialize the map to the omp related calls
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (auto call = dyn_cast<CallBase>(&I)) {
        if (call->getCalledFunction() != nullptr && call->getCalledFunction()->hasName()) {
          auto funcName = call->getCalledFunction()->getName();
          if (funcName.startswith(KMPC_STATIC_INIT_PREFIX)) {
            this->ompStaticInitBlocks.insert(std::make_pair(&BB, call));
          } else if (funcName.startswith(KMPC_DISPATCH_INIT_PREFIX)) {
            this->ompDispatchInitBlocks.insert(std::make_pair(&BB, call));
          }
        }
      }
    }
  }
}

int64_t OpenMPLoopManager::resolveBoundValue(const AllocaInst *V, const CallBase *initCall) const {

  const llvm::StoreInst *storeInst = nullptr;
  for (auto user : V->users()) {
    if (auto SI = llvm::dyn_cast<llvm::StoreInst>(user)) {
      // simple cases, only has one store instruction
      if (storeInst == nullptr) {
        if (this->DT->dominates(SI, initCall)) {
          storeInst = SI;
        }
      } else {
        if (this->DT->dominates(SI, initCall)) {
          // LOG_DEBUG("omp bound has one than one store!!"); // TODO: Peiming clarify
          return InvalidLoopBound();
        }
      }
    }
  }

  if (storeInst) {
    auto bound = dyn_cast<ConstantInt>(storeInst->getValueOperand());
    if (bound) {
      return bound->getSExtValue();
    }
    return InvalidLoopBound();
  } else {
    // LOG_DEBUG("omp bound has no store??");
    return InvalidLoopBound();
  }
}

std::pair<int64_t, int64_t> OpenMPLoopManager::resolveOMPLoopBound(const CallBase *initForCall) const {
  Value *ompLB = nullptr, *ompUB = nullptr; // up bound and lower bound
  if (initForCall->getCalledFunction()->getName().startswith(KMPC_STATIC_INIT_PREFIX)) {
    ompLB = initForCall->getArgOperand(4);
    ompUB = initForCall->getArgOperand(5);
  } else if (initForCall->getCalledFunction()->getName().startswith(KMPC_DISPATCH_INIT_PREFIX)) {
    ompLB = initForCall->getArgOperand(3);
    ompUB = initForCall->getArgOperand(4);
  } else {
    return std::make_pair(InvalidLoopBound(), InvalidLoopBound());
  }


  auto allocaLB = llvm::dyn_cast<llvm::AllocaInst>(ompLB);
  auto allocaUB = llvm::dyn_cast<llvm::AllocaInst>(ompUB);

  // omp.ub and omp.lb are always alloca?
  if (allocaLB == nullptr || allocaUB == nullptr) {
    return std::make_pair(InvalidLoopBound(), InvalidLoopBound());
  }

  auto LB = resolveBoundValue(allocaLB, initForCall);
  auto UB = resolveBoundValue(allocaUB, initForCall);
  return std::make_pair(LB, UB);
}


OpenMPAnalysis::OpenMPAnalysis() { PB.registerFunctionAnalyses(FAM); }

bool OpenMPAnalysis::canIndexOverlap(const race::MemAccessEvent* event1, const race::MemAccessEvent* event2) {
  auto gep1 = getArrayAccess(event1);
  if (!gep1) return false;

  auto gep2 = getArrayAccess(event2);
  if (!gep2) return false;

  // should be in same function
  if (gep1->getFunction() != gep2->getFunction()) {
    return false;
  }

  auto &targetFun = *const_cast<llvm::Function*>(gep1->getFunction());
  static OpenMPLoopManager ompManager(FAM, targetFun);

  if (&targetFun != ompManager.getTargetFunction()) {
    // we are now analyzing a new function
    ompManager.rebuildWith(FAM, targetFun);
  }

  // TODO: get rid of const cast? Also does FAM cache these results (I think it does?)
  auto& LI = FAM.getResult<LoopAnalysis>(targetFun);
  auto& scev = FAM.getResult<ScalarEvolutionAnalysis>(targetFun);

  BitExtSCEVRewriter rewriter(scev);
  auto scev1 = scev.getSCEV(const_cast<llvm::Value*>(llvm::cast<llvm::Value>(gep1)));
  auto scev2 = scev.getSCEV(const_cast<llvm::Value*>(llvm::cast<llvm::Value>(gep2)));
  // the rewriter here move the bit extension operation into the deepest scopy
  // e.g., (4 + (4 * (sext i32 (2 * %storemerge2) to i64))<nsw> + %a) will be rewrited to
  //   ==> (4 + (8 * (sext i32 %storemerge2 to i64)) + %a)
  // this will simplied the scev expression as sext and zext are considered as variable instead of constant
  // during the computation between two scev expression.
  scev1 = rewriter.visit(scev1);
  scev2 = rewriter.visit(scev2);

  auto diff = dyn_cast<SCEVConstant>(scev.getMinusSCEV(scev1, scev2));

  if (diff == nullptr) {
    // TODO: we are unable to analyze unknown gap array index for now.
    return true;
  }

  if (diff->isZero()) {
    // simplest case, two access the same element, no race for sure
    return false;
  }

  /* two gep points to different elements **/

  // strip off the scev
  auto omp1 = getOMPLoopSCEV(scev1, ompManager);
  auto omp2 = getOMPLoopSCEV(scev2, ompManager);

  // some SCEV is in the form %base + {expr,+,%strip}<omp.loop>
  // since here the gap between two accesses are constant, the variable %base can simply be ignored.
  scev1 = stripSCEVBaseAddr(scev1);
  scev2 = stripSCEVBaseAddr(scev2);

  if (omp1 && omp2) {
    // both access are in the omp loop
    if (omp1 == scev1 && omp2 == scev2) {
      // the outside-most loop are the parallel loop as well instead of being parallel loop nested in a
      // nested loop.
      uint64_t distance = diff->getAPInt().abs().getLimitedValue();

      if (omp1->isAffine()) {
        auto step = omp1->getOperand(1);
        if (auto constStep = llvm::dyn_cast<llvm::SCEVConstant>(step)) {
          // the step of the loop
          uint64_t loopStep = constStep->getAPInt().abs().getLimitedValue();
          // assume we iterate at least one time
          if (distance == loopStep) {
            return true;
          }

          CallBase *initForCall = ompManager.getStaticInitCallIfExist(omp1->getLoop());
          auto bounds = ompManager.resolveOMPLoopBound(initForCall);
          int64_t lowerBound = std::abs(bounds.first);
          int64_t upperBound = std::abs(bounds.second);

          if (std::max(lowerBound, upperBound) < (distance / loopStep)) {
            return false;
          }
        }
      }
    }
  }

  // If unsure report they do alias
  llvm::errs() << "unsure so reporting alias\n";
  return true;
}

namespace {
// Get list of (non-nested) event regions
// template definition can be in cpp as long as we dont expose the template outside of this file
template <IR::Type Start, IR::Type End>
std::vector<Region> getRegions(const ThreadTrace& thread) {
  std::vector<Region> regions;

  std::optional<EventID> start;
  for (auto const& event : thread.getEvents()) {
    switch (event->getIRInst()->type) {
      case Start: {
        assert(!start.has_value() && "encountered two start types in a row");
        start = event->getID();
        break;
      }
      case End: {
        assert(start.has_value() && "encountered end type without a matching start type");
        regions.emplace_back(start.value(), event->getID());
        start.reset();
        break;
      }
      default:
        // Nothing
        break;
    }
  }

  return regions;
}

auto constexpr getLoopRegions = getRegions<IR::Type::OpenMPForInit, IR::Type::OpenMPForFini>;
auto constexpr getSingleRegions = getRegions<IR::Type::OpenMPSingleStart, IR::Type::OpenMPSingleEnd>;
}  // namespace

const std::vector<OpenMPAnalysis::LoopRegion>& OpenMPAnalysis::getOmpForLoops(const ThreadTrace& thread) {
  // Check if result is already computed
  auto it = ompForLoops.find(thread.id);
  if (it != ompForLoops.end()) {
    return it->second;
  }

  // Else find the loop regions
  auto const loopRegions = getLoopRegions(thread);
  ompForLoops[thread.id] = loopRegions;

  return ompForLoops.at(thread.id);
}

bool OpenMPAnalysis::inParallelFor(const race::MemAccessEvent* event) {
  auto loopRegions = getOmpForLoops(event->getThread());
  auto const eid = event->getID();
  for (auto const& region : loopRegions) {
    if (region.contains(eid)) return true;
    // Break early if we pass the eid without finding matching region
    if (region.end > eid) return false;
  }

  return false;
}

bool OpenMPAnalysis::isLoopArrayAccess(const race::MemAccessEvent* event1, const race::MemAccessEvent* event2) {
  auto gep1 = getArrayAccess(event1);
  if (!gep1) return false;

  auto gep2 = getArrayAccess(event2);
  if (!gep2) return false;

  return inParallelFor(event1) && inParallelFor(event2);
}

bool OpenMPAnalysis::inSameTeam(const Event* event1, const Event* event2) const {
  // Check both spawn events are OpenMP forks
  auto e1Spawn = event1->getThread().spawnSite;
  if (!e1Spawn || (e1Spawn.value()->getIRInst()->type != IR::Type::OpenMPFork)) return false;

  auto e2Spawn = event2->getThread().spawnSite;
  if (!e2Spawn || (e2Spawn.value()->getIRInst()->type != IR::Type::OpenMPFork)) return false;

  // Check they are spawned from same thread
  if (e1Spawn.value()->getThread().id != e2Spawn.value()->getThread().id) return false;

  // Check that they are adjacent. Only matching omp forks can be adjacent, because they are always followed by joins
  auto const eid1 = e1Spawn.value()->getID();
  auto const eid2 = e2Spawn.value()->getID();
  auto const diff = (eid1 > eid2) ? (eid1 - eid2) : (eid2 - eid1);
  return diff == 1;
}

bool OpenMPAnalysis::inSameSingleBlock(const Event* event1, const Event* event2) const {
  assert(inSameTeam(event1, event2));

  auto const eid1 = event1->getID();
  auto const eid2 = event2->getID();

  // Trace events are ordered, so we can save time by finding the region containing the smaller
  // ID first, and then checking if that region also contains the bigger ID.
  auto const minID = (eid1 < eid2) ? eid1 : eid2;
  auto const maxID = (eid1 > eid2) ? eid1 : eid2;

  // Omp threads in same team will have identical traces so we only need one set of events
  auto const singleRegions = getSingleRegions(event1->getThread());
  for (auto const& region : singleRegions) {
    // If region contains one, check if it also contains the other
    if (region.contains(minID)) return region.contains(maxID);

    // End early if end of this region passes smaller event ID
    if (region.end > minID) return false;
  }
  return false;
}

