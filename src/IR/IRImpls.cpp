#include "IR/IRImpls.h"

#include "IR/Utils.h"

using namespace race;

const llvm::Instruction *OpenMPReduce::getEnd() const {
  auto const startInst = getInst();
  auto const switchInst = llvm::dyn_cast<llvm::SwitchInst>(startInst->getNextNode());
  assert(switchInst && "Instruction after reduce call should always be switch");
  // Default dest of reduce switch points to end of reduce code
  auto const endBlock = switchInst->getDefaultDest();
  return endBlock->getFirstNonPHIOrDbg();
}

bool OpenMPReduce::contains(const llvm::Instruction *inst) const {
  // reduce should be within same function, so doa  quick check first
  if (inst->getFunction() != getInst()->getFunction()) return false;
  // if not check that inst is between start and end of reduce
  return isDominatedBy(inst, getInst()) && isPostDominatedBy(inst, getEnd());
}