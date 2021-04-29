#pragma once

#include <llvm/IR/Instruction.h>

namespace race {

// return true if all pre must always be executed before inst on all paths
bool isDominatedBy(const llvm::Instruction *inst, const llvm::Instruction *pre);

// return true if post must always be exected after inst (but before exiting) on all paths
bool isPostDominatedBy(const llvm::Instruction *inst, const llvm::Instruction *post);

}  // namespace race