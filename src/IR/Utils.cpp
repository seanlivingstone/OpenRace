#include <IR/Utils.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <set>

using namespace race;

bool race::isDominatedBy(const llvm::Instruction *inst, const llvm::Instruction *pre) {
  auto const targetBlock = pre->getParent();
  auto const &entryBlock = pre->getFunction()->getEntryBlock();

  std::vector<const llvm::BasicBlock *> worklist;
  std::set<const llvm::BasicBlock *> visited;
  worklist.push_back(inst->getParent());

  while (!worklist.empty()) {
    auto const block = worklist.back();
    worklist.pop_back();
    visited.insert(block);

    // stop traversing this path if we hit target block
    if (block == targetBlock) continue;

    // Found a pth reaching entry without going through target first
    if (block == &entryBlock) return false;

    // Keep exploring this path
    for (auto const pred : llvm::predecessors(block)) {
      if (visited.find(pred) == visited.end()) {
        worklist.push_back(pred);
      }
    }
  }
  return true;
}

bool race::isPostDominatedBy(const llvm::Instruction *inst, const llvm::Instruction *post) {
  auto const isExitBlock = [](const llvm::BasicBlock *block) {
    return llvm::isa<llvm::ReturnInst>(block->getTerminator());
  };

  auto const targetBlock = post->getParent();

  std::vector<const llvm::BasicBlock *> worklist;
  std::set<const llvm::BasicBlock *> visited;
  worklist.push_back(inst->getParent());

  while (!worklist.empty()) {
    auto const block = worklist.back();
    worklist.pop_back();
    visited.insert(block);

    // stop traversing this path if we hit target block
    if (block == targetBlock) continue;

    // Found a path reaching exit without going through target first
    if (isExitBlock(block)) return false;

    // Keep exploring this path
    for (auto const succ : llvm::successors(block)) {
      if (visited.find(succ) == visited.end()) {
        worklist.push_back(succ);
      }
    }
  }
  return true;
}