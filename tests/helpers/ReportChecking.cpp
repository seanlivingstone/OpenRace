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

#include "helpers/ReportChecking.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>

#include <catch2/catch.hpp>

#include "RaceDetect.h"

namespace {
unsigned int stringToUnsigned(llvm::StringRef s) {
  std::istringstream read(s);
  unsigned int val = 0;
  read >> val;
  return val;
}

// assumes s is in format "file:line:col"
race::SourceLoc locFromString(llvm::StringRef s) {
  auto parts = s.split(":");
  auto file = parts.first;
  parts = parts.second.split(":");
  auto line = stringToUnsigned(parts.first);
  auto col = stringToUnsigned(parts.second);

  return race::SourceLoc{file, line, col};
}

}  // namespace

TestRace TestRace::fromString(llvm::StringRef s) {
  auto parts = s.split(" ");
  auto loc1 = locFromString(parts.first);
  auto loc2 = locFromString(parts.second);

  return TestRace{loc1, loc2};
}

// build set of races from strings
std::vector<TestRace> TestRace::fromStrings(std::vector<llvm::StringRef> strings) {
  std::vector<TestRace> out;

  for (auto const &s : strings) {
    out.push_back(TestRace::fromString(s));
  }

  return out;
}

bool TestRace::equals(const race::Race &race) const {
  if (race.missingLocation()) return false;

  return race.first.location.value() == first && race.second.location.value() == second;
}

bool reportContains(const race::Report &report, TestRace race) {
  return std::find_if(report.races.begin(), report.races.end(),
                      [&](const race::Race &reportRace) { return race.equals(reportRace); }) != report.races.end();
}
bool reportContains(const race::Report &report, std::vector<TestRace> races) {
  // loop over report, removing any matched races from the list of test races
  for (auto const &reportRace : report.races) {
    auto it = std::find_if(races.begin(), races.end(), [&](const TestRace &race) { return race.equals(reportRace); });
    if (it != races.end()) {
      races.erase(it);
      if (races.empty()) break;
    }
  }

  return races.empty();
}

Oracle::Oracle(llvm::StringRef filename, std::vector<llvm::StringRef> races) : filename(filename) {
  expectedRaces = TestRace::fromStrings(races);
}

void checkOracles(const std::vector<Oracle> &oracles, llvm::StringRef llPath) {
  llvm::LLVMContext context;
  llvm::SMDiagnostic err;

  for (auto const &oracle : oracles) {
    SECTION("test " + oracle.filename.str()) {
      auto testfile = llPath.str() + oracle.filename.str();
      auto module = llvm::parseIRFile(testfile, err, context);
      if (!module) {
        err.print(oracle.filename.str().c_str(), llvm::errs());
      }
      REQUIRE(module.get() != nullptr);

      auto report = race::detectRaces(module.get());
      // llvm::errs() << "===> Detected Races:\n";
      // for (auto const &race : report.races) {
      //   llvm::errs() << race << "\n";
      // }
      // llvm::errs() << "\n";

      for (auto const &expectedRace : oracle.expectedRaces) {
        CHECK(reportContains(report, expectedRace));
      }
      CHECK(report.size() == oracle.expectedRaces.size());
    }
  }
}