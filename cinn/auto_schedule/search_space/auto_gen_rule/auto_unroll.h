// Copyright (c) 2022 CINN Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>
#include <vector>

#include "cinn/auto_schedule/search_space/auto_gen_rule/auto_gen_rule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_schedule.h"

namespace cinn {
namespace auto_schedule {

// This rule can be applied in a ScheduleBlock has reduce axis or has loops with non-serial type.
// As a result, it will set a attribute with key named ir::attr::auto_unroll_max_step and value
// indicating max permitted unrolled step in the applied ScheduleBlock. Finally, UnrollLoop pass
// will do unroll based on actual situation.
class AutoUnroll : public AutoGenRule {
 public:
  AutoUnroll(const common::Target& target) : AutoGenRule(target) {}
  ~AutoUnroll() = default;

  RuleApplyType Init(const ir::IRSchedule& init_schedule) override;

  ir::IRSchedule Apply(int index) override;

  std::string GetRuleName() const override { return "AutoUnroll"; }

  AutoGenRule* NewPointer() const override { return new AutoUnroll(*target_); }

 private:
  bool MeetCondition(const ir::ScheduleBlock* schedule_block);

 private:
  std::unique_ptr<ir::IRSchedule> ir_schedule_;
  std::vector<ir::ScheduleBlock*> applicable_schedule_blocks_;
};

}  // namespace auto_schedule
}  // namespace cinn