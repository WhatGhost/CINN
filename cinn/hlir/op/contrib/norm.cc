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

#include "cinn/hlir/op/contrib/norm.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cinn/common/cas.h"
#include "cinn/common/common.h"
#include "cinn/common/context.h"
#include "cinn/common/macros.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/op/op_util.h"
#include "cinn/hlir/pe/elementwise.h"
#include "cinn/hlir/pe/ir_schedule_pe.h"
#include "cinn/hlir/pe/nn.h"
#include "cinn/hlir/pe/schedule.h"
#include "cinn/ir/ir.h"
#include "cinn/ir/ir_base.h"
#include "cinn/ir/tensor.h"
#include "cinn/lang/builtin.h"
#include "cinn/lang/compute.h"
#include "gflags/gflags.h"
DECLARE_bool(cinn_ir_schedule);

namespace cinn {
namespace hlir {
namespace op {

using common::CINNValue;
using common::CINNValuePack;

ir::Tensor Norm(const ir::Tensor &in_tensor, const int32_t axis, const float epsilon, const std::string &output_name) {
  int ndim = static_cast<int>(in_tensor->shape.size());
  CHECK_GT(in_tensor->shape.size(), 0) << "Norm requires input tensor's shape to be greater than 0\n";
  CHECK(axis == -1 || (0 <= axis && axis < ndim)) << "Axis must be -1 or between -1 and ndim - 1";
  CHECK(in_tensor->type().is_float(16) || in_tensor->type().is_float(32) || in_tensor->type().is_float(64))
      << "norm's input tensor currently only float is supported in norm.";
  int true_axis = (axis == -1) ? ndim : axis;
  std::vector<ir::Var> axis_vars;
  axis_vars.push_back(ir::Var(true_axis, common::UniqName("norm_axis_vars")));
  return lang::Compute(
      in_tensor->shape,
      [&](const std::vector<ir::Expr> &indices) {
        ir::Tensor in_tensor(in_tensor);
        auto ori_data = in_tensor(indices);
        // auto pow_res = lang::Pow(ori_data, common::make_const("int32", 2));
        auto pow_sum  = lang::ReduceSum(lang::Pow(ori_data, common::make_const(2)), axis_vars);
        auto sqrt_res = lang::Sqrt(pow_sum + common::make_const(epsilon));
        return common::AutoSimplify(ori_data / sqrt_res);
      },
      common::UniqName(output_name));
}

std::shared_ptr<framework::OpStrategy> StrategyForNorm(const framework::NodeAttr &attrs,
                                                       const std::vector<ir::Tensor> &inputs,
                                                       const std::vector<Type> &out_type,
                                                       const std::vector<std::vector<int>> &output_shapes,
                                                       const Target &target) {
  std::string op_name("norm");
  const auto &attr_store = attrs.attr_store;
  CHECK(attr_store.count("axis")) << "find no attr of axis";
  CHECK(attr_store.count("epsilon")) << "find no attr of epsilon";
  auto axis    = absl::get<int32_t>(attr_store.at("axis"));
  auto epsilon = absl::get<float>(attr_store.at("epsilon"));

  framework::CINNCompute norm_compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK(!args.empty()) << "The input argument of " << op_name << " compute is empty! Please check.\n";
    CINNValuePack pack_args = args[0];
    CHECK(!pack_args.empty()) << "at least one input tensor for " << op_name << " compute\n";

    std::string tensor_name = UniqName("Norm_out");

    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2);
      CHECK(pack_args[1].is_string());
      tensor_name = pack_args[1].operator std::string();
    }

    Expr A = pack_args[0];
    CHECK(A.as_tensor());
    CHECK(!output_shapes.empty());
    auto tensor_A = A.as_tensor_ref();

    auto stages = CreateStages({tensor_A});
    VLOG(3) << "A shape: " << utils::Join(tensor_A->shape, ", ")
            << ", output_shapes: " << utils::Join(output_shapes[0], ", ");

    if (FLAGS_cinn_ir_schedule) {
      CHECK_EQ(pack_args.size(), 2U);
      tensor_name = pack_args[1].operator std::string();
    }

    ir::Tensor out = Norm(tensor_A, axis, epsilon, tensor_name);

    std::vector<CINNValue> res;
    stages->InsertLazily(out);
    res.push_back(CINNValue(out));
    CHECK(!out_type.empty()) << "Output type of Norm is empty! Please check.\n";
    res.push_back(CINNValue(stages));
    *ret = CINNValuePack{res};
  });

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(norm_compute, GetInjectiveScheduleFunc(output_shapes, target), "strategy.norm.x86", 1);
  return strategy;
}

std::vector<Type> InferDtypeForNorm(const std::vector<Type> &inputs_type, const framework::AttrMapType &attrs) {
  CHECK(!inputs_type.empty()) << "The input's type size is 0! Please check again.";
  std::vector<Type> res{inputs_type[0]};
  return res;
}

std::vector<framework::shape_t> InferShapeForNorm(const std::vector<framework::shape_t> &inputs_shape,
                                                  const framework::AttrMapType &attrs) {
  CHECK(!inputs_shape.empty() && !inputs_shape[0].empty()) << "The input's shape size is 0! Please check again.";
  std::vector<framework::shape_t> res{inputs_shape[0]};
  return res;
}

}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(norm_ops) {
  CINN_REGISTER_OP(norm)
      .describe("Norm.")
      .set_num_inputs(1)
      .set_num_outputs(1)
      .set_attr<cinn::hlir::framework::StrategyFunction>("CINNStrategy", cinn::hlir::op::StrategyForNorm)
      .set_attr("infershape", MakeOpFunction(cinn::hlir::op::InferShapeForNorm))
      .set_attr("inferdtype", MakeOpFunction(cinn::hlir::op::InferDtypeForNorm))
      .set_attr<cinn::hlir::framework::OpPatternKind>("OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible);
  // .set_support_level(4);

  return true;
}