#include "cinn/hlir/framework/op.h"

#include <gtest/gtest.h>

#include <functional>

#include <string>

#include "cinn/cinn.h"

#include "cinn/hlir/framework/node.h"
#include "cinn/hlir/framework/op_strategy.h"
#include "cinn/hlir/pe/broadcast.h"

namespace cinn {
namespace hlir {
namespace framework {
using StrategyFunction = std::function<std::shared_ptr<OpStrategy>(
    const NodeAttr, const std::vector<ir::Tensor>, common::Type, const common::Target)>;

using CCompute = std::function<std::shared_ptr<ir::Tensor>(const std::vector<ir::Tensor>)>;

CINN_REGISTER_OP(add)
    .describe("test of op Add")
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<std::string>("nick_name", "plus")
    .set_support_level(4);

std::shared_ptr<OpStrategy> StrategyTest(const NodeAttr &attr,
                                         const std::vector<ir::Tensor> &inputs,
                                         common::Type out_type,
                                         const common::Target &target) {
  ir::PackedFunc::body_t compute_body = [](ir::Args args, ir::RetValue *ret) {
    common::CINNValuePackShared a = args[0];
    ir::Expr A                    = a[0];
    ir::Expr B                    = a[1];
    *ret                          = common::CINNValuePack::Make(
        {common::CINNValue(ir::Expr(pe::Add(A.as_tensor_ref(), B.as_tensor_ref(), "C").get()))});
  };
  ir::PackedFunc fcompute(compute_body);

  ir::PackedFunc::body_t schedule_body = [](ir::Args args, ir::RetValue *ret) {
    common::CINNValuePackShared a = args[0];
    ir::Expr A                    = a[0];
    A.as_tensor_ref()->stage()->Vectorize(1, 16);
    A.as_tensor_ref()->stage()->Unroll(1);
    *ret = common::CINNValuePack::Make({common::CINNValue(A)});
  };
  ir::PackedFunc fschedule(schedule_body);

  std::shared_ptr<OpStrategy> strategy = std::make_shared<OpStrategy>();

  if (target.arch == common::Target::Arch ::X86) {
    strategy->AddImpl(fcompute, fschedule, "test.strategy.x86", 10);
  } else {
    strategy->AddImpl(fcompute, fschedule, "test.strategy.else", 10);
  }
  strategy->AddImpl(fcompute, fschedule, "test.strategy.lowPlevel.x86", 5);
  return strategy;
}

TEST(Operator, GetAttr) {
  auto add      = Operator::Get("add");
  Operator temp = *add;
  temp.set_attr<StrategyFunction>("CINNStrategy", StrategyTest);
  auto nick     = Operator::GetAttr<std::string>("nick_name");
  auto strategy = Operator::GetAttr<StrategyFunction>("CINNStrategy");

  Expr M(100), N(32);
  Placeholder<float> A("A", {M, N});
  Placeholder<float> B("B", {M, N});

  NodeAttr attr;
  std::vector<ir::Tensor> inputs{A, B};
  common::Type type;
  common::Target target;
  target.arch = common::Target::Arch::X86;
  auto impl   = SelectImpl(strategy[add](attr, inputs, type, target));

  common::CINNValuePackShared cinn_input = common::CINNValuePack::Make({common::CINNValue(A), common::CINNValue(B)});
  common::CINNValuePackShared C          = impl->fcompute(cinn_input);
  C                                      = impl->fschedule(C);
  for (int i = 0; i < C.get()->size(); i++) {
    ir::Expr temp = C[i];
    inputs.push_back(temp.as_tensor_ref());
  }
  auto func = Lower("add1", inputs);
  LOG(INFO) << "Test Strategy Codegen:\n" << func;

  ASSERT_EQ(impl->name, "test.strategy.x86");
  ASSERT_EQ(add->description, "test of op Add");
  ASSERT_EQ(nick[add], "plus");
}

}  // namespace framework
}  // namespace hlir
}  // namespace cinn