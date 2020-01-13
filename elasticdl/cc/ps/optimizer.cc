#include "elasticdl/cc/ps/optimizer.h"

#include "elasticdl/cc/kernel/sgd.h"

using namespace elasticdl::common;
using namespace elasticdl::kernel;

#define DISPATCH(function, type, ...) \
  switch (type) {                     \
    case ElemType::Float32:           \
      function<float>(__VA_ARGS__);   \
      break;                          \
    case ElemType::Float64:           \
      function<double>(__VA_ARGS__);  \
      break;                          \
    case ElemType::Int32:             \
      function<int32_t>(__VA_ARGS__); \
      break;                          \
    default:                          \
      LOG(FATAL) << "Invalid Type";   \
  }

namespace elasticdl {
namespace ps {

void SGDOptimizer::ApplyGradients(const std::vector<common::Tensor>& grads,
                                  Parameter* p) {
  if (grads.empty()) return;
  ElemType type = grads[0].element_type();
  auto& non_embedding_params = p->non_embedding_params();
  auto& embedding_params = p->embedding_params();
  for (auto& grad : grads) {
    auto& grad_name = grad.name();
    if (grad.IsSparse()) {
      auto* test_data = grad.data<float>();
      CHECK(embedding_params.count(grad_name));
      auto* embedding_table = embedding_params[grad_name];
      DISPATCH(SparseSGD, type, grad, embedding_table, lr());
    } else {
      CHECK(non_embedding_params.count(grad_name));
      auto* tensor = non_embedding_params[grad_name];
      DISPATCH(SGD, type, grad, tensor, lr());
    }
  }
}
}  // namespace ps
}  // namespace elasticdl