#ifndef ELASTICDL_CC_OPTIMIZER_SGD_H_
#define ELASTICDL_CC_OPTIMIZER_SGD_H_

#include <Eigen/Dense>

#include "elasticdl/cc/common/embedding_table.h"
#include "elasticdl/cc/common/tensor.h"
#include "glog/logging.h"

namespace elasticdl {
namespace optimizer {

template <typename T>
void SGD(const common::Tensor& grad, common::Tensor* parameter, double lr) {
  auto g_size = grad.GetSize();
  auto p_size = parameter->GetSize();
  CHECK_EQ(g_size, p_size);
  auto* g = grad.data<T>();
  auto* p = parameter->mutable_data<T>();
  Eigen::Map<const Eigen::Array<T, 1, Eigen::Dynamic>> eg{
      g, static_cast<Eigen::Index>(g_size)};

  Eigen::Map<Eigen::Array<T, 1, Eigen::Dynamic>> ep{
      p, static_cast<Eigen::Index>(g_size)};

  ep -= lr * eg;
}

template <typename T>
void SparseSGD(const common::Tensor& grad,
               common::EmbeddingTable* parameter,
               double lr) {
  int64_t w = grad.GetWidth();
  int64_t embedding_dim = parameter->embedding_dim();
  CHECK_EQ(w, embedding_dim);
  int64_t h = grad.GetHeight();
  auto& indices = grad.indices();
  auto* g = grad.data<T>();
  for (int64_t i = 0; i < h; i++) {
    int64_t index = indices[i];
    T* embedding_vector = parameter->GetEmbeddingVector<T>(index);
    for (int64_t j = 0; j < embedding_dim; j++) {
      embedding_vector[j] -= lr * g[j + i * embedding_dim];
    }
  }
}
}  // namespace optimizer
}  // namespace elasticdl
#endif
