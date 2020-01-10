#ifndef ELASTICDL_CC_OPTIMIZER_SGD_H_
#define ELASTICDL_CC_OPTIMIZER_SGD_H_

#include <Eigen/Dense>

#include "elasticdl/cc/common/embedding_table.h"
#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace optimizer {

template <typename T>
void SGD(const Tensor& grad, Tensor* parameter, double lr) {}

template <typename T>
void SparseSGD(const Tensor& grad, EmbeddingTable* parameter, double lr) {}

}  // namespace optimizer
}  // namespace elasticdl
#endif
