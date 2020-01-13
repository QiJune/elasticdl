#include "elasticdl/cc/optimizer/sgd.h"
namespace elasticdl {
namespace optimizer {

template <>
void SGD<float>(const common::Tensor& grad,
                common::Tensor* parameter,
                double lr);

template <>
void SGD<double>(const common::Tensor& grad,
                 common::Tensor* parameter,
                 double lr);

template <>
void SparseSGD<float>(const common::Tensor& grad,
                      common::EmbeddingTable* parameter,
                      double lr);

template <>
void SparseSGD<double>(const common::Tensor& grad,
                       common::EmbeddingTable* parameter,
                       double lr);

}  // namespace optimizer
}  // namespace elasticdl