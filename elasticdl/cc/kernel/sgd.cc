#include "elasticdl/cc/kernel/sgd.h"
namespace elasticdl {
namespace kernel {

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

}  // namespace kernel
}  // namespace elasticdl