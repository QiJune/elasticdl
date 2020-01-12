#include "elasticdl/cc/optimizer/sgd.h"
namespace elasticdl {
namespace optimizer {

template <>
void SGD<float>(const Tensor& grad, Tensor* parameter, double lr);

template <>
void SGD<double>(const Tensor& grad, Tensor* parameter, double lr);

template <>
void SparseSGD<float>(const Tensor& grad, EmbeddingTable* parameter, double lr);

template <>
void SparseSGD<double>(const Tensor& grad,
                       EmbeddingTable* parameter,
                       double lr);

}  // namespace optimizer
}  // namespace elasticdl