#include "elasticdl/cc/ps/optimizer.h"

#include <algorithm>

#include "gtest/gtest.h"

using namespace elasticdl::common;
using namespace elasticdl::ps;

TEST(Optimizer, SGD) {
  Parameter p;
  std::vector<float> buffer = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  std::vector<int64_t> dim = {3, 3};
  p.CreateNonEmbeddingParam("t1", ElemType::Float32, dim, buffer.data());
  p.CreateEmbeddingTable("e1", ElemType::Float32, 2, "zero");

  Tensor grad1("t1", ElemType::Float32, {3, 3});
  auto* grad1_data = grad1.mutable_data<float>();
  std::fill(grad1_data, grad1_data + 9, 2.0);

  std::vector<int64_t> indices = {1, 3, 4, 6, 8};
  Tensor grad2("e1", ElemType::Float32, {5, 2}, indices);
  auto* grad2_data = grad2.mutable_data<float>();
  std::fill(grad2_data, grad2_data + 10, 0.2);

  double lr = 0.1;
  SGDOptimizer opt(lr);

  std::vector<Tensor> grads;
  grads.emplace_back(std::move(grad1));
  grads.emplace_back(std::move(grad2));

  opt.ApplyGradients(grads, &p);

  int64_t size = 5 * 2;
  float* res = new float[size];
  p.GetEmbeddingParam<float>("e1", indices, res, size);

  for (int64_t i = 0; i < size; i++) {
    EXPECT_FLOAT_EQ(res[i], -lr * 0.2);
  }

  auto* t = p.non_embedding_params()["t1"];
  auto* t_data = t->data<float>();
  for (int64_t i = 0; i < 9; i++) {
    EXPECT_FLOAT_EQ(t_data[i], 0.1 * (i + 1) - lr * 2.0);
  }
  delete[] res;
}