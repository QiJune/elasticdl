#include "elasticdl/cc/kernel/sgd.h"

#include "gtest/gtest.h"

using namespace elasticdl::common;
using namespace elasticdl::kernel;

TEST(SGD, Dense) {
  std::vector<float> param(24, 1.0);
  std::vector<float> grad(24, 1.0);
  double lr = 0.1;
  Tensor p("t1", ElemType::Float32, {2, 3, 4}, param.data());
  Tensor g("t1", ElemType::Float32, {2, 3, 4}, grad.data());

  SGD<float>(g, &p, lr);

  for (auto p : param) {
    EXPECT_FLOAT_EQ(p, 0.9);
  }
}

TEST(SGD, Sparse) {
  std::vector<float> grad(10, -1.0);
  std::vector<int64_t> indices = {1, 3, 5, 7, 7};
  Tensor g("t1", ElemType::Float32, {5, 2}, grad.data(), indices);
  EmbeddingTable p("t1", ElemType::Float32, 2);
  double lr = 0.1;
  SparseSGD<float>(g, &p, lr);

  auto& ev = p.embedding_vectors();
  EXPECT_EQ(ev.size(), 4);

  float* r1 = p.GetEmbeddingVector<float>(1);
  EXPECT_FLOAT_EQ(r1[0], 0.1);

  float* r3 = p.GetEmbeddingVector<float>(3);
  EXPECT_FLOAT_EQ(r3[0], 0.1);

  float* r5 = p.GetEmbeddingVector<float>(5);
  EXPECT_FLOAT_EQ(r5[0], 0.1);

  float* r7 = p.GetEmbeddingVector<float>(7);
  EXPECT_FLOAT_EQ(r7[0], 0.2);

  float* r0 = p.GetEmbeddingVector<float>(0);
  EXPECT_FLOAT_EQ(r0[0], 0.0);
}