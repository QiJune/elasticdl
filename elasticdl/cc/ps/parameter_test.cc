#include "elasticdl/cc/ps/parameter.h"

#include "gtest/gtest.h"

using namespace elasticdl::common;
using namespace elasticdl::ps;

TEST(Parameter, CreateEmbeddingTable) {
  Parameter p;
  p.CreateEmbeddingTable("e1", ElemType::Float32, 2, "zero");
  EXPECT_EQ(p.embedding_params().count("e1"), 1);

  int64_t size = 10;
  float* buffer = new float[size];
  p.GetEmbeddingParam<float>("e1", {1, 3, 5, 7, 9}, buffer, size);
  EXPECT_FLOAT_EQ(buffer[0], 0.0);
  EXPECT_EQ(p.embedding_params()["e1"]->embedding_vectors().count(1), 1);
  EXPECT_EQ(p.embedding_params()["e1"]->embedding_vectors().count(3), 1);
  delete[] buffer;
}

TEST(Parameter, CreateNonEmbeddingParam) {
  Parameter p;
  std::vector<float> buffer = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
  std::vector<int64_t> dim = {3, 3};
  p.CreateNonEmbeddingParam("t1", ElemType::Float32, dim, buffer.data());
  EXPECT_EQ(p.non_embedding_params().count("t1"), 1);

  auto* data = p.non_embedding_params()["t1"]->data<float>();
  EXPECT_FLOAT_EQ(data[0], 0.1);
  EXPECT_FLOAT_EQ(data[3], 0.3);
}