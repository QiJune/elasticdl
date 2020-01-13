#include "elasticdl/cc/common/embedding_table.h"

#include "gtest/gtest.h"

using namespace elasticdl::common;

TEST(EmbeddingTable, Init) {
  EmbeddingTable et("e1", ElemType::Float32, 32);
  EXPECT_EQ("e1", et.name());

  float* v1 = et.GetEmbeddingVector<float>(100);
  EXPECT_EQ(et.embedding_vectors().count(100), 1);
  EXPECT_FLOAT_EQ(v1[3], 0.0);

  v1[2] = 0.3;
  float* v2 = et.GetEmbeddingVector<float>(100);
  EXPECT_FLOAT_EQ(v2[2], 0.3);
}

TEST(EmbeddingTable, GetEmbeddingVectors) {
  EmbeddingTable et("e1", ElemType::Float32, 2);
  float* v1 = et.GetEmbeddingVector<float>(1);
  v1[0] = 1.0;
  v1[1] = 2.0;
  int64_t size = 10;
  float* v = new float[size];
  std::vector<int64_t> indices = {1, 2, 4, 5, 7};
  et.GetEmbeddingVectors<float>(indices, v, size);
  EXPECT_FLOAT_EQ(v[0], 1.0);
  EXPECT_FLOAT_EQ(v[1], 2.0);
  EXPECT_FLOAT_EQ(v[3], 0.0);
}