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