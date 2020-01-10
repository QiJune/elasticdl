#include "elasticdl/cc/common/embedding_table.h"

namespace elasticdl {
namespace common {

int64_t EmbeddingTable::GetSize() {
  return embedding_dim_ * embedding_vectors_.size();
}

}  // namespace common
}  // namespace elasticdl