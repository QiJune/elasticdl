#include "elasticdl/cc/ps/parameter.h"

namespace elasticdl {
namespace ps {

void Parameter::CreateEmbeddingTable(const std::string& name,
                                     const common::ElemType& type,
                                     int64_t embedding_dim,
                                     const std::string& initializer) {
  if (embedding_params_.count(name)) return;
  auto* p = new common::EmbeddingTable(name, type, embedding_dim, initializer);
  embedding_params_.emplace(name, p);
}

void Parameter::CreateNonEmbeddingParam(const std::string& name,
                                        const common::ElemType& type,
                                        const std::vector<int64_t>& dim,
                                        void* data) {
  if (non_embedding_params_.count(name)) return;
  auto* p = new common::Tensor(name, type, dim);
  non_embedding_params_.emplace(name, p);
  std::memcpy(static_cast<void*>(p->GetRawDataPointer()),
              data,
              p->GetSize() * GetElementSize(type));
}

}  // namespace ps
}  // namespace elasticdl