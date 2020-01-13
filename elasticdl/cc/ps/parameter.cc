#include "elasticdl/cc/ps/parameter.h"

namespace elasticdl {
namespace ps {

void Parameter::CreateEmbeddingTable(const std::string& name,
                                     const common::ElemType& type,
                                     int64_t embedding_dim,
                                     const std::string& initializer) {
  if (HasEmbeddingParam(name)) return;
  auto* p = new common::EmbeddingTable(name, type, embedding_dim, initializer);
  embedding_params_.emplace(name, p);
}

bool Parameter::HasEmbeddingParam(const std::string& name) {
  for (auto it = embedding_params_.begin(); it != embedding_params_.end();
       it++) {
    if (it->first == name) {
      return true;
    }
  }
  return false;
}

bool Parameter::HasNonEmbeddingParam(const std::string& name) {
  for (auto it = non_embedding_params_.begin();
       it != non_embedding_params_.end();
       it++) {
    if (it->first == name) {
      return true;
    }
  }
  return false;
}

void Parameter::CreateNonEmbeddingParam(const std::string& name,
                                        const common::ElemType& type,
                                        const std::vector<int64_t>& dim,
                                        void* data) {
  if (HasNonEmbeddingParam(name)) return;
  auto* p = new common::Tensor(name, type, dim);
  non_embedding_params_.emplace(name, p);
  std::memcpy(data,
              static_cast<void*>(p->mutable_data<char>()),
              p->GetSize() * GetElementSize(type));
}

}  // namespace ps
}  // namespace elasticdl