#ifndef ELASTICDL_CC_PS_PARAMETER_H_
#define ELASTICDL_CC_PS_PARAMETER_H_

#include "elasticdl/cc/common/embedding_table.h"
#include "elasticdl/cc/common/tensor.h"

namespace elasticdl {
namespace ps {

class Parameter {
 public:
  ~Parameter() {
    for (auto& p : non_embedding_params_) {
      delete p.second;
    }

    for (auto& p : embedding_params_) {
      delete p.second;
    }
  }

  void CreateEmbeddingTable(const std::string& name,
                            const common::ElemType& type,
                            int64_t embedding_dim,
                            const std::string& initializer);

  template <typename T>
  void GetEmbeddingParam(const std::string& name,
                         const std::vector<int64_t>& indices,
                         T* buffer,
                         int64_t size) {
    CHECK(HasEmbeddingParam(name));
    embedding_params_[name]->GetEmbeddingVectors<T>(indices, buffer, size);
  }

  bool HasEmbeddingParam(const std::string& name);

  bool HasNonEmbeddingParam(const std::string& name);

  void CreateNonEmbeddingParam(const std::string& name,
                               const common::ElemType& type,
                               const std::vector<int64_t>& dim,
                               void* data);

 private:
  std::unordered_map<std::string, common::Tensor*> non_embedding_params_;
  std::unordered_map<std::string, common::EmbeddingTable*> embedding_params_;
};

}  // namespace ps
}  // namespace elasticdl
#endif