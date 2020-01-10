#ifndef ELASTICDL_CC_COMMON_EMBEDDING_TABLE_H_
#define ELASTICDL_CC_COMMON_EMBEDDING_TABLE_H_

#include <cassert>
#include <unordered_map>

#include "elasticdl/cc/common/type.h"

namespace elasticdl {
namespace common {

class EmbeddingTable {
 public:
  EmbeddingTable(const std::string& name,
                 const ElemType& type,
                 int64_t embedding_dim,
                 const std::string& initializer = "zero")
      : name_(name),
        element_type_(type),
        embedding_dim_(embedding_dim),
        initializer_(initializer) {}

  ~EmbeddingTable() {
    for (auto& v : embedding_vectors_) {
      delete v.second;
    }
  }

  template <typename T>
  T* GetEmbeddingVector(int64_t index) {
    assert(IsType<T>(element_type_));

    for (auto it = embedding_vectors_.begin(); it != embedding_vectors_.end();
         it++) {
      if (it->first == index) {
        return reinterpret_cast<T*>(it->second);
      }
    }
    T* new_vector = new T[embedding_dim_];
    embedding_vectors_.insert({index, reinterpret_cast<char*>(new_vector)});
    // TODO(qijun) only support zero initializer now
    std::fill(new_vector, new_vector + embedding_dim_, 0);
    return new_vector;
  }

  auto& embedding_vectors() { return embedding_vectors_; }

  int64_t GetSize();

  std::string& name() { return name_; }

 private:
  std::string name_;
  ElemType element_type_;
  int64_t embedding_dim_;
  std::string initializer_;
  std::unordered_map<int64_t, char*> embedding_vectors_;
};
}  // namespace common
}  // namespace elasticdl

#endif