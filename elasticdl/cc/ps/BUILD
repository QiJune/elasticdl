unittest_deps = [
    "@gtest//:gtest",
    "@gtest//:gtest_main",
    "@glog//:glog",
    "@com_github_gflags_gflags//:gflags",
]

common_deps = [
    "//elasticdl/cc/common:tensor",
    "//elasticdl/cc/common:embedding_table",
]

cc_library(
    name = "parameter",
    srcs = ["parameter.cc"],
    hdrs = ["parameter.h"],
    deps = common_deps,
    visibility = ["//visibility:public"],
)
