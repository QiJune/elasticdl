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
    name = "sgd",
    srcs = ["sgd.cc"],
    hdrs = ["sgd.h"],
    copts = ["-Iexternal/eigen"],
    deps = common_deps + [
        "@eigen//:eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "sgd_test",
    srcs = ["sgd_test.cc"],
    copts = ["-Iexternal/gtest/include"],
    deps = unittest_deps + [
       "//elasticdl/cc/kernel:sgd",
    ]
)