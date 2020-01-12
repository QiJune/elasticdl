load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
git_repository(
    name = "gtest",
    remote = "https://github.com/google/googletest",
    commit = "3306848f697568aacf4bcca330f6bdd5ce671899",
)

http_archive(
    name = "com_github_grpc_grpc",
    urls = [
        "https://github.com/grpc/grpc/archive/de893acb6aef88484a427e64b96727e4926fdcfd.tar.gz",
    ],
    strip_prefix = "grpc-de893acb6aef88484a427e64b96727e4926fdcfd",
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")
grpc_deps()
load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")
grpc_extra_deps()

git_repository(
    name = "glog",
    remote = "https://github.com/google/glog",
    tag = "v0.4.0",
)

http_archive(
    name = "eigen",
    build_file = "//elasticdl/cc/optimizer:eigen.BUILD",
    sha256 = "3a66f9bfce85aff39bc255d5a341f87336ec6f5911e8d816dd4a3fdc500f8acf",
    url = "https://bitbucket.org/eigen/eigen/get/c5e90d9.tar.gz",
    strip_prefix="eigen-eigen-c5e90d9e764e"
)
