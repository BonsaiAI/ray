load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def ray_deps_setup():
    git_repository(
        name = "rules_jvm_external",
        tag = "2.10",
        remote = "https://github.com/bazelbuild/rules_jvm_external",
    )

    git_repository(
        name = "bazel_common",
        commit = "bf87eb1a4ddbfc95e215b0897f3edc89b2254a1a",
        remote = "https://github.com/google/bazel-common",
    )
 
    git_repository(
        name = "com_github_checkstyle_java",
        commit = "ef367030d1433877a3360bbfceca18a5d0791bdd",
        remote = "https://github.com/ray-project/checkstyle_java",
    )
   
    http_archive(
        # This rule is used by @com_github_nelhage_rules_boost and
        # declaring it here allows us to avoid patching the latter.
        name = "boost",
        build_file = "@com_github_nelhage_rules_boost//:BUILD.boost",
        sha256 = "da3411ea45622579d419bfda66f45cd0f8c32a181d84adfa936f5688388995cf",
        strip_prefix = "boost_1_68_0",
        url = "https://sourceforge.net/projects/boost/files/boost/1.68.0/boost_1_68_0.tar.gz",
        patches = [
            "//thirdparty/patches:boost-exception-no_warn_typeid_evaluated.patch",
        ],
    )

    git_repository(
        name = "com_github_nelhage_rules_boost",
        commit = "df908358c605a7d5b8bbacde07afbaede5ac12cf",
        remote = "https://github.com/nelhage/rules_boost",
        patches = [
            "//thirdparty/patches:rules_boost-undefine-boost_fallthrough.patch",
            "//thirdparty/patches:rules_boost-windows-linkopts.patch",
        ],
    )
   
    git_repository(
        name = "com_github_google_flatbuffers",
        commit = "63d51afd1196336a7d1f56a988091ef05deb1c62",
        remote = "https://github.com/google/flatbuffers.git",
    )
   
    git_repository(
        name = "com_google_googletest",
        commit = "3306848f697568aacf4bcca330f6bdd5ce671899",
        remote = "https://github.com/google/googletest",
    )
   
    git_repository(
        name = "com_github_gflags_gflags",
        remote = "https://github.com/gflags/gflags.git",
        tag = "v2.2.2",
    )
   
    new_git_repository(
        name = "com_github_google_glog",
        build_file = "@//bazel:BUILD.glog",
        commit = "5c576f78c49b28d89b23fbb1fc80f54c879ec02e",
        remote = "https://github.com/google/glog",
    )
   
    new_git_repository(
        name = "plasma",
        build_file = "@//bazel:BUILD.plasma",
        commit = "d00497b38be84fd77c40cbf77f3422f2a81c44f9",
        remote = "https://github.com/apache/arrow",
    )
   
    new_git_repository(
        name = "cython",
        build_file = "@//bazel:BUILD.cython",
        commit = "49414dbc7ddc2ca2979d6dbe1e44714b10d72e7e",
        remote = "https://github.com/cython/cython",
    )
   
    http_archive(
        name = "io_opencensus_cpp",
        strip_prefix = "opencensus-cpp-3aa11f20dd610cb8d2f7c62e58d1e69196aadf11",
        urls = ["https://github.com/census-instrumentation/opencensus-cpp/archive/3aa11f20dd610cb8d2f7c62e58d1e69196aadf11.zip"],
    )
   
    # OpenCensus depends on Abseil so we have to explicitly pull it in.
    # This is how diamond dependencies are prevented.
    git_repository(
        name = "com_google_absl",
        commit = "5b65c4af5107176555b23a638e5947686410ac1f",
        remote = "https://github.com/abseil/abseil-cpp.git",
    )

    # OpenCensus depends on jupp0r/prometheus-cpp
    git_repository(
        name = "com_github_jupp0r_prometheus_cpp",
        commit = "5c45ba7ddc0585d765a43d136764dd2a542bd495",
        # TODO(qwang): We should use the repository of `jupp0r` here when this PR
        # `https://github.com/jupp0r/prometheus-cpp/pull/225` getting merged.
        remote = "https://github.com/ray-project/prometheus-cpp.git",
    )
