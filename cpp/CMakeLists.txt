cmake_minimum_required(VERSION 3.10)
project(Which_are_you)

# 共通する設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-march=native")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# LibTorch
list(APPEND CMAKE_PREFIX_PATH ../../libtorch-1.11.0)
find_package(Torch REQUIRED)

# ソース
file(GLOB SRCS ./*.cc)

# 実行ファイルの定義
add_executable(Which_are_you ${SRCS})
target_link_libraries(Which_are_you ${TORCH_LIBRARIES})
