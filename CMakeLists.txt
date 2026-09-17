cmake_minimum_required(VERSION 3.20)
project(SciComputeInfra VERSION 0.1.0 LANGUAGES CXX CUDA)

option(SCI_BUILD_TESTS "Build tests" OFF)
option(SCI_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(SCI_ENABLE_CUDA "Enable CUDA support" ON)
option(SCI_ENABLE_OPENMP "Enable OpenMP for CPU parallelism" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

if(SCI_ENABLE_CUDA)
    # 默认覆盖常见架构；可通过 -DSCI_CUDA_ARCHITECTURES="89;120" 覆盖。
    # 注意不能用 CMAKE_CUDA_ARCHITECTURES 直接放缓存变量：CMake 在 project()
    # 检测 CUDA 编译器时已写入该缓存项，不带 FORCE 的 set(... CACHE ...) 会被忽略。
    set(SCI_CUDA_ARCHITECTURES "75;89;90;120"
        CACHE STRING "CUDA architectures to build for")
    set(CMAKE_CUDA_ARCHITECTURES "${SCI_CUDA_ARCHITECTURES}")
    add_definitions(-DSCI_ENABLE_CUDA=1 -DSCI_USE_CUDA=1)
endif()

if(SCI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

find_package(Threads REQUIRED)

# CUDA libraries：由 FindCUDAToolkit 解析（CUDA::cudart / CUDA::cublas / CUDA::cusparse），
# 可用 -DCUDAToolkit_ROOT=/path/to/cuda 指定非默认安装位置
if(SCI_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

add_subdirectory(src)
add_subdirectory(examples)

if(SCI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SCI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# 打印配置信息
message(STATUS "SciComputeInfra GPU Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA: ${SCI_ENABLE_CUDA}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  OpenMP: ${SCI_ENABLE_OPENMP}")
message(STATUS "  Tests: ${SCI_BUILD_TESTS}")
message(STATUS "  Benchmarks: ${SCI_BUILD_BENCHMARKS}")
