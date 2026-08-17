cmake_minimum_required(VERSION 3.20)
project(SciComputeInfra VERSION 0.1.0 LANGUAGES CXX)

option(SCI_BUILD_TESTS "Build tests" OFF)
option(SCI_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(SCI_BUILD_PYTHON_BINDINGS "Build Python bindings" OFF)
option(SCI_ENABLE_CUDA "Enable CUDA support" OFF)
option(SCI_ENABLE_OPENMP "Enable OpenMP for CPU parallelism" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter")
endif()

if(SCI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

find_package(Threads REQUIRED)

add_subdirectory(src)
add_subdirectory(examples)

if(SCI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SCI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
