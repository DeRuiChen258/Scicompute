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
    # RTX 13.2 GPU support (compute capability 8.9, 9.0)
    set(CMAKE_CUDA_ARCHITECTURES 75;89;90;120)
    add_definitions(-DSCI_ENABLE_CUDA=1 -DSCI_USE_CUDA=1)
endif()

if(SCI_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

find_package(Threads REQUIRED)

# CUDA libraries
if(SCI_ENABLE_CUDA)
    set(CUDA_INCLUDE_DIRS /usr/local/cuda-13.2/include)
    set(CUDA_LIBRARY_DIRS /usr/local/cuda-13.2/lib64)
    
    add_library(CUDA::cudart SHARED IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "${CUDA_LIBRARY_DIRS}/libcudart.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}"
    )
    
    add_library(CUDA::cublas SHARED IMPORTED)
    set_target_properties(CUDA::cublas PROPERTIES
        IMPORTED_LOCATION "${CUDA_LIBRARY_DIRS}/libcublas.so"
        INTERFACE_LINK_LIBRARIES "CUDA::cudart"
    )
    
    add_library(CUDA::cusparse SHARED IMPORTED)
    set_target_properties(CUDA::cusparse PROPERTIES
        IMPORTED_LOCATION "${CUDA_LIBRARY_DIRS}/libcusparse.so"
        INTERFACE_LINK_LIBRARIES "CUDA::cudart"
    )
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
