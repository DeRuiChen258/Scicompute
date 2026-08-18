cmake_minimum_required(VERSION 3.20)

# 核心子目录
add_subdirectory(core)
add_subdirectory(device)
add_subdirectory(memory)
add_subdirectory(tensor)
add_subdirectory(scheduler)
add_subdirectory(math)
add_subdirectory(benchmark)

# 桥接层 - 连接旧项目
add_subdirectory(bridges)

# GPU 支持 (CUDA 13.2 RTX)
if(SCI_ENABLE_CUDA)
    add_subdirectory(gpu)
endif()

# 主库
add_library(sci_compute INTERFACE)
target_include_directories(sci_compute INTERFACE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gpu
)
target_link_libraries(sci_compute INTERFACE
    Threads::Threads
    sci_core
    sci_device
    sci_memory
    sci_tensor
    sci_scheduler
    sci_math
    sci_benchmark
    sci_bridges
)

if(SCI_ENABLE_OPENMP)
    target_link_libraries(sci_compute INTERFACE OpenMP::OpenMP_CXX)
endif()

if(SCI_ENABLE_CUDA)
    target_link_libraries(sci_compute INTERFACE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        sci_gpu
    )
    target_compile_definitions(sci_compute INTERFACE
        SCI_USE_CUDA=1
        SCI_ENABLE_CUDA=1
    )
endif()
