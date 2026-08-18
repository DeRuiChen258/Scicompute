cmake_minimum_required(VERSION 3.20)

# GPU 源文件
set(GPU_SOURCES
    gpu_memory.cpp
    gpu_tensor.cpp
    gpu_scheduler.cpp
)

# GPU 库 - 使用 STATIC 模式, 使对象文件能通过 sci_compute 的
# INTERFACE 链接传递到所有消费者 (OBJECT 库不会传播到接口库)
add_library(sci_gpu STATIC ${GPU_SOURCES})

target_include_directories(sci_gpu PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/gpu
)

set_target_properties(sci_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# GPU编译选项
target_compile_options(sci_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -Wall -Wextra -Wpedantic
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        -gencode=arch=compute_89,code=sm_89
        -gencode=arch=compute_90,code=sm_90
        -Xcompiler=-fPIC
    >
)

target_link_libraries(sci_gpu PUBLIC
    sci_core
    sci_tensor
    sci_memory
)

if(SCI_ENABLE_CUDA)
    target_compile_definitions(sci_gpu PUBLIC
        SCI_USE_CUDA=1
        SCI_ENABLE_CUDA=1
    )
    
    target_link_libraries(sci_gpu PUBLIC
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
endif()
