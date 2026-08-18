cmake_minimum_required(VERSION 3.20)

set(BRIDGES_SOURCES
    pool_bridge.cpp
    thread_pool_bridge.cpp
    ipc_bridge.cpp
)

if(SCI_ENABLE_CUDA)
    list(APPEND BRIDGES_SOURCES 
        cuda_kernel_bridge.cu
        ${CMAKE_SOURCE_DIR}/cuda/kernels/layernorm.cu
        ${CMAKE_SOURCE_DIR}/cuda/kernels/softmax.cu
    )
endif()

add_library(sci_bridges OBJECT ${BRIDGES_SOURCES})

target_include_directories(sci_bridges PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bridges
    ${CMAKE_SOURCE_DIR}/cuda/kernels
)

if(SCI_ENABLE_CUDA)
    target_include_directories(sci_bridges PUBLIC /usr/local/cuda-13.2/include)
    target_compile_definitions(sci_bridges PUBLIC SCI_USE_CUDA=1)
    target_link_libraries(sci_bridges PUBLIC CUDA::cudart)
endif()

set_target_properties(sci_bridges PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(sci_bridges PUBLIC
    sci_core
    sci_memory
    sci_tensor
    sci_scheduler
)
