# ============================================================================
# CUDA Module
# ============================================================================

if(NOT SCI_ENABLE_CUDA OR NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA support disabled or nvcc not found")
    return()
endif()

# CUDA kernels library
add_library(sci_cuda STATIC
    kernels/elementwise.cu
    kernels/reduction.cu
    kernels/softmax.cu
    kernels/layernorm.cu
)

# Set CUDA architecture
set_target_properties(sci_cuda PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE OFF
)

target_include_directories(sci_cuda PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/cuda
)

target_link_libraries(sci_cuda PUBLIC
    sci_core
    sci_tensor
    CUDA::cudart
)

# Enable relocatable device code for better compatibility
target_compile_options(sci_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Install CUDA headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/cuda/kernels/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sci/cuda/kernels
    FILES_MATCHING PATTERN "*.cuh"
)
