#pragma once

#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Softmax CUDA Kernels
// ============================================================================

// Row-wise softmax
template<typename T>
__global__ void softmax_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream);

// Row-wise log softmax
template<typename T>
__global__ void log_softmax_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_log_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream);

// Stable softmax (subtract max for numerical stability)
template<typename T>
__global__ void softmax_stable_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_softmax_stable(const T* in, T* out, int rows, int cols, cudaStream_t stream);

// Softmax with temperature
template<typename T>
__global__ void softmax_temperature_kernel(const T* in, T* out, int rows, int cols, T temperature);

template<typename T>
void launch_softmax_temperature(const T* in, T* out, int rows, int cols, 
                                T temperature, cudaStream_t stream);

// ============================================================================
// Warp-level softmax utilities
// ============================================================================
template<typename T>
__device__ T warp_softmax_sum(T val, int cols);

} // namespace kernels
} // namespace cuda
} // namespace sci
