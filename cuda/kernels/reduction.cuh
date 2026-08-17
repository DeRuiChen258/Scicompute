#pragma once

#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Reduction CUDA Kernels
// ============================================================================

// Sum reduction
template<typename T, typename OutT = T>
__global__ void sum_kernel(const T* in, OutT* out, size_t n);

template<typename T, typename OutT = T>
void launch_sum(const T* in, OutT* out, size_t n, cudaStream_t stream);

// Max reduction
template<typename T>
__global__ void max_kernel(const T* in, T* out, size_t n);

template<typename T>
void launch_max(const T* in, T* out, size_t n, cudaStream_t stream);

// Min reduction
template<typename T>
__global__ void min_kernel(const T* in, T* out, size_t n);

template<typename T>
void launch_min(const T* in, T* out, size_t n, cudaStream_t stream);

// ArgMax: returns index of maximum value
template<typename T>
__global__ void argmax_kernel(const T* in, T* max_val, int* max_idx, size_t n);

template<typename T>
void launch_argmax(const T* in, T* max_val, int* max_idx, size_t n, cudaStream_t stream);

// Block-wise reduction with shared memory
template<typename T, int kBlockSize>
__device__ T block_reduce_sum(T val, T* shared);

template<typename T, int kBlockSize>
__device__ T block_reduce_max(T val, T* shared);

// Warp-level reduction utilities
template<typename T>
__device__ T warp_reduce_sum(T val);

template<typename T>
__device__ T warp_reduce_max(T val);

// Row-wise reduction (reduce along axis 1 for 2D tensor)
template<typename T>
__global__ void row_sum_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_row_sum(const T* in, T* out, int rows, int cols, cudaStream_t stream);

template<typename T>
__global__ void row_max_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_row_max(const T* in, T* out, int rows, int cols, cudaStream_t stream);

// L2 norm along rows
template<typename T>
__global__ void row_l2_norm_kernel(const T* in, T* out, int rows, int cols);

template<typename T>
void launch_row_l2_norm(const T* in, T* out, int rows, int cols, cudaStream_t stream);

} // namespace kernels
} // namespace cuda
} // namespace sci
