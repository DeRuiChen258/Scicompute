#pragma once

#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// LayerNorm CUDA Kernels
// ============================================================================

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
template<typename T>
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight,
                                 int rows, int cols, T eps);

template<typename T>
void launch_rms_norm(T* out, const T* in, const T* weight,
                     int rows, int cols, T eps, cudaStream_t stream);

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
template<typename T>
__global__ void layer_norm_kernel(T* out, const T* in, const T* weight,
                                   const T* bias, int rows, int cols, T eps);

template<typename T>
void launch_layer_norm(T* out, const T* in, const T* weight, const T* bias,
                       int rows, int cols, T eps, cudaStream_t stream);

// Fused LayerNorm with add (for residual connections)
// out = LayerNorm(x + residual) where residual is stored in a separate buffer
template<typename T>
__global__ void layer_norm_add_kernel(T* out, const T* x, const T* residual,
                                      const T* weight, const T* bias,
                                      int rows, int cols, T eps);

template<typename T>
void launch_layer_norm_add(T* out, const T* x, const T* residual,
                           const T* weight, const T* bias,
                           int rows, int cols, T eps, cudaStream_t stream);

// ============================================================================
// Warp-level reduction for LayerNorm
// ============================================================================
template<typename T>
__device__ T warp_reduce_sq_sum(T val);

template<typename T>
__device__ T warp_reduce_sum(T val);

} // namespace kernels
} // namespace cuda
} // namespace sci
