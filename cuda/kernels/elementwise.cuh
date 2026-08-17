#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Elementwise CUDA Kernels
// ============================================================================

// Add: out = a + b
template<typename T>
__global__ void add_kernel(T* out, const T* a, const T* b, size_t n);

template<typename T>
void launch_add(T* out, const T* a, const T* b, size_t n, cudaStream_t stream);

// Sub: out = a - b
template<typename T>
__global__ void sub_kernel(T* out, const T* a, const T* b, size_t n);

template<typename T>
void launch_sub(T* out, const T* a, const T* b, size_t n, cudaStream_t stream);

// Mul: out = a * b
template<typename T>
__global__ void mul_kernel(T* out, const T* a, const T* b, size_t n);

template<typename T>
void launch_mul(T* out, const T* a, const T* b, size_t n, cudaStream_t stream);

// Div: out = a / b
template<typename T>
__global__ void div_kernel(T* out, const T* a, const T* b, size_t n);

template<typename T>
void launch_div(T* out, const T* a, const T* b, size_t n, cudaStream_t stream);

// Add scalar: out = a + scalar
template<typename T>
__global__ void add_scalar_kernel(T* out, const T* a, T scalar, size_t n);

template<typename T>
void launch_add_scalar(T* out, const T* a, T scalar, size_t n, cudaStream_t stream);

// Mul scalar: out = a * scalar
template<typename T>
__global__ void mul_scalar_kernel(T* out, const T* a, T scalar, size_t n);

template<typename T>
void launch_mul_scalar(T* out, const T* a, T scalar, size_t n, cudaStream_t stream);

// Sqrt: out = sqrt(a)
template<typename T>
__global__ void sqrt_kernel(T* out, const T* a, size_t n);

template<typename T>
void launch_sqrt(T* out, const T* a, size_t n, cudaStream_t stream);

// Rsqrt: out = rsqrt(a) = 1/sqrt(a)
template<typename T>
__global__ void rsqrt_kernel(T* out, const T* a, size_t n);

template<typename T>
void launch_rsqrt(T* out, const T* a, size_t n, cudaStream_t stream);

// ============================================================================
// Specializations for float16 (FP16)
// ============================================================================
#ifdef CUDA_AVAILABLE
void launch_add_fp16(__half* out, const __half* a, const __half* b, size_t n, cudaStream_t stream);
void launch_mul_fp16(__half* out, const __half* a, const __half* b, size_t n, cudaStream_t stream);
#endif

// ============================================================================
// Fused kernels
// ============================================================================

// Fused add + mul: out = (a + b) * c
template<typename T>
__global__ void fused_add_mul_kernel(T* out, const T* a, const T* b, const T* c, size_t n);

template<typename T>
void launch_fused_add_mul(T* out, const T* a, const T* b, const T* c, size_t n, cudaStream_t stream);

// Fused multiply + add: out = a * b + c
template<typename T>
__global__ void fused_mul_add_kernel(T* out, const T* a, const T* b, const T* c, size_t n);

template<typename T>
void launch_fused_mul_add(T* out, const T* a, const T* b, const T* c, size_t n, cudaStream_t stream);

} // namespace kernels
} // namespace cuda
} // namespace sci
