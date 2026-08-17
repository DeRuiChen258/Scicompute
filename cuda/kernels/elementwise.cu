#include "elementwise.cuh"
#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Elementwise CUDA Kernel Implementations
// ============================================================================

// Add kernel
template<typename T>
__global__ void add_kernel(T* out, const T* a, const T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

template<typename T>
void launch_add(T* out, const T* a, const T* b, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, b, n);
}

// Mul kernel
template<typename T>
__global__ void mul_kernel(T* out, const T* a, const T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

template<typename T>
void launch_mul(T* out, const T* a, const T* b, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    mul_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, b, n);
}

// Sub kernel
template<typename T>
__global__ void sub_kernel(T* out, const T* a, const T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] - b[i];
    }
}

template<typename T>
void launch_sub(T* out, const T* a, const T* b, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    sub_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, b, n);
}

// Div kernel
template<typename T>
__global__ void div_kernel(T* out, const T* a, const T* b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] / b[i];
    }
}

template<typename T>
void launch_div(T* out, const T* a, const T* b, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    div_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, b, n);
}

// Scalar kernels
template<typename T>
__global__ void add_scalar_kernel(T* out, const T* a, T scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] + scalar;
    }
}

template<typename T>
void launch_add_scalar(T* out, const T* a, T scalar, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_scalar_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, scalar, n);
}

template<typename T>
__global__ void mul_scalar_kernel(T* out, const T* a, T scalar, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = a[i] * scalar;
    }
}

template<typename T>
void launch_mul_scalar(T* out, const T* a, T scalar, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    mul_scalar_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, scalar, n);
}

// Sqrt kernel
template<typename T>
__global__ void sqrt_kernel(T* out, const T* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = sqrtf(a[i]);
    }
}

template<typename T>
void launch_sqrt(T* out, const T* a, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    sqrt_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, n);
}

// Rsqrt kernel
template<typename T>
__global__ void rsqrt_kernel(T* out, const T* a, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < n; i += stride) {
        out[i] = rsqrtf(a[i]);
    }
}

template<typename T>
void launch_rsqrt(T* out, const T* a, size_t n, cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    rsqrt_kernel<T><<<grid_size, block_size, 0, stream>>>(out, a, n);
}

// Explicit template instantiations
template void launch_add<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void launch_mul<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void launch_sub<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void launch_div<float>(float*, const float*, const float*, size_t, cudaStream_t);
template void launch_add_scalar<float>(float*, const float*, float, size_t, cudaStream_t);
template void launch_mul_scalar<float>(float*, const float*, float, size_t, cudaStream_t);
template void launch_sqrt<float>(float*, const float*, size_t, cudaStream_t);
template void launch_rsqrt<float>(float*, const float*, size_t, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace sci
