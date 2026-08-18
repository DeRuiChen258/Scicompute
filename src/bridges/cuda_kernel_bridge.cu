/**
 * @file cuda_kernel_bridge.cu
 * @brief CUDA Kernel 桥接层 - 调用 vLLM kernels
 */

#include "bridges/cuda_kernel_bridge.hpp"
#include <cuda_runtime.h>

// 包含 vLLM kernels
#include "layernorm.cuh"
#include "softmax.cuh"

namespace sci {
namespace bridges {

#ifdef SCI_USE_CUDA

// ============================================================================
// CUDA 辅助函数实现
// ============================================================================

bool check_cuda_error(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        return false;
    }
    return true;
}

cudaStream_t get_cuda_stream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        return nullptr;
    }
    return stream;
}

void cuda_stream_synchronize(cudaStream_t stream) {
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}

// ============================================================================
// CUDA Kernel 包装器
// ============================================================================
namespace cuda_kernels {

void rms_norm(float* out, const float* in, const float* weight,
             int rows, int cols, float eps, cudaStream_t stream) {
    sci::cuda::kernels::launch_rms_norm(out, in, weight, rows, cols, eps, stream);
}

void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                int rows, int cols, float eps, cudaStream_t stream) {
    sci::cuda::kernels::launch_layer_norm(out, in, weight, bias, rows, cols, eps, stream);
}

void softmax(float* out, const float* x, int rows, int cols, cudaStream_t stream) {
    sci::cuda::kernels::launch_softmax(x, out, rows, cols, stream);
}

void softmax_stable(float* out, const float* x, int rows, int cols, float scale, cudaStream_t stream) {
    // vLLM softmax_stable 内部处理数值稳定性，scale 参数暂时忽略
    sci::cuda::kernels::launch_softmax_stable(x, out, rows, cols, stream);
}

void add(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void mul(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void sub(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void div(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void reduce_sum(float* out, const float* in, int n, cudaStream_t stream);
void reduce_max(float* out, const float* in, int n, cudaStream_t stream);

} // namespace cuda_kernels

// ============================================================================
// Elementwise CUDA Kernels (定义)
// ============================================================================
__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void mul_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void sub_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void div_kernel(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && b[idx] != 0.0f) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void reduce_sum_kernel(float* out, const float* in, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float sum = 0.0f;
    if (idx < n) {
        sum = in[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && idx + s < n) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

namespace cuda_kernels {

void add(float* out, const float* a, const float* b, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
}

void mul(float* out, const float* a, const float* b, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    mul_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
}

void sub(float* out, const float* a, const float* b, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sub_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
}

void div(float* out, const float* a, const float* b, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    div_kernel<<<grid, block, 0, stream>>>(out, a, b, n);
}

void reduce_sum(float* out, const float* in, int n, cudaStream_t stream) {
    int block = 256;
    int grid = 1;
    reduce_sum_kernel<<<grid, block, block * sizeof(float), stream>>>(out, in, n);
    cudaStreamSynchronize(stream);
}

void reduce_max(float* out, const float* in, int n, cudaStream_t stream) {
    // 简化实现：单线程求最大值
    float result = -HUGE_VALF;
    for (int i = 0; i < n; ++i) {
        result = fmaxf(result, in[i]);
    }
    cudaMemcpyAsync(out, &result, sizeof(float), cudaMemcpyHostToDevice, stream);
}

} // namespace cuda_kernels

#else // SCI_USE_CUDA

namespace cuda_kernels {
    inline void rms_norm(...) {}
    inline void layer_norm(...) {}
    inline void softmax(...) {}
    inline void softmax_stable(...) {}
    inline void add(...) {}
    inline void mul(...) {}
    inline void sub(...) {}
    inline void div(...) {}
    inline void reduce_sum(...) {}
    inline void reduce_max(...) {}
} // namespace cuda_kernels

#endif // SCI_USE_CUDA

} // namespace bridges
} // namespace sci
