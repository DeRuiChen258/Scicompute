#include "softmax.cuh"
#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Softmax CUDA Kernel Implementations
// ============================================================================

// Warp-level softmax utilities
template<typename T>
__device__ T warp_softmax_sum(T val, int cols) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Row-wise softmax kernel
template<typename T>
__global__ void softmax_kernel(const T* in, T* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    
    // Find max for numerical stability
    T max_val = -HUGE_VALF;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, in[row * cols + i]);
    }
    
    // Warp reduction for max
    max_val = warp_reduce_max(max_val);
    if (tid % warpSize == 0) {
        extern __shared__ T shared_max[];
        shared_max[tid / warpSize] = max_val;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; ++i) {
            max_val = max(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute exp sum
    T exp_sum = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        T exp_val = exp(in[row * cols + i] - max_val);
        out[row * cols + i] = exp_val;
        exp_sum += exp_val;
    }
    
    // Warp reduction for sum
    exp_sum = warp_softmax_sum(exp_sum, cols);
    if (tid % warpSize == 0) {
        extern __shared__ T shared_sum[];
        shared_sum[tid / warpSize] = exp_sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        exp_sum = shared_sum[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; ++i) {
            exp_sum += shared_sum[i];
        }
        shared_sum[0] = exp_sum;
    }
    __syncthreads();
    exp_sum = shared_sum[0];
    
    // Normalize
    T inv_sum = static_cast<T>(1) / exp_sum;
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] *= inv_sum;
    }
}

template<typename T>
void launch_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    int block_size = min(256, cols);
    int shared_bytes = ((block_size + 31) / 32) * 2 * sizeof(T);
    softmax_kernel<T><<<rows, block_size, shared_bytes, stream>>>(in, out, rows, cols);
}

// Stable softmax (explicit max subtraction)
template<typename T>
__global__ void softmax_stable_kernel(const T* in, T* out, int rows, int cols) {
    // Same as softmax_kernel since it's already stable
    launch_softmax(in, out, rows, cols, stream);
}

template<typename T>
void launch_softmax_stable(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    launch_softmax(in, out, rows, cols, stream);
}

// Log softmax kernel
template<typename T>
__global__ void log_softmax_kernel(const T* in, T* out, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    
    // Find max
    T max_val = -HUGE_VALF;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, in[row * cols + i]);
    }
    
    max_val = warp_reduce_max(max_val);
    if (tid % warpSize == 0) {
        extern __shared__ T shared_max[];
        shared_max[tid / warpSize] = max_val;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; ++i) {
            max_val = max(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute sum
    T sum = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += exp(in[row * cols + i] - max_val);
    }
    
    sum = warp_softmax_sum(sum, cols);
    if (tid % warpSize == 0) {
        extern __shared__ T shared_sum[];
        shared_sum[tid / warpSize] = sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        sum = shared_sum[0];
        for (int i = 1; i < (blockDim.x + warpSize - 1) / warpSize; ++i) {
            sum += shared_sum[i];
        }
        shared_sum[0] = sum;
    }
    __syncthreads();
    sum = shared_sum[0];
    
    // Compute log_softmax
    T log_sum = log(sum) + max_val;
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = in[row * cols + i] - log_sum;
    }
}

template<typename T>
void launch_log_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    int block_size = min(256, cols);
    int shared_bytes = ((block_size + 31) / 32) * 2 * sizeof(T);
    log_softmax_kernel<T><<<rows, block_size, shared_bytes, stream>>>(in, out, rows, cols);
}

// Explicit template instantiations
template void launch_softmax<float>(const float*, float*, int, int, cudaStream_t);
template void launch_log_softmax<float>(const float*, float*, int, int, cudaStream_t);
template void launch_softmax_stable<float>(const float*, float*, int, int, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace sci
