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
__device__ T warp_softmax_sum(T val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warp_reduce_max(T val) {
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Row-wise softmax kernel
template<typename T>
__global__ void softmax_kernel(const T* in, T* out, int rows, int cols) {
    extern __shared__ char shared_mem[];
    T* shared_max = reinterpret_cast<T*>(shared_mem);
    T* shared_sum = reinterpret_cast<T*>(shared_mem + (blockDim.x / 32) * sizeof(T));
    
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Find max for numerical stability
    T max_val = -HUGE_VALF;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, in[row * cols + i]);
    }
    
    // Warp reduction for max
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    // Block-level reduction
    if (tid == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < (blockDim.x + 32 - 1) / 32; ++i) {
            max_val = max(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute exp and sum
    T exp_sum = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        T exp_val = expf(in[row * cols + i] - max_val);
        out[row * cols + i] = exp_val;
        exp_sum += exp_val;
    }
    
    // Warp reduction for sum
    exp_sum = warp_softmax_sum(exp_sum);
    if (lane_id == 0) {
        shared_sum[warp_id] = exp_sum;
    }
    __syncthreads();
    
    // Block-level reduction
    if (tid == 0) {
        exp_sum = shared_sum[0];
        for (int i = 1; i < (blockDim.x + 32 - 1) / 32; ++i) {
            exp_sum += shared_sum[i];
        }
        shared_sum[0] = exp_sum;
    }
    __syncthreads();
    exp_sum = shared_sum[0];
    
    // Normalize
    T inv_sum = static_cast<T>(1) / (exp_sum  + static_cast<T>(1e-10));
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] *= inv_sum;
    }
}

template<typename T>
void launch_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    int block_size = 256;
    int shared_bytes = (block_size / 32 + 1) * sizeof(T);
    softmax_kernel<T><<<rows, block_size, shared_bytes, stream>>>(in, out, rows, cols);
}

// Stable softmax (explicit max subtraction)
template<typename T>
__global__ void softmax_stable_kernel(const T* in, T* out, int rows, int cols) {
    softmax_kernel<T><<<rows, 256, 0, 0>>>(in, out, rows, cols);
}

template<typename T>
void launch_softmax_stable(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    launch_softmax(in, out, rows, cols, stream);
}

// Log softmax kernel
template<typename T>
__global__ void log_softmax_kernel(const T* in, T* out, int rows, int cols) {
    extern __shared__ char shared_mem[];
    T* shared_max = reinterpret_cast<T*>(shared_mem);
    T* shared_sum = reinterpret_cast<T*>(shared_mem + (blockDim.x / 32) * sizeof(T));
    
    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Find max
    T max_val = -HUGE_VALF;
    for (int i = tid; i < cols; i += blockDim.x) {
        max_val = max(max_val, in[row * cols + i]);
    }
    
    max_val = warp_reduce_max(max_val);
    if (lane_id == 0) {
        shared_max[warp_id] = max_val;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_val = shared_max[0];
        for (int i = 1; i < (blockDim.x + 32 - 1) / 32; ++i) {
            max_val = max(max_val, shared_max[i]);
        }
        shared_max[0] = max_val;
    }
    __syncthreads();
    max_val = shared_max[0];
    
    // Compute sum
    T sum = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += expf(in[row * cols + i] - max_val);
    }
    
    sum = warp_softmax_sum(sum);
    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (tid == 0) {
        sum = shared_sum[0];
        for (int i = 1; i < (blockDim.x + 32 - 1) / 32; ++i) {
            sum += shared_sum[i];
        }
        shared_sum[0] = sum;
    }
    __syncthreads();
    sum = shared_sum[0];
    
    // Compute log_softmax
    T log_sum = logf(sum  + static_cast<T>(1e-10)) + max_val;
    for (int i = tid; i < cols; i += blockDim.x) {
        out[row * cols + i] = in[row * cols + i] - log_sum;
    }
}

template<typename T>
void launch_log_softmax(const T* in, T* out, int rows, int cols, cudaStream_t stream) {
    int block_size = 256;
    int shared_bytes = (block_size / 32 + 1) * sizeof(T);
    log_softmax_kernel<T><<<rows, block_size, shared_bytes, stream>>>(in, out, rows, cols);
}

// Explicit template instantiations
template void launch_softmax<float>(const float*, float*, int, int, cudaStream_t);
template void launch_log_softmax<float>(const float*, float*, int, int, cudaStream_t);
template void launch_softmax_stable<float>(const float*, float*, int, int, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace sci
