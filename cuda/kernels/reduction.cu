#include "reduction.cuh"
#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Reduction CUDA Kernel Implementations
// ============================================================================

// Warp-level reduction utilities
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction
template<typename T, int kBlockSize>
__device__ T block_reduce_sum(T val, T* shared) {
    int tid = threadIdx.x;
    
    // Store to shared memory
    shared[tid] = val;
    __syncthreads();
    
    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    return shared[0];
}

// Sum reduction kernel
template<typename T, typename OutT>
__global__ void sum_kernel(const T* in, OutT* out, size_t n) {
    extern __shared__ char shared_tmp[];
    T* shared = reinterpret_cast<T*>(shared_tmp);
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t stride = blockDim.x * 2 * gridDim.x;
    
    T sum = static_cast<T>(0);
    for (size_t i = idx; i < n; i += stride) {
        sum += in[i];
        if (i + blockDim.x < n) {
            sum += in[i + blockDim.x];
        }
    }
    
    __syncthreads();
    sum = block_reduce_sum<T, 256>(sum, shared);
    
    if (tid == 0) {
        out[blockIdx.x] = static_cast<OutT>(sum);
    }
}

template<typename T, typename OutT>
void launch_sum(const T* in, OutT* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = min((n + 511) / 512, 256);
    int shared_bytes = block_size * sizeof(T);
    
    sum_kernel<T, OutT><<<grid_size, block_size, shared_bytes, stream>>>(in, out, n);
}

// Max reduction kernel
template<typename T>
__global__ void max_kernel(const T* in, T* out, size_t n) {
    extern __shared__ char shared_tmp[];
    T* shared = reinterpret_cast<T*>(shared_tmp);
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t stride = blockDim.x * 2 * gridDim.x;
    
    T max_val = -HUGE_VALF;
    for (size_t i = idx; i < n; i += stride) {
        max_val = max(max_val, in[i]);
        if (i + blockDim.x < n) {
            max_val = max(max_val, in[i + blockDim.x]);
        }
    }
    
    max_val = warp_reduce_max(max_val);
    
    if (tid % warpSize == 0) {
        shared[tid / warpSize] = max_val;
    }
    __syncthreads();
    
    if (tid == 0) {
        max_val = shared[0];
        for (int i = 1; i < blockDim.x / warpSize; ++i) {
            max_val = max(max_val, shared[i]);
        }
        out[blockIdx.x] = max_val;
    }
}

template<typename T>
void launch_max(const T* in, T* out, size_t n, cudaStream_t stream) {
    int block_size = 256;
    int grid_size = min((n + 511) / 512, 256);
    int shared_bytes = (block_size / 32) * sizeof(T);
    
    max_kernel<T><<<grid_size, block_size, shared_bytes, stream>>>(in, out, n);
}

// Explicit template instantiations
template void launch_sum<float, float>(const float*, float*, size_t, cudaStream_t);
template void launch_max<float>(const float*, float*, size_t, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace sci
