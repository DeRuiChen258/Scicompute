#include "layernorm.cuh"
#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// LayerNorm CUDA Kernel Implementations
// ============================================================================

constexpr int kWarpSize = 32;

template<typename T>
__device__ T warp_reduce_sq_sum(T val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// RMSNorm kernel
template<typename T>
__global__ void rms_norm_kernel(T* out, const T* in, const T* weight,
                                 int rows, int cols, T eps) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    T sum_sq = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        T val = in[row * cols + i];
        sum_sq += val * val;
    }

    // Warp-level reduction
    sum_sq = warp_reduce_sq_sum(sum_sq);

    if (tid % kWarpSize == 0) {
        sdata[tid / kWarpSize] = sum_sq;
    }
    __syncthreads();

    T rms;
    if (tid == 0) {
        T sum = static_cast<T>(0);
        int num_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
        for (int i = 0; i < num_warps; i++) {
            sum += sdata[i];
        }
        rms = rsqrtf(sum / static_cast<T>(cols) + eps);
        sdata[0] = rms;
    }
    __syncthreads();

    rms = sdata[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        T w = weight ? weight[i] : static_cast<T>(1);
        out[row * cols + i] = in[row * cols + i] * rms * w;
    }
}

template<typename T>
void launch_rms_norm(T* out, const T* in, const T* weight,
                     int rows, int cols, T eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int shared_bytes = ((kBlockSize + kWarpSize - 1) / kWarpSize) * sizeof(T);
    rms_norm_kernel<T><<<rows, kBlockSize, shared_bytes, stream>>>(out, in, weight, rows, cols, eps);
}

// LayerNorm kernel
template<typename T>
__global__ void layer_norm_kernel(T* out, const T* in, const T* weight,
                                   const T* bias, int rows, int cols, T eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ T shared[];
    T* s_mean = shared;
    T* s_var = shared + blockDim.x;

    T sum = static_cast<T>(0);
    T sum_sq = static_cast<T>(0);
    for (int i = tid; i < cols; i += blockDim.x) {
        T val = in[row * cols + i];
        sum += val;
        sum_sq += val * val;
    }
    s_mean[tid] = sum;
    s_var[tid] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        T total = static_cast<T>(0);
        T total_sq = static_cast<T>(0);
        for (int i = 0; i < blockDim.x; i++) {
            total += s_mean[i];
            total_sq += s_var[i];
        }
        T mean = total / static_cast<T>(cols);
        T var = total_sq / static_cast<T>(cols) - mean * mean;
        s_mean[0] = mean;
        s_var[0] = rsqrtf(var + eps);
    }
    __syncthreads();

    T mean = s_mean[0];
    T inv_std = s_var[0];
    for (int i = tid; i < cols; i += blockDim.x) {
        T normalized = (in[row * cols + i] - mean) * inv_std;
        T w = weight ? weight[i] : static_cast<T>(1);
        T b = bias ? bias[i] : static_cast<T>(0);
        out[row * cols + i] = normalized * w + b;
    }
}

template<typename T>
void launch_layer_norm(T* out, const T* in, const T* weight, const T* bias,
                       int rows, int cols, T eps, cudaStream_t stream) {
    constexpr int kBlockSize = 256;
    int shared_bytes = 2 * kBlockSize * sizeof(T);
    layer_norm_kernel<T><<<rows, kBlockSize, shared_bytes, stream>>>(out, in, weight, bias, rows, cols, eps);
}

// Explicit template instantiations
template void launch_rms_norm<float>(float*, const float*, const float*, int, int, float, cudaStream_t);
template void launch_layer_norm<float>(float*, const float*, const float*, const float*, int, int, float, cudaStream_t);

} // namespace kernels
} // namespace cuda
} // namespace sci
