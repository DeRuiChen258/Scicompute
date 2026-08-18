#pragma once
/**
 * @file cuda_kernel_bridge.hpp
 * @brief CUDA Kernel 桥接层
 */

#include "../core/common.hpp"

// CUDA类型定义
#ifdef SCI_USE_CUDA
#include <cuda_runtime.h>
namespace sci {
namespace bridges {
    using cudaStream_t = ::cudaStream_t;
}
}
#else
namespace sci {
namespace bridges {
    using cudaStream_t = void*;
}
}
#endif

namespace sci {
namespace bridges {

// ============================================================================
// CUDA Kernel 命名空间
// ============================================================================
namespace cuda_kernels {

#ifdef SCI_USE_CUDA
void rms_norm(float* out, const float* in, const float* weight, int rows, int cols, float eps, cudaStream_t stream);
void layer_norm(float* out, const float* in, const float* weight, const float* bias,
                int rows, int cols, float eps, cudaStream_t stream);
void softmax(float* out, const float* x, int n, int dim, cudaStream_t stream);
void softmax_stable(float* out, const float* x, int n, int dim, float scale, cudaStream_t stream);
void add(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void mul(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void sub(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void div(float* out, const float* a, const float* b, int n, cudaStream_t stream);
void reduce_sum(float* out, const float* in, int n, cudaStream_t stream);
void reduce_max(float* out, const float* in, int n, cudaStream_t stream);
#else
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
#endif

} // namespace cuda_kernels

// ============================================================================
// CUDA 辅助函数
// ============================================================================

#ifdef SCI_USE_CUDA
bool check_cuda_error(cudaError_t err, const char* file, int line);
cudaStream_t get_cuda_stream();
void cuda_stream_synchronize(cudaStream_t stream);
#else
inline bool check_cuda_error(...) { return false; }
inline cudaStream_t get_cuda_stream() { return nullptr; }
inline void cuda_stream_synchronize(...) {}
#endif

} // namespace bridges
} // namespace sci

#ifdef SCI_USE_CUDA
#define CUDA_CHECK(call) \
    sci::bridges::check_cuda_error(call, __FILE__, __LINE__)
#else
#define CUDA_CHECK(call) ((void)0)
#endif
