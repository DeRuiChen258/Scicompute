#pragma once

#include <cuda_runtime.h>

namespace sci {
namespace cuda {
namespace kernels {

// ============================================================================
// Attention CUDA Kernels
// ============================================================================

// Scaled Dot-Product Attention
// Q: [batch, heads, seq_len, head_dim]
// K: [batch, heads, seq_len, head_dim]
// V: [batch, heads, seq_len, head_dim]
// Output: [batch, heads, seq_len, head_dim]
template<typename T>
__global__ void scaled_dot_product_attention_kernel(
    T* out, const T* q, const T* k, const T* v,
    int batch, int heads, int seq_len, int head_dim);

template<typename T>
void launch_scaled_dot_product_attention(
    T* out, const T* q, const T* k, const T* v,
    int batch, int heads, int seq_len, int head_dim,
    cudaStream_t stream);

// Flash Attention v1 (simplified)
template<typename T>
void launch_flash_attention(
    T* out, const T* q, const T* k, const T* v,
    int batch, int heads, int seq_len, int head_dim,
    T scale, cudaStream_t stream);

// Masked attention (for autoregressive decoding)
template<typename T>
__global__ void masked_attention_kernel(
    T* out, const T* q, const T* k, const T* v,
    const bool* mask, int batch, int heads, int seq_len, int head_dim, T scale);

template<typename T>
void launch_masked_attention(
    T* out, const T* q, const T* k, const T* v, const bool* mask,
    int batch, int heads, int seq_len, int head_dim,
    T scale, cudaStream_t stream);

// ============================================================================
// Rotary Embedding (RoPE)
// ============================================================================
template<typename T>
__global__ void rotary_embedding_kernel(
    T* out, const T* in, const T* cos, const T* sin,
    int batch, int seq_len, int heads, int head_dim);

template<typename T>
void launch_rotary_embedding(
    T* out, const T* in, const T* cos, const T* sin,
    int batch, int seq_len, int heads, int head_dim,
    cudaStream_t stream);

// Rotary embedding inverse (for attention score rotation back)
template<typename T>
__global__ void rotary_embedding_inv_kernel(
    T* out, const T* in, const T* cos, const T* sin,
    int batch, int seq_len, int heads, int head_dim);

template<typename T>
void launch_rotary_embedding_inv(
    T* out, const T* in, const T* cos, const T* sin,
    int batch, int seq_len, int heads, int head_dim,
    cudaStream_t stream);

} // namespace kernels
} // namespace cuda
} // namespace sci
