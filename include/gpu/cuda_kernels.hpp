#pragma once
/**
 * @file cuda_kernels.hpp
 * @brief CUDA Kernel 桥接层 - RTX 13.2 GPU计算核心
 * @author SciComputeInfra
 * @date 2026-08-17
 * 
 * 功能:
 * - GEMM矩阵乘法 (cuBLAS集成)
 * - Softmax计算
 * - Layer Norm
 * - Attention计算
 * - Reshape和Transpose
 */

#include "gpu_memory.hpp"
#include "../tensor/tensor.hpp"

#ifdef SCI_USE_CUDA
#include <cuda_fp16.h>
#include <cublas_v2.h>
#endif

namespace sci {
namespace gpu {
namespace kernels {

// ============================================================================
// 基础CUDA Kernel配置
// ============================================================================
struct KernelConfig {
    dim3 block_dim{256, 1, 1};
    dim3 grid_dim{1, 1, 1};
    size_t shared_mem_bytes{0};
    cudaStream_t stream{nullptr};
};

// ============================================================================
// GEMM矩阵乘法
// ============================================================================
/**
 * @brief CUDA GEMM - 矩阵乘法
 * @param alpha 标量系数
 * @param A 左矩阵 [M, K]
 * @param B 右矩阵 [K, N]
 * @param beta 偏置系数
 * @param C 输出矩阵 [M, N]
 * @param M 行数
 * @param N 列数
 * @param K 内部维度
 * @param trans_a 转置A
 * @param trans_b 转置B
 */
void gemm(float alpha, const void* A, const void* B, float beta, void* C,
          int M, int N, int K, bool trans_a = false, bool trans_b = false);

/**
 * @brief 半精度GEMM (FP16)
 */
void gemm_fp16(__half alpha, const __half* A, const __half* B, 
               __half beta, __half* C, int M, int N, int K,
               bool trans_a = false, bool trans_b = false);

// ============================================================================
// Softmax
// ============================================================================
/**
 * @brief Softmax计算
 * @param input 输入张量 [*, seq_len, seq_len]
 * @param output 输出张量
 * @param dim 计算维度
 */
void softmax(const tensor::Tensor& input, tensor::Tensor& output, int dim = -1);

/**
 * @brief Softmax Mask (带mask的softmax)
 */
void softmax_mask(const tensor::Tensor& input, const tensor::Tensor& mask, 
                  tensor::Tensor& output, float mask_value = -1e9f);

// ============================================================================
// Layer Normalization
// ============================================================================
/**
 * @brief Layer Normalization
 * @param input 输入 [batch, seq_len, hidden]
 * @param output 输出
 * @param weight 权重 [hidden]
 * @param bias 偏置 [hidden]
 * @param eps epsilon
 */
void layer_norm(const tensor::Tensor& input, tensor::Tensor& output,
                const tensor::Tensor& weight, const tensor::Tensor& bias,
                float eps = 1e-5f);

/**
 * @brief RMS Layer Normalization
 */
void rms_norm(const tensor::Tensor& input, tensor::Tensor& output,
              const tensor::Tensor& weight, float eps = 1e-5f);

// ============================================================================
// Attention
// ============================================================================
/**
 * @brief Scaled Dot-Product Attention
 * @param Q Query [batch, heads, seq_len, head_dim]
 * @param K Key [batch, heads, seq_len, head_dim]
 * @param V Value [batch, heads, seq_len, head_dim]
 * @param mask Attention mask
 * @param output 输出 [batch, heads, seq_len, head_dim]
 * @param scale 缩放因子 (通常为1/sqrt(head_dim))
 */
void scaled_dot_product_attention(
    const tensor::Tensor& Q, const tensor::Tensor& K, const tensor::Tensor& V,
    const tensor::Tensor& mask, tensor::Tensor& output, float scale);

// ============================================================================
// 激活函数
// ============================================================================
void relu(tensor::Tensor& tensor);
void gelu(tensor::Tensor& tensor);
void sigmoid(tensor::Tensor& tensor);
void tanh(tensor::Tensor& tensor);

// ============================================================================
// 元素级操作
// ============================================================================
void add(const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& output);
void multiply(const tensor::Tensor& a, const tensor::Tensor& b, tensor::Tensor& output);
void scale(tensor::Tensor& tensor, float scalar);

// ============================================================================
// Memory Operations
// ============================================================================
void memset_zero(void* ptr, size_t bytes);
void memcpy_async(void* dst, const void* src, size_t bytes);
void synchronize();

} // namespace kernels
} // namespace gpu
} // namespace sci
