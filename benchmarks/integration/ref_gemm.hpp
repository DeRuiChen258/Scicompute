#pragma once
/**
 * @file ref_gemm.hpp
 * @brief 朴素 CPU 参考 GEMM (row-major), 供集成基准测试复用
 */

#include <cmath>
#include <cstddef>

namespace sci {
namespace bench {

// C[m, n] = A[m, k] * B[k, n] (均为 row-major)
inline void gemm_nt(const float* a, const float* b, float* c,
                    size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

// C[m, n] = A[m, k] * Bt^T, 其中 Bt 按 [rows=m, dim=k] row-major 存储
// (即 B 的转置视图, 用于 QK^T 时无需显式转置 K)
inline void gemm_qk(const float* a, const float* bt, float* c,
                    size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t p = 0; p < k; ++p) {
                acc += a[i * k + p] * bt[j * k + p];
            }
            c[i * n + j] = acc;
        }
    }
}

// 就地按行 softmax (数值稳定版)
inline void softmax_rows(float* x, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; ++r) {
        float* row = x + r * cols;
        float maxv = row[0];
        for (size_t c = 1; c < cols; ++c) {
            maxv = maxv > row[c] ? maxv : row[c];
        }

        float sum = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            row[c] = std::exp(row[c] - maxv);
            sum += row[c];
        }

        float inv = 1.0f / sum;
        for (size_t c = 0; c < cols; ++c) {
            row[c] *= inv;
        }
    }
}

} // namespace bench
} // namespace sci
