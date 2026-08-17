#pragma once

#include "../tensor/tensor.hpp"

namespace sci {
namespace math {

// ============================================================================
// Normalization Operations
// ============================================================================

// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
Result<Tensor> layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          float eps = 1e-5f, Stream* stream = nullptr);

// LayerNorm with given normalized_shape
Result<Tensor> layer_norm(const Tensor& x, const std::vector<int>& normalized_shape,
                          float eps = 1e-5f, Stream* stream = nullptr);

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
Result<Tensor> rms_norm(const Tensor& x, const Tensor& weight,
                        float eps = 1e-5f, Stream* stream = nullptr);

// RMSNorm with given normalized_shape
Result<Tensor> rms_norm(const Tensor& x, const std::vector<int>& normalized_shape,
                        float eps = 1e-5f, Stream* stream = nullptr);

// BatchNorm1d (simplified for feature dimension)
Result<Tensor> batch_norm(const Tensor& x, const Tensor& running_mean,
                          const Tensor& running_var, const Tensor& weight,
                          const Tensor& bias, float eps = 1e-5f, 
                          float momentum = 0.1f, Stream* stream = nullptr);

// Reference implementations
namespace ref {
void layer_norm_f32(const float* x, const float* weight, const float* bias,
                    float* out, int rows, int cols, float eps);
void rms_norm_f32(const float* x, const float* weight,
                  float* out, int rows, int cols, float eps);
} // namespace ref

} // namespace math
} // namespace sci
