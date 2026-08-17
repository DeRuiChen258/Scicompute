#pragma once

#include "../tensor/tensor.hpp"

namespace sci {
namespace math {

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum: reduce to scalar
Result<Tensor> sum(const Tensor& a, Stream* stream = nullptr);

// Sum along axis
Result<Tensor> sum(const Tensor& a, int axis, Stream* stream = nullptr);

// Mean: reduce to scalar
Result<Tensor> mean(const Tensor& a, Stream* stream = nullptr);

// Mean along axis
Result<Tensor> mean(const Tensor& a, int axis, Stream* stream = nullptr);

// Max: reduce to scalar
Result<Tensor> max(const Tensor& a, Stream* stream = nullptr);

// Max along axis (returns values and indices)
Result<std::pair<Tensor, Tensor>> max_with_indices(const Tensor& a, int axis, Stream* stream = nullptr);

// Min: reduce to scalar
Result<Tensor> min(const Tensor& a, Stream* stream = nullptr);

// ArgMax: index of maximum value
Result<Tensor> argmax(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// ArgMin: index of minimum value
Result<Tensor> argmin(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// Prod: product of all elements
Result<Tensor> prod(const Tensor& a, Stream* stream = nullptr);

// Std: standard deviation
Result<Tensor> std(const Tensor& a, Stream* stream = nullptr);

// Var: variance
Result<Tensor> var(const Tensor& a, Stream* stream = nullptr);

// Norm: Lp norm (default L2)
Result<Tensor> norm(const Tensor& a, float p = 2.0f, Stream* stream = nullptr);

// Reference implementations
namespace ref {
float sum_f32(const float* data, size_t n);
float max_f32(const float* data, size_t n);
float min_f32(const float* data, size_t n);
float mean_f32(const float* data, size_t n);
float var_f32(const float* data, size_t n);
} // namespace ref

} // namespace math
} // namespace sci
