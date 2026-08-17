#pragma once

#include "../tensor/tensor.hpp"

namespace sci {
namespace math {

// ============================================================================
// Softmax Operations
// ============================================================================

// Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
Result<Tensor> softmax(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// Log Softmax: log_softmax(x)[i] = x[i] - log(sum(exp(x[j])))
Result<Tensor> log_softmax(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// Softmax with temperature: softmax(x / T)
Result<Tensor> softmax_with_temperature(const Tensor& a, float temperature, 
                                         int axis = -1, Stream* stream = nullptr);

// Hardmax: returns one-hot vector with 1 at max position
Result<Tensor> hardmax(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// Stable softmax (subtract max for numerical stability)
Result<Tensor> softmax_stable(const Tensor& a, int axis = -1, Stream* stream = nullptr);

// Reference implementations
namespace ref {
void softmax_f32(const float* x, float* out, size_t n, size_t stride);
void softmax_axis_f32(const float* x, float* out, const int* shape, int ndim, int axis);
void log_softmax_f32(const float* x, float* out, size_t n, size_t stride);
} // namespace ref

} // namespace math
} // namespace sci
