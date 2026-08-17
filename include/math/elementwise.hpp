#pragma once

#include "../tensor/tensor.hpp"
#include "../device/stream.hpp"

namespace sci {
namespace math {

// ============================================================================
// Elementwise Operations - CPU implementations
// ============================================================================

// Add: out = a + b
Result<Tensor> add(const Tensor& a, const Tensor& b, Stream* stream = nullptr);
Result<Tensor> add(const Tensor& a, const Tensor& b, const Tensor& out, Stream* stream = nullptr);

// In-place add: a += b
void add_inplace(Tensor& a, const Tensor& b, Stream* stream = nullptr);

// Add scalar: out = a + scalar
Result<Tensor> add_scalar(const Tensor& a, float scalar, Stream* stream = nullptr);

// Subtract: out = a - b
Result<Tensor> sub(const Tensor& a, const Tensor& b, Stream* stream = nullptr);
Result<Tensor> sub(const Tensor& a, const Tensor& b, const Tensor& out, Stream* stream = nullptr);

// Multiply: out = a * b
Result<Tensor> mul(const Tensor& a, const Tensor& b, Stream* stream = nullptr);
Result<Tensor> mul(const Tensor& a, const Tensor& b, const Tensor& out, Stream* stream = nullptr);

// In-place multiply: a *= scalar
void mul_scalar_inplace(Tensor& a, float scalar, Stream* stream = nullptr);

// Divide: out = a / b
Result<Tensor> div(const Tensor& a, const Tensor& b, Stream* stream = nullptr);
Result<Tensor> div(const Tensor& a, const Tensor& b, const Tensor& out, Stream* stream = nullptr);

// Negate: out = -a
Result<Tensor> negate(const Tensor& a, Stream* stream = nullptr);

// Abs: out = abs(a)
Result<Tensor> abs(const Tensor& a, Stream* stream = nullptr);

// Reciprocal: out = 1 / a
Result<Tensor> reciprocal(const Tensor& a, Stream* stream = nullptr);

// Square: out = a * a
Result<Tensor> square(const Tensor& a, Stream* stream = nullptr);

// Sqrt: out = sqrt(a)
Result<Tensor> sqrt(const Tensor& a, Stream* stream = nullptr);

// Rsqrt: out = 1 / sqrt(a)
Result<Tensor> rsqrt(const Tensor& a, Stream* stream = nullptr);

// Pow: out = a ^ b
Result<Tensor> pow(const Tensor& a, float exponent, Stream* stream = nullptr);

// Clamp: out = clamp(a, min, max)
Result<Tensor> clamp(const Tensor& a, float min_val, float max_val, Stream* stream = nullptr);

// ============================================================================
// Reference implementations (for testing)
// ============================================================================
namespace ref {
void add_f32(const float* a, const float* b, float* out, size_t n);
void mul_f32(const float* a, const float* b, float* out, size_t n);
void sub_f32(const float* a, const float* b, float* out, size_t n);
void div_f32(const float* a, const float* b, float* out, size_t n);
} // namespace ref

} // namespace math
} // namespace sci
