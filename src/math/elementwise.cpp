#include "math/elementwise.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include <cmath>
#include <algorithm>

namespace sci {
namespace math {

// ============================================================================
// Reference Implementations
// ============================================================================
namespace ref {

void add_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] + b[i];
    }
}

void mul_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] * b[i];
    }
}

void sub_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] - b[i];
    }
}

void div_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = a[i] / b[i];
    }
}

} // namespace ref

// ============================================================================
// Elementwise Operations Implementation
// ============================================================================

Result<Tensor> add(const Tensor& a, const Tensor& b, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for add");
    SCI_ASSERT(a.dtype() == b.dtype(), "Dtype mismatch for add");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    ref::add_f32(a_data, b_data, out_data, n);
    
    return Ok(std::move(out));
}

Result<Tensor> add(const Tensor& a, const Tensor& b, const Tensor& out, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for add");
    SCI_ASSERT(a.is_same_shape(out), "Shape mismatch for add");
    
    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    ref::add_f32(a_data, b_data, out_data, n);
    
    return Ok(Tensor{});  // Returns empty since out is modified in-place
}

void add_inplace(Tensor& a, const Tensor& b, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for add_inplace");
    
    float* a_data = static_cast<float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        a_data[i] += b_data[i];
    }
}

Result<Tensor> add_scalar(const Tensor& a, float scalar, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = a_data[i] + scalar;
    }
    
    return Ok(std::move(out));
}

Result<Tensor> sub(const Tensor& a, const Tensor& b, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for sub");
    SCI_ASSERT(a.dtype() == b.dtype(), "Dtype mismatch for sub");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    ref::sub_f32(a_data, b_data, out_data, n);
    
    return Ok(std::move(out));
}

Result<Tensor> mul(const Tensor& a, const Tensor& b, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for mul");
    SCI_ASSERT(a.dtype() == b.dtype(), "Dtype mismatch for mul");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    ref::mul_f32(a_data, b_data, out_data, n);
    
    return Ok(std::move(out));
}

void mul_scalar_inplace(Tensor& a, float scalar, Stream* stream) {
    float* a_data = static_cast<float*>(a.data());
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        a_data[i] *= scalar;
    }
}

Result<Tensor> div(const Tensor& a, const Tensor& b, Stream* stream) {
    SCI_ASSERT(a.is_same_shape(b), "Shape mismatch for div");
    SCI_ASSERT(a.dtype() == b.dtype(), "Dtype mismatch for div");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    const float* b_data = static_cast<const float*>(b.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    ref::div_f32(a_data, b_data, out_data, n);
    
    return Ok(std::move(out));
}

Result<Tensor> negate(const Tensor& a, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = -a_data[i];
    }
    
    return Ok(std::move(out));
}

Result<Tensor> abs(const Tensor& a, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::abs(a_data[i]);
    }
    
    return Ok(std::move(out));
}

Result<Tensor> sqrt(const Tensor& a, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::sqrt(a_data[i]);
    }
    
    return Ok(std::move(out));
}

Result<Tensor> rsqrt(const Tensor& a, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = 1.0f / std::sqrt(a_data[i]);
    }
    
    return Ok(std::move(out));
}

Result<Tensor> pow(const Tensor& a, float exponent, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::pow(a_data[i], exponent);
    }
    
    return Ok(std::move(out));
}

Result<Tensor> clamp(const Tensor& a, float min_val, float max_val, Stream* stream) {
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* a_data = static_cast<const float*>(a.data());
    float* out_data = const_cast<float*>(static_cast<const float*>(out.data()));
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        out_data[i] = std::clamp(a_data[i], min_val, max_val);
    }
    
    return Ok(std::move(out));
}

} // namespace math
} // namespace sci
