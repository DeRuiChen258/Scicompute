#include "math/softmax.hpp"
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

void softmax_f32(const float* x, float* out, size_t n, size_t stride) {
    // Find max for numerical stability
    float max_val = x[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i * stride]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        out[i * stride] = std::exp(x[i * stride] - max_val);
        sum += out[i * stride];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < n; ++i) {
        out[i * stride] *= inv_sum;
    }
}

void softmax_axis_f32(const float* x, float* out, const int* shape, int ndim, int axis) {
    // Simplified 2D softmax along axis
    if (ndim == 2 && axis == 1) {
        int rows = shape[0];
        int cols = shape[1];
        for (int r = 0; r < rows; ++r) {
            softmax_f32(x + r * cols, out + r * cols, cols, 1);
        }
    }
}

void log_softmax_f32(const float* x, float* out, size_t n, size_t stride) {
    float max_val = x[0];
    for (size_t i = 1; i < n; ++i) {
        max_val = std::max(max_val, x[i * stride]);
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += std::exp(x[i * stride] - max_val);
    }
    
    float log_sum = std::log(sum) + max_val;
    for (size_t i = 0; i < n; ++i) {
        out[i * stride] = x[i * stride] - log_sum;
    }
}

} // namespace ref

// ============================================================================
// Softmax Implementation
// ============================================================================

Result<Tensor> softmax(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(a.dtype() == DType::kFloat32, "Only float32 supported for now");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* in_data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());
    
    if (axis == -1 || axis == a.ndims() - 1) {
        // Last axis (row-wise for 2D)
        if (a.ndims() == 2) {
            int rows = static_cast<int>(a.dim(0));
            int cols = static_cast<int>(a.dim(1));
            for (int r = 0; r < rows; ++r) {
                ref::softmax_f32(in_data + r * cols, out_data + r * cols, cols, 1);
            }
        } else {
            // Flatten and apply
            ref::softmax_f32(in_data, out_data, a.num_elements(), 1);
        }
    } else {
        return MakeUnexpected<Tensor>(Status::NotImplemented("Only last-axis softmax supported"));
    }
    
    return Ok(std::move(out));
}

Result<Tensor> log_softmax(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(a.dtype() == DType::kFloat32, "Only float32 supported for now");
    
    Tensor out(a.shape(), a.dtype(), a.device());
    
    const float* in_data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());
    
    if (axis == -1 || axis == a.ndims() - 1) {
        if (a.ndims() == 2) {
            int rows = static_cast<int>(a.dim(0));
            int cols = static_cast<int>(a.dim(1));
            for (int r = 0; r < rows; ++r) {
                ref::log_softmax_f32(in_data + r * cols, out_data + r * cols, cols, 1);
            }
        } else {
            ref::log_softmax_f32(in_data, out_data, a.num_elements(), 1);
        }
    } else {
        return MakeUnexpected<Tensor>(Status::NotImplemented("Only last-axis log_softmax supported"));
    }
    
    return Ok(std::move(out));
}

Result<Tensor> softmax_with_temperature(const Tensor& a, float temperature,
                                        int axis, Stream* stream) {
    SCI_ASSERT(a.dtype() == DType::kFloat32, "Only float32 supported for now");
    SCI_ASSERT(temperature > 0, "Temperature must be positive");
    
    Tensor scaled = Tensor::Empty(a.shape(), a.dtype(), a.device());
    
    // Scale by temperature
    const float* in_data = static_cast<const float*>(a.data());
    float* scaled_data = static_cast<float*>(scaled.data());
    float inv_temp = 1.0f / temperature;
    size_t n = a.num_elements();
    
    for (size_t i = 0; i < n; ++i) {
        scaled_data[i] = in_data[i] * inv_temp;
    }
    
    return softmax(scaled, axis, stream);
}

Result<Tensor> hardmax(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(a.dtype() == DType::kFloat32, "Only float32 supported for now");
    
    Tensor out(a.shape(), DType::kFloat32, a.device());
    out.fill_zero();
    
    const float* in_data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());
    
    if (axis == -1 || axis == a.ndims() - 1) {
        if (a.ndims() == 2) {
            int rows = static_cast<int>(a.dim(0));
            int cols = static_cast<int>(a.dim(1));
            for (int r = 0; r < rows; ++r) {
                int max_idx = 0;
                float max_val = in_data[r * cols];
                for (int c = 1; c < cols; ++c) {
                    if (in_data[r * cols + c] > max_val) {
                        max_val = in_data[r * cols + c];
                        max_idx = c;
                    }
                }
                out_data[r * cols + max_idx] = 1.0f;
            }
        }
    }
    
    return Ok(std::move(out));
}

Result<Tensor> softmax_stable(const Tensor& a, int axis, Stream* stream) {
    // Stable softmax is the default implementation (subtracts max)
    return softmax(a, axis, stream);
}

} // namespace math
} // namespace sci
