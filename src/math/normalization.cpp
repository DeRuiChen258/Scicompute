#include "math/normalization.hpp"
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

void layer_norm_f32(const float* x, const float* weight, const float* bias,
                    float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        // Compute mean
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) {
            mean += x[r * cols + c];
        }
        mean /= static_cast<float>(cols);
        
        // Compute variance
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float diff = x[r * cols + c] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        
        // Normalize
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int c = 0; c < cols; ++c) {
            float normalized = (x[r * cols + c] - mean) * inv_std;
            float w = weight ? weight[c] : 1.0f;
            float b = bias ? bias[c] : 0.0f;
            out[r * cols + c] = normalized * w + b;
        }
    }
}

void rms_norm_f32(const float* x, const float* weight,
                  float* out, int rows, int cols, float eps) {
    for (int r = 0; r < rows; ++r) {
        // Compute RMS
        float sum_sq = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float val = x[r * cols + c];
            sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(cols) + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize
        for (int c = 0; c < cols; ++c) {
            float w = weight ? weight[c] : 1.0f;
            out[r * cols + c] = x[r * cols + c] * inv_rms * w;
        }
    }
}

} // namespace ref

// ============================================================================
// Normalization Implementation
// ============================================================================

Result<Tensor> layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                          float eps, Stream* stream) {
    SCI_ASSERT(x.dtype() == DType::kFloat32, "Only float32 supported");
    SCI_ASSERT(weight.dtype() == DType::kFloat32, "Weight must be float32");
    SCI_ASSERT(bias.dtype() == DType::kFloat32, "Bias must be float32");
    
    Tensor out(x.shape(), x.dtype(), x.device());
    
    int rows = 1;
    for (int i = 0; i < x.ndims() - 1; ++i) {
        rows *= x.dim(i);
    }
    int cols = x.dim(x.ndims() - 1);
    
    const float* x_data = static_cast<const float*>(x.data());
    const float* w_data = static_cast<const float*>(weight.data());
    const float* b_data = static_cast<const float*>(bias.data());
    float* out_data = static_cast<float*>(out.data());
    
    ref::layer_norm_f32(x_data, w_data, b_data, out_data, rows, cols, eps);
    
    return Ok(std::move(out));
}

Result<Tensor> layer_norm(const Tensor& x, const std::vector<int>& normalized_shape,
                          float eps, Stream* stream) {
    // Create default weight and bias
    int norm_dim = normalized_shape.back();
    
    Tensor weight(TensorShape{norm_dim}, DType::kFloat32, x.device());
    Tensor bias(TensorShape{norm_dim}, DType::kFloat32, x.device());
    
    // Initialize weight to 1 and bias to 0
    float w_val = 1.0f;
    float b_val = 0.0f;
    weight.fill(&w_val);
    bias.fill(&b_val);
    
    return layer_norm(x, weight, bias, eps, stream);
}

Result<Tensor> rms_norm(const Tensor& x, const Tensor& weight,
                        float eps, Stream* stream) {
    SCI_ASSERT(x.dtype() == DType::kFloat32, "Only float32 supported");
    
    Tensor out(x.shape(), x.dtype(), x.device());
    
    int rows = 1;
    for (int i = 0; i < x.ndims() - 1; ++i) {
        rows *= x.dim(i);
    }
    int cols = x.dim(x.ndims() - 1);
    
    const float* x_data = static_cast<const float*>(x.data());
    const float* w_data = weight.data() ? static_cast<const float*>(weight.data()) : nullptr;
    float* out_data = static_cast<float*>(out.data());
    
    ref::rms_norm_f32(x_data, w_data, out_data, rows, cols, eps);
    
    return Ok(std::move(out));
}

Result<Tensor> rms_norm(const Tensor& x, const std::vector<int>& normalized_shape,
                        float eps, Stream* stream) {
    int norm_dim = normalized_shape.back();
    
    Tensor weight(TensorShape{norm_dim}, DType::kFloat32, x.device());
    float w_val = 1.0f;
    weight.fill(&w_val);
    
    return rms_norm(x, weight, eps, stream);
}

Result<Tensor> batch_norm(const Tensor& x, const Tensor& running_mean,
                          const Tensor& running_var, const Tensor& weight,
                          const Tensor& bias, float eps,
                          float momentum, Stream* stream) {
    return MakeUnexpected<Tensor>(Status::NotImplemented("BatchNorm not yet implemented"));
}

} // namespace math
} // namespace sci
