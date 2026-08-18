#include "math/reduction.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace sci {
namespace math {

// ============================================================================
// Reference Implementations
// ============================================================================
namespace ref {

float sum_f32(const float* data, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result += data[i];
    }
    return result;
}

float max_f32(const float* data, size_t n) {
    float result = data[0];
    for (size_t i = 1; i < n; ++i) {
        result = std::max(result, data[i]);
    }
    return result;
}

float min_f32(const float* data, size_t n) {
    float result = data[0];
    for (size_t i = 1; i < n; ++i) {
        result = std::min(result, data[i]);
    }
    return result;
}

float mean_f32(const float* data, size_t n) {
    return sum_f32(data, n) / static_cast<float>(n);
}

float var_f32(const float* data, size_t n) {
    float m = mean_f32(data, n);
    float var = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = data[i] - m;
        var += diff * diff;
    }
    return var / static_cast<float>(n);
}

} // namespace ref

// ============================================================================
// Reduction Operations Implementation
// ============================================================================

Result<Tensor> sum(const Tensor& a, Stream* stream) {
    // Scalar reductions must produce a 1-element tensor; TensorShape{} gives
    // a zero-byte buffer with a null data pointer and any write segfaults.
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    *out_data = ref::sum_f32(data, a.num_elements());

    return Ok<Tensor>(std::move(out));
}

Result<Tensor> sum(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(axis >= 0 && axis < a.ndims(), "Invalid axis");

    TensorShape out_shape = a.shape();
    out_shape[axis] = 1;

    Tensor out(out_shape, a.dtype(), a.device());

    // TODO: Implement along-axis reduction
    return MakeUnexpected<Tensor>(Status::NotImplemented("Along-axis reduction not yet implemented"));
}

Result<Tensor> mean(const Tensor& a, Stream* stream) {
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    *out_data = ref::mean_f32(data, a.num_elements());

    return Ok<Tensor>(std::move(out));
}

Result<Tensor> mean(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(axis >= 0 && axis < a.ndims(), "Invalid axis");
    
    // TODO: Implement along-axis mean
    return MakeUnexpected<Tensor>(Status::NotImplemented("Along-axis mean not yet implemented"));
}

Result<Tensor> max(const Tensor& a, Stream* stream) {
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    *out_data = ref::max_f32(data, a.num_elements());

    return Ok<Tensor>(std::move(out));
}

Result<std::pair<Tensor, Tensor>> max_with_indices(const Tensor& a, int axis, Stream* stream) {
    SCI_ASSERT(axis >= 0 && axis < a.ndims(), "Invalid axis");
    
    // TODO: Implement along-axis max with indices
    return MakeUnexpected<std::pair<Tensor, Tensor>>(Status::NotImplemented("Along-axis max_with_indices not yet implemented"));
}

Result<Tensor> min(const Tensor& a, Stream* stream) {
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    *out_data = ref::min_f32(data, a.num_elements());

    return Ok<Tensor>(std::move(out));
}

Result<Tensor> argmax(const Tensor& a, int axis, Stream* stream) {
    if (axis == -1) {
        // Global argmax
        Tensor out(TensorShape{1}, DType::kInt64, a.device());
        
        const float* data = static_cast<const float*>(a.data());
        int64_t* out_data = static_cast<int64_t*>(out.data());
        
        size_t max_idx = 0;
        float max_val = data[0];
        for (size_t i = 1; i < a.num_elements(); ++i) {
            if (data[i] > max_val) {
                max_val = data[i];
                max_idx = i;
            }
        }
        *out_data = static_cast<int64_t>(max_idx);
        
        return Ok<Tensor>(std::move(out));
    }
    
    // TODO: Implement along-axis argmax
    return MakeUnexpected<Tensor>(Status::NotImplemented("Along-axis argmax not yet implemented"));
}

Result<Tensor> argmin(const Tensor& a, int axis, Stream* stream) {
    // TODO: Implement argmin
    return MakeUnexpected<Tensor>(Status::NotImplemented("argmin not yet implemented"));
}

Result<Tensor> prod(const Tensor& a, Stream* stream) {
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    float result = 1.0f;
    for (size_t i = 0; i < a.num_elements(); ++i) {
        result *= data[i];
    }
    *out_data = result;

    return Ok<Tensor>(std::move(out));
}

Result<Tensor> std(const Tensor& a, Stream* stream) {
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());
    
    *out_data = std::sqrt(ref::var_f32(data, a.num_elements()));
    
    return Ok<Tensor>(std::move(out));
}

Result<Tensor> var(const Tensor& a, Stream* stream) {
    // 标量归约必须输出 1 元素张量; 空 TensorShape 会产生空缓冲区,
    // 写入时会发生空指针解引用。
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    *out_data = ref::var_f32(data, a.num_elements());

    return Ok<Tensor>(std::move(out));
}

Result<Tensor> norm(const Tensor& a, float p, Stream* stream) {
    // 同 var: 标量结果必须使用 1 元素形状。
    Tensor out(TensorShape{1}, a.dtype(), a.device());

    const float* data = static_cast<const float*>(a.data());
    float* out_data = static_cast<float*>(out.data());

    float sum = 0.0f;
    for (size_t i = 0; i < a.num_elements(); ++i) {
        sum += std::pow(std::abs(data[i]), p);
    }
    *out_data = std::pow(sum, 1.0f / p);

    return Ok<Tensor>(std::move(out));
}

} // namespace math
} // namespace sci
