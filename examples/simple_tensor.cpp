#include <iostream>
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "math/elementwise.hpp"
#include "math/reduction.hpp"
#include "math/softmax.hpp"
#include "math/normalization.hpp"

using namespace sci;

void print_tensor(const Tensor& t, const std::string& name) {
    std::cout << name << " (shape: [";
    for (int i = 0; i < t.ndims(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << t.dim(i);
    }
    std::cout << "], dtype: " << static_cast<int>(t.dtype()) << "):\n";
    
    const float* data = static_cast<const float*>(t.data());
    size_t n = std::min(static_cast<size_t>(t.num_elements()), static_cast<size_t>(20));
    
    std::cout << "  [";
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << data[i];
    }
    if (static_cast<size_t>(t.num_elements()) > n) {
        std::cout << ", ...";
    }
    std::cout << "]\n\n";
}

int main() {
    std::cout << "=== SciComputeInfra Simple Tensor Demo ===\n\n";
    
    auto device = DeviceManager::Instance().get_default_device();
    std::cout << "Using device: " << device->name() << "\n\n";
    
    // Create tensors
    Tensor a = Tensor::Ones(TensorShape({3, 4}), DType::kFloat32, *device);
    print_tensor(a, "Tensor a (ones)");
    
    float val = 2.0f;
    Tensor b = Tensor::Full(TensorShape({3, 4}), DType::kFloat32, *device, &val);
    print_tensor(b, "Tensor b (filled with 2.0)");
    
    // Element-wise operations
    auto add_result = math::add(a, b);
    if (add_result) {
        print_tensor(*add_result, "a + b");
    }
    
    auto mul_result = math::mul(a, b);
    if (mul_result) {
        print_tensor(*mul_result, "a * b");
    }
    
    // Reduction operations
    auto sum_result = math::sum(a);
    if (sum_result) {
        const float* sum_data = static_cast<const float*>((*sum_result).data());
        std::cout << "Sum of all elements in a: " << sum_data[0] << "\n\n";
    }
    
    auto mean_result = math::mean(a);
    if (mean_result) {
        const float* mean_data = static_cast<const float*>((*mean_result).data());
        std::cout << "Mean of all elements in a: " << mean_data[0] << "\n\n";
    }
    
    auto max_result = math::max(a);
    if (max_result) {
        const float* max_data = static_cast<const float*>((*max_result).data());
        std::cout << "Max of all elements in a: " << max_data[0] << "\n\n";
    }
    
    // Softmax
    Tensor logits = Tensor::Empty(TensorShape({4}), DType::kFloat32, *device);
    float* logits_data = static_cast<float*>(logits.data());
    logits_data[0] = 1.0f; logits_data[1] = 2.0f; logits_data[2] = 3.0f; logits_data[3] = 4.0f;
    print_tensor(logits, "Logits");
    
    auto softmax_result = math::softmax(logits);
    if (softmax_result) {
        print_tensor(*softmax_result, "Softmax(logits)");
    }
    
    // LayerNorm
    Tensor x = Tensor::Empty(TensorShape({2, 8}), DType::kFloat32, *device);
    float* x_data = static_cast<float*>(x.data());
    for (int i = 0; i < 16; ++i) x_data[i] = static_cast<float>(i % 8);
    print_tensor(x, "Input x for LayerNorm");
    
    Tensor weight = Tensor::Ones(TensorShape({8}), DType::kFloat32, *device);
    Tensor bias = Tensor::Zeros(TensorShape({8}), DType::kFloat32, *device);
    
    auto ln_result = math::layer_norm(x, weight, bias);
    if (ln_result) {
        print_tensor(*ln_result, "LayerNorm(x)");
    }
    
    std::cout << "=== Demo Complete ===\n";
    return 0;
}
