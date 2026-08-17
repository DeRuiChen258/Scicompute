#pragma once

#include "../core/common.hpp"
#include "../tensor/tensor.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <typeindex>

namespace sci {

// ============================================================================
// Kernel Traits
// ============================================================================
template<typename Output, typename... Inputs>
struct KernelTraits {
    using OutputType = Output;
    using InputTypes = std::tuple<Inputs...>;
    static constexpr size_t kNumInputs = sizeof...(Inputs);
};

// ============================================================================
// Kernel Signature
// ============================================================================
using KernelSignature = std::string;

inline KernelSignature MakeKernelSignature(const std::string& op, const std::string& dtype) {
    return op + "_" + dtype;
}

inline KernelSignature MakeKernelSignature(const std::string& op, DType dtype) {
    return MakeKernelSignature(op, kDTypeName(dtype));
}

// ============================================================================
// Kernel Registry - Central registry for all kernels
// ============================================================================
class KernelRegistry {
public:
    static KernelRegistry& Instance();
    
    // Register a kernel
    template<typename KernelFunc>
    void register_kernel(const std::string& name, KernelFunc func);
    
    // Get a kernel
    template<typename... Args>
    Result<Args...> execute(const std::string& name, Args&&... args);
    
    // Check existence
    bool has_kernel(const std::string& name) const;
    std::vector<std::string> list_kernels() const;
    
    // Unregister
    void unregister(const std::string& name);
    void clear();
    
private:
    KernelRegistry() = default;
    
    struct KernelEntry {
        std::function<Result<Tensor>(const Tensor&)> func;
        std::string name;
        DType dtype;
    };
    
    std::unordered_map<std::string, KernelEntry> kernels_;
};

// Helper macros for kernel registration
#define SCI_REGISTER_KERNEL(registry, name, func) \
    registry.register_kernel(name, func)

#define SCI_REGISTER_ELEMENTWISE(op, dtype, impl) \
    do { \
        auto sig = sci::MakeKernelSignature(#op, dtype); \
        sci::KernelRegistry::Instance().register_kernel(sig, impl); \
    } while (false)

// ============================================================================
// Kernel Launcher - Unified interface for launching kernels
// ============================================================================
class KernelLauncher {
public:
    KernelLauncher() = default;
    
    // Launch a kernel by name
    Result<Tensor> launch(const std::string& name, const Tensor& input);
    Result<Tensor> launch(const std::string& name, const Tensor& a, const Tensor& b);
    
    // Launch with options
    struct LaunchOptions {
        Device* device = nullptr;
        int block_size = 256;
        int grid_size = 0;
        size_t shared_mem = 0;
        Stream* stream = nullptr;
    };
    
    Result<Tensor> launch(const std::string& name, const Tensor& input,
                          const LaunchOptions& options);
};

// ============================================================================
// Built-in Kernel Names
// ============================================================================
namespace Kernels {
    // Elementwise
    constexpr const char* kAdd = "add";
    constexpr const char* kMul = "mul";
    constexpr const char* kSub = "sub";
    constexpr const char* kDiv = "div";
    
    // Reduction
    constexpr const char* kSum = "sum";
    constexpr const char* kMax = "max";
    constexpr const char* kMin = "min";
    constexpr const char* kMean = "mean";
    
    // Softmax
    constexpr const char* kSoftmax = "softmax";
    constexpr const char* kLogSoftmax = "log_softmax";
    
    // Normalization
    constexpr const char* kLayerNorm = "layer_norm";
    constexpr const char* kRMSNorm = "rms_norm";
    
    // Sampling
    constexpr const char* kTopK = "topk";
    constexpr const char* kArgMax = "argmax";
    constexpr const char* kSampling = "sampling";
}

} // namespace sci
