// ============================================================================
// Elementwise Operations Benchmark
// ============================================================================

#include <benchmark/benchmark.h>
#include "core/common.hpp"
#include "tensor/tensor.hpp"
#include "math/elementwise.hpp"
#include "device/device.hpp"

using namespace sci;

static void BM_TensorAdd(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    Tensor b = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::add(a, b);
        if (!result.ok()) {
            state.SkipWithError("Add operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n);
    state.SetBytesProcessed(state.iterations() * n * sizeof(float) * 3);
}

BENCHMARK(BM_TensorAdd)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TensorMul(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    Tensor b = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::mul(a, b);
        if (!result.ok()) {
            state.SkipWithError("Mul operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_TensorMul)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TensorFill(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Empty({static_cast<int>(n)}, DType::kFloat32, *device);
    float val = 42.0f;
    
    for (auto _ : state) {
        a.fill(&val);
    }
    
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_TensorFill)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
