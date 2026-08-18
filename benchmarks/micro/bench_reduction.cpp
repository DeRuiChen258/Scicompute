// ============================================================================
// Reduction Operations Benchmark
// ============================================================================

#include <benchmark/benchmark.h>
#include "core/common.hpp"
#include "tensor/tensor.hpp"
#include "math/reduction.hpp"
#include "device/device.hpp"

using namespace sci;

static void BM_TensorSum(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::sum(a);
        if (!result.ok()) {
            state.SkipWithError("Sum operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_TensorSum)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TensorMean(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::mean(a);
        if (!result.ok()) {
            state.SkipWithError("Mean operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_TensorMean)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

static void BM_TensorMax(benchmark::State& state) {
    auto device = GetCPUDevice();
    size_t n = state.range(0);
    
    Tensor a = Tensor::Ones({static_cast<int>(n)}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::max(a);
        if (!result.ok()) {
            state.SkipWithError("Max operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * n);
}

BENCHMARK(BM_TensorMax)
    ->RangeMultiplier(2)
    ->Range(1024, 1 << 20)
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
