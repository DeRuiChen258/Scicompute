// ============================================================================
// Softmax Operations Benchmark
// ============================================================================

#include <benchmark/benchmark.h>
#include "core/common.hpp"
#include "tensor/tensor.hpp"
#include "math/softmax.hpp"
#include "device/device.hpp"

using namespace sci;

static void BM_Softmax(benchmark::State& state) {
    auto device = GetCPUDevice();
    int rows = state.range(0);
    int cols = state.range(1);
    
    Tensor a = Tensor::Ones({rows, cols}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::softmax(a, -1);
        if (!result.ok()) {
            state.SkipWithError("Softmax operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(BM_Softmax)
    ->Args({32, 512})
    ->Args({64, 512})
    ->Args({128, 512})
    ->Args({32, 1024})
    ->Args({64, 1024})
    ->Unit(benchmark::kMicrosecond);

static void BM_LogSoftmax(benchmark::State& state) {
    auto device = GetCPUDevice();
    int rows = state.range(0);
    int cols = state.range(1);
    
    Tensor a = Tensor::Ones({rows, cols}, DType::kFloat32, *device);
    
    for (auto _ : state) {
        auto result = math::log_softmax(a, -1);
        if (!result.ok()) {
            state.SkipWithError("LogSoftmax operation failed");
            break;
        }
    }
    
    state.SetItemsProcessed(state.iterations() * rows * cols);
}

BENCHMARK(BM_LogSoftmax)
    ->Args({32, 512})
    ->Args({64, 512})
    ->Args({128, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
