// ============================================================================
// LayerNorm Operations Benchmark
// ============================================================================

#include <benchmark/benchmark.h>

#include "core/common.hpp"
#include "tensor/tensor.hpp"
#include "math/normalization.hpp"
#include "device/device.hpp"

using namespace sci;

static void BM_LayerNorm(benchmark::State& state) {
    auto device = GetCPUDevice();
    int rows = state.range(0);
    int cols = state.range(1);

    Tensor x({rows, cols}, DType::kFloat32, *device);
    float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.num_elements(); ++i) {
        // 确定性伪随机数据, 避免依赖未实现的 Tensor::Randn
        xd[i] = static_cast<float>((i * 73) % 97 - 48);
    }

    Tensor weight = Tensor::Ones({cols}, DType::kFloat32, *device);
    Tensor bias = Tensor::Zeros({cols}, DType::kFloat32, *device);

    for (auto _ : state) {
        auto result = math::layer_norm(x, weight, bias, 1e-5f);
        if (!result.ok()) {
            state.SkipWithError("LayerNorm operation failed");
            break;
        }
        benchmark::DoNotOptimize(result->data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows) * cols);
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(rows) * cols * sizeof(float) * 2);
}

BENCHMARK(BM_LayerNorm)
    ->Args({32, 512})
    ->Args({64, 512})
    ->Args({128, 512})
    ->Args({32, 1024})
    ->Args({64, 1024})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
