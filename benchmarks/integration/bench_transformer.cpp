// ============================================================================
// Transformer 层核心流水线 Benchmark (CPU 参考实现)
//   LayerNorm -> FFN (elementwise) -> Residual -> Softmax -> Reduction
// ============================================================================

#include <benchmark/benchmark.h>

#include "core/common.hpp"
#include "tensor/tensor.hpp"
#include "math/normalization.hpp"
#include "math/elementwise.hpp"
#include "math/softmax.hpp"
#include "math/reduction.hpp"
#include "device/device.hpp"

using namespace sci;

static void BM_TransformerLayerCore(benchmark::State& state) {
    auto device = GetCPUDevice();
    int rows = state.range(0);   // batch * seq_len
    int hidden = state.range(1);

    Tensor x({rows, hidden}, DType::kFloat32, *device);
    float* xd = x.data_ptr<float>();
    for (int64_t i = 0; i < x.num_elements(); ++i) {
        xd[i] = static_cast<float>((i * 73) % 97 - 48) * 0.1f;
    }

    Tensor ln_w = Tensor::Ones({hidden}, DType::kFloat32, *device);
    Tensor ln_b = Tensor::Zeros({hidden}, DType::kFloat32, *device);
    Tensor ffn_w = Tensor::Full({rows, hidden}, DType::kFloat32, *device, 0.5f);

    for (auto _ : state) {
        auto ln = math::layer_norm(x, ln_w, ln_b, 1e-5f);
        if (!ln.ok()) {
            state.SkipWithError("LayerNorm failed");
            break;
        }
        auto ff = math::mul(*ln, ffn_w);
        if (!ff.ok()) {
            state.SkipWithError("FFN mul failed");
            break;
        }
        auto res = math::add(x, *ff);
        if (!res.ok()) {
            state.SkipWithError("Residual add failed");
            break;
        }
        auto sm = math::softmax(*res, -1);
        if (!sm.ok()) {
            state.SkipWithError("Softmax failed");
            break;
        }
        auto mean = math::mean(*sm);
        if (!mean.ok()) {
            state.SkipWithError("Mean failed");
            break;
        }
        benchmark::DoNotOptimize(mean->data());
    }

    state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows) * hidden);
    state.SetBytesProcessed(
        state.iterations() * static_cast<int64_t>(rows) * hidden * sizeof(float) * 6);
}

BENCHMARK(BM_TransformerLayerCore)
    ->Args({64, 256})
    ->Args({128, 256})
    ->Args({64, 512})
    ->Args({128, 512})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
