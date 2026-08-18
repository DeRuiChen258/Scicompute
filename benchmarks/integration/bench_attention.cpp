// ============================================================================
// Attention 核心计算 Benchmark (CPU 参考实现)
//   QK^T -> Softmax -> PV
// ============================================================================

#include <benchmark/benchmark.h>

#include <vector>

#include "ref_gemm.hpp"

using namespace sci;

static void BM_Attention(benchmark::State& state) {
    int heads = state.range(0);
    int seq = state.range(1);
    int dim = state.range(2);

    const size_t m = static_cast<size_t>(heads) * static_cast<size_t>(seq);
    const size_t qkv_bytes = m * static_cast<size_t>(dim);
    const size_t score_bytes = m * static_cast<size_t>(seq);

    std::vector<float> q(qkv_bytes);
    std::vector<float> k(qkv_bytes);
    std::vector<float> v(qkv_bytes);
    std::vector<float> scores(score_bytes);
    std::vector<float> out(qkv_bytes); // PV 输出形状为 [m, dim]

    for (size_t i = 0; i < qkv_bytes; ++i) {
        q[i] = static_cast<float>((i * 73) % 97 - 48) * 0.1f;
        k[i] = static_cast<float>((i * 31) % 89 - 44) * 0.1f;
        v[i] = static_cast<float>((i * 17) % 83 - 41) * 0.1f;
    }

    for (auto _ : state) {
        // scores[heads*seq, seq] = q[heads*seq, dim] * k[heads*seq, dim]^T
        bench::gemm_qk(q.data(), k.data(), scores.data(), m, dim, seq);
        bench::softmax_rows(scores.data(), m, seq);
        // out[heads*seq, dim] = scores[heads*seq, seq] * v[heads*seq, dim]
        bench::gemm_nt(scores.data(), v.data(), out.data(), m, seq, dim);
        benchmark::DoNotOptimize(out.data());
    }

    const int64_t flops =
        static_cast<int64_t>(heads) * seq * seq * dim * 2;
    state.SetItemsProcessed(state.iterations() * flops);
    state.SetBytesProcessed(state.iterations() * flops * sizeof(float));
}

BENCHMARK(BM_Attention)
    ->Args({4, 16, 32})
    ->Args({4, 32, 32})
    ->Args({8, 32, 64})
    ->Args({8, 64, 64})
    ->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
