#pragma once

#include "../tensor/tensor.hpp"

namespace sci {
namespace math {

// ============================================================================
// Sampling Operations
// ============================================================================

// TopK: returns top k values and indices
struct TopKResult {
    Tensor values;
    Tensor indices;
};

Result<TopKResult> topk(const Tensor& a, int k, int axis = -1, 
                        bool largest = true, bool sorted = true,
                        Stream* stream = nullptr);

// Sampling: multinomial sampling
Result<Tensor> multinomial(const Tensor& probs, int num_samples,
                           bool replacement = false, Stream* stream = nullptr);

// Weighted sampling
Result<Tensor> weighted_sampling(const Tensor& weights, int num_samples,
                                  uint64_t seed = 0, Stream* stream = nullptr);

// Categorical sampling (alias for multinomial with 1 sample)
Result<Tensor> categorical(const Tensor& logits, uint64_t seed = 0,
                           Stream* stream = nullptr);

// Reference implementations
namespace ref {
void topk_f32(const float* x, float* values, int* indices, 
              size_t n, int k, bool largest, bool sorted);
int multinomial_f32(const float* probs, size_t n, int num_samples, 
                    bool replacement, uint64_t seed);
} // namespace ref

} // namespace math
} // namespace sci
