#pragma once

#include "../core/types.hpp"

namespace sci {

// Extended tensor shape utilities
class TensorShapeEx {
public:
    // Expand dimensions
    static TensorShape expand_dims(const TensorShape& shape, size_t axis);
    
    // Squeeze dimensions
    static TensorShape squeeze(const TensorShape& shape);
    static TensorShape squeeze(const TensorShape& shape, size_t axis);
    
    // Broadcast shapes
    static Result<TensorShape> broadcast(const TensorShape& a, const TensorShape& b);
    
    // Validate broadcast compatibility
    static bool can_broadcast(const TensorShape& a, const TensorShape& b);
    
    // Flatten to 1D
    static TensorShape flatten(const TensorShape& shape);
    static TensorShape flatten(const TensorShape& shape, size_t start_dim, size_t end_dim);
    
    // Unflatten from 1D
    static TensorShape unflatten(const TensorShape& shape, size_t flat_dim, const TensorShape& target_shape);
};

} // namespace sci
