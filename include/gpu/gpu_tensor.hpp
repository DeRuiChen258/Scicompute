#pragma once
/**
 * @file gpu_tensor.hpp
 * @brief GPU张量 - CUDA GPU上的多维数组 (支持统一共享内存)
 */

#include "gpu_memory.hpp"
#include "../core/common.hpp"
#include <vector>
#include <memory>

namespace sci {
namespace gpu {

enum class GpuDType { kFloat32 = 0, kFloat16 = 1, kInt32 = 2, kInt64 = 3, kInt8 = 4, kUInt8 = 5 };

inline size_t get_dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 4; case 1: return 2; case 2: return 4; case 3: return 8;
        case 4: return 1; case 5: return 1; default: return 4;
    }
}

class GpuTensor {
public:
    GpuTensor() : data_(nullptr), dtype_(0), owns_data_(false) {}
    GpuTensor(const std::vector<int64_t>& shape, int dtype = 0);
    GpuTensor(void* data, const std::vector<int64_t>& shape, const std::vector<int64_t>& strides, int dtype);
    ~GpuTensor();
    GpuTensor(const GpuTensor& other);
    GpuTensor& operator=(const GpuTensor& other);
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;
    void* data() { return data_; }
    const void* data() const { return data_; }
    template<typename T> T* data() { return reinterpret_cast<T*>(data_); }
    template<typename T> const T* data() const { return reinterpret_cast<const T*>(data_); }
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    int dtype() const { return dtype_; }
    int64_t size() const { int64_t t = 1; for (auto d : shape_) t *= d; return t; }
    int64_t ndim() const { return static_cast<int64_t>(shape_.size()); }
    size_t nbytes() const { return static_cast<size_t>(size()) * get_dtype_size(dtype_); }
    bool is_contiguous() const;
    GpuTensor view(const std::vector<int64_t>& new_shape);
    GpuTensor reshape(const std::vector<int64_t>& new_shape);
    void copy_from(const void* host_data, size_t bytes);
    void copy_to(void* host_data, size_t bytes) const;
    void fill(float value);
    void zero();
    friend class GpuTensorFactory;
protected:
    void allocate();
    void deallocate();
    void compute_strides();
    void* data_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    int dtype_;
    bool owns_data_;
};

class GpuTensorFactory {
public:
    static GpuTensor create(const std::vector<int64_t>& shape, int dtype = 0) {
        return GpuTensor(shape, dtype);
    }
    static GpuTensor create_uninitialized(const std::vector<int64_t>& shape, int dtype = 0) {
        GpuTensor t;
        t.shape_ = shape;
        t.dtype_ = dtype;
        t.owns_data_ = true;
        t.compute_strides();
        t.allocate();
        return t;
    }
    static GpuTensor from_host(const void* data, const std::vector<int64_t>& shape, int dtype = 0) {
        GpuTensor t(shape, dtype);
        t.copy_from(data, t.nbytes());
        return t;
    }
    static GpuTensor zeros(const std::vector<int64_t>& shape, int dtype = 0) {
        GpuTensor t(shape, dtype);
        t.zero();
        return t;
    }
    static GpuTensor randn(const std::vector<int64_t>& shape, float mean = 0.0f, float std_dev = 1.0f) {
        GpuTensor t(shape, 0);
        t.zero();
        return t;
    }
};

} // namespace gpu
} // namespace sci
