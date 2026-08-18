/**
 * @file gpu_tensor.cpp - GPU张量实现
 */

#include "gpu/gpu_tensor.hpp"
#include "gpu/gpu_memory.hpp"

namespace sci {
namespace gpu {

GpuTensor::GpuTensor(const std::vector<int64_t>& shape, int dtype)
    : shape_(shape), dtype_(dtype), owns_data_(true) {
    compute_strides();
    allocate();
}

GpuTensor::GpuTensor(void* data, const std::vector<int64_t>& shape,
                     const std::vector<int64_t>& strides, int dtype)
    : data_(data), shape_(shape), strides_(strides), dtype_(dtype), owns_data_(false) {}

GpuTensor::~GpuTensor() {
    if (owns_data_ && data_) { CudaAllocator::Instance().deallocate_managed(data_); }
}

GpuTensor::GpuTensor(const GpuTensor& other)
    : shape_(other.shape_), strides_(other.strides_), dtype_(other.dtype_), owns_data_(true) {
    if (other.data_ && other.owns_data_) {
        allocate();
        CudaAllocator::Instance().copy_managed(data_, other.data_, nbytes());
    } else {
        owns_data_ = false;
        data_ = other.data_;
    }
}

GpuTensor& GpuTensor::operator=(const GpuTensor& other) {
    if (this != &other) {
        if (owns_data_ && data_) { CudaAllocator::Instance().deallocate_managed(data_); }
        shape_ = other.shape_; strides_ = other.strides_; dtype_ = other.dtype_; owns_data_ = true;
        allocate();
        CudaAllocator::Instance().copy_managed(data_, other.data_, nbytes());
    }
    return *this;
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      dtype_(other.dtype_), owns_data_(other.owns_data_) {
    other.data_ = nullptr; other.owns_data_ = false;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        if (owns_data_ && data_) { CudaAllocator::Instance().deallocate_managed(data_); }
        data_ = other.data_; shape_ = std::move(other.shape_); strides_ = std::move(other.strides_);
        dtype_ = other.dtype_; owns_data_ = other.owns_data_;
        other.data_ = nullptr; other.owns_data_ = false;
    }
    return *this;
}

void GpuTensor::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) return;
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

void GpuTensor::allocate() {
    data_ = CudaAllocator::Instance().allocate_managed(nbytes());
}

void GpuTensor::deallocate() {
    if (owns_data_ && data_) {
        CudaAllocator::Instance().deallocate_managed(data_);
        data_ = nullptr;
    }
}

bool GpuTensor::is_contiguous() const {
    int64_t expected = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
        if (shape_[i] == 1) continue;
        if (strides_[i] != expected) return false;
        expected *= shape_[i];
    }
    return true;
}

GpuTensor GpuTensor::view(const std::vector<int64_t>& new_shape) {
    return GpuTensor(data_, new_shape, {}, dtype_);
}

GpuTensor GpuTensor::reshape(const std::vector<int64_t>& new_shape) {
    GpuTensor result(data_, new_shape, {}, dtype_);
    result.owns_data_ = false;
    return result;
}

void GpuTensor::copy_from(const void* host_data, size_t bytes) {
    CudaAllocator::Instance().copy_managed(data_, host_data, bytes);
}

void GpuTensor::copy_to(void* host_data, size_t bytes) const {
    CudaAllocator::Instance().copy_managed(host_data, data_, bytes);
}

void GpuTensor::fill(float value) {
    auto& alloc = CudaAllocator::Instance();
    auto stream = alloc.stream();
#ifdef SCI_USE_CUDA
    cudaMemsetAsync(data_, static_cast<int>(value), nbytes(), static_cast<cudaStream_t>(stream));
    alloc.synchronize();
#else
    std::fill_n(static_cast<float*>(data_), size(), value);
#endif
}

void GpuTensor::zero() { fill(0.0f); }

} // namespace gpu
} // namespace sci
