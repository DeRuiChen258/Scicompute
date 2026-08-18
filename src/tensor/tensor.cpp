#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "memory/allocator.hpp"
#include <cstring>
#include <algorithm>

namespace sci {

// ============================================================================
// Tensor Implementation
// ============================================================================
Tensor::Tensor() = default;

Tensor::Tensor(const TensorShape& shape, DType dtype, Device& device)
    : shape_(shape), dtype_(dtype), device_(&device) {
    size_t bytes = num_elements() * kDTypeSize(dtype);
    buffer_ = BufferHandle(&device, bytes);
}

Tensor::Tensor(const TensorShape& shape, DType dtype, Device& device,
               void* external_data, bool owns_data)
    : shape_(shape), dtype_(dtype), device_(&device) {
    size_t bytes = num_elements() * kDTypeSize(dtype);
    if (owns_data) {
        buffer_ = BufferHandle(external_data, bytes, device.type(),
                              [&device](void* p) { device.deallocate(p); });
    } else {
        buffer_ = BufferHandle(external_data, bytes, device.type(), nullptr);
    }
}

Tensor::Tensor(std::initializer_list<index_t> dims, DType dtype, Device& device)
    : Tensor(TensorShape(dims), dtype, device) {}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(other.shape_), dtype_(other.dtype_), layout_(other.layout_),
      device_(other.device_), buffer_(std::move(other.buffer_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = other.shape_;
        dtype_ = other.dtype_;
        layout_ = other.layout_;
        device_ = other.device_;
        buffer_ = std::move(other.buffer_);
    }
    return *this;
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), layout_(other.layout_),
      device_(other.device_) {
    if (other.buffer_.data()) {
        size_t bytes = other.num_bytes();
        buffer_ = BufferHandle(other.device_, bytes);
        other.device_->copy_to_device(buffer_.data(), other.buffer_.data(), bytes);
    }
}

// Factory methods
Tensor Tensor::Empty(const TensorShape& shape, DType dtype, Device& device) {
    return Tensor(shape, dtype, device);
}

Tensor Tensor::Zeros(const TensorShape& shape, DType dtype, Device& device) {
    Tensor t(shape, dtype, device);
    t.fill_zero();
    return t;
}

Tensor Tensor::Ones(const TensorShape& shape, DType dtype, Device& device) {
    Tensor t(shape, dtype, device);
    t.fill_ones();
    return t;
}

Tensor Tensor::Full(const TensorShape& shape, DType dtype, Device& device, const void* value) {
    Tensor t(shape, dtype, device);
    t.fill(value);
    return t;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        Tensor tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

TensorView Tensor::view() const {
    return TensorView(*this);
}

TensorView Tensor::reshape(const TensorShape& new_shape) const {
    SCI_ASSERT(num_elements() == new_shape.num_elements(),
               "Cannot reshape: element count mismatch");
    return TensorView(buffer_.data(), new_shape, dtype_, device_->type(), layout_);
}

TensorView Tensor::slice(size_t dim, index_t start, index_t end) const {
    SCI_ASSERT(dim < ndims(), "Invalid dimension");
    SCI_ASSERT(start >= 0 && end <= dim(dim), "Invalid slice range");
    
    TensorShape new_shape = shape_;
    new_shape[dim] = end - start;
    
    // Calculate offset
    size_t offset = start * stride(dim);
    const char* data_ptr = static_cast<const char*>(buffer_.data());
    data_ptr += offset * kDTypeSize(dtype_);
    
    return TensorView(data_ptr, new_shape, dtype_, device_->type(), layout_);
}

TensorView Tensor::transpose(size_t dim0, size_t dim1) const {
    SCI_ASSERT(dim0 < ndims() && dim1 < ndims(), "Invalid dimensions");
    
    TensorShape new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    
    // Calculate new strides
    std::vector<index_t> new_strides = strides();
    std::swap(new_strides[dim0], new_strides[dim1]);
    
    return TensorView(new_shape, dtype_, device_->type(),
                      Layout::kStrided, new_strides, buffer_.data());
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_, *device_);
    t.copy_from(*this);
    return t;
}

Tensor Tensor::to(Device& target_device) const {
    Tensor t(shape_, dtype_, target_device);
    if (device_->type() == target_device.type()) {
        t.copy_from(*this);
    } else {
        // Cross-device copy
        target_device.copy_from_device(t.data(), buffer_.data(), num_bytes());
    }
    return t;
}

Result<Tensor> Tensor::to(DeviceType type, int device_id) const {
    auto device = DeviceManager::Instance().get_device(type, device_id);
    if (!device) {
        return MakeUnexpected<Tensor>(Status::InvalidArgument("Invalid device type"));
    }
    return Ok<sci::Tensor>(to(*device));
}

void Tensor::fill(const void* value) {
    size_t elem_size = kDTypeSize(dtype_);
    size_t n = num_elements();
    if (n == 0) return;
    char* data = static_cast<char*>(buffer_.data());

    // Seed the first element with the value, then double-broadcast until full.
    std::memcpy(data, value, elem_size);
    for (size_t step = 1; step < n; step *= 2) {
        size_t copy = std::min(step, n - step);
        std::memcpy(data + step * elem_size, data, copy * elem_size);
    }
}

void Tensor::fill_zero() {
    device_->memset(buffer_.data(), 0, num_bytes());
}

void Tensor::fill_ones() {
    if (dtype_ == DType::kFloat32) {
        float val = 1.0f;
        fill(&val);
    } else if (dtype_ == DType::kInt32) {
        int val = 1;
        fill(&val);
    }
    // TODO: Handle other dtypes
}

void Tensor::copy_from(const Tensor& other) {
    SCI_ASSERT(is_same_shape(other), "Shape mismatch for copy");
    SCI_ASSERT(dtype_ == other.dtype_, "Dtype mismatch for copy");
    
    size_t bytes = num_bytes();
    if (device_->type() == other.device_->type()) {
        device_->copy_to_device(buffer_.data(), other.buffer_.data(), bytes);
    } else {
        // Cross-device copy
        device_->copy_from_device(buffer_.data(), other.buffer_.data(), bytes);
    }
}

void Tensor::copy_from(const void* data, size_t bytes) {
    SCI_ASSERT(bytes == num_bytes(), "Size mismatch for copy");
    device_->copy_to_device(buffer_.data(), data, bytes);
}

bool Tensor::is_same_shape(const Tensor& other) const {
    return shape_ == other.shape_;
}

Result<Tensor> Tensor::to_dtype(DType new_dtype) const {
    if (dtype_ == new_dtype) {
        return Ok<Tensor>(clone());
    }
    return MakeUnexpected<Tensor>(Status::NotImplemented("Dtype conversion not yet implemented"));
}

// ============================================================================
// TensorView Implementation
// ============================================================================
TensorView::TensorView(const Tensor& tensor)
    : data_(tensor.data()), shape_(tensor.shape()), dtype_(tensor.dtype()),
      device_(tensor.device_type()), layout_(tensor.layout()),
      strides_(tensor.strides()) {}

TensorView::TensorView(const void* data, const TensorShape& shape, DType dtype,
                       DeviceType device, Layout layout)
    : data_(data), shape_(shape), dtype_(dtype), device_(device), layout_(layout),
      strides_(shape.strides()) {}

TensorView::TensorView(const TensorShape& shape, DType dtype, DeviceType device,
                       Layout layout, const std::vector<index_t>& strides,
                       const void* data)
    : data_(data), shape_(shape), dtype_(dtype), device_(device),
      layout_(layout), strides_(strides) {}

Tensor TensorView::clone() const {
    auto device = DeviceManager::Instance().get_device(device_, 0);
    Tensor t(shape_, dtype_, *device);
    device->copy_to_device(t.data(), data_, num_bytes());
    return t;
}

bool TensorView::is_contiguous() const {
    if (layout_ != Layout::kContiguous) return false;
    
    index_t expected_stride = 1;
    for (index_t i = ndims() - 1; i >= 0; --i) {
        if (strides_[i] != expected_stride) return false;
        expected_stride *= dim(i);
    }
    return true;
}

template<typename IndexT>
const void* TensorView::at(const std::vector<IndexT>& indices) const {
    size_t offset = this->offset(indices);
    const char* data_ptr = static_cast<const char*>(data_);
    return data_ptr + offset * kDTypeSize(dtype_);
}

template<typename IndexT>
void* TensorView::at(const std::vector<IndexT>& indices) {
    return const_cast<void*>(const_cast<const TensorView*>(this)->at(indices));
}

template<typename IndexT>
size_t TensorView::offset(const std::vector<IndexT>& indices) const {
    size_t offset = 0;
    for (size_t i = 0; i < indices.size() && i < strides_.size(); ++i) {
        offset += indices[i] * strides_[i];
    }
    return offset;
}

// Explicit template instantiations
template const void* TensorView::at<int>(const std::vector<int>&) const;
template void* TensorView::at<int>(const std::vector<int>&);
template size_t TensorView::offset<int>(const std::vector<int>&) const;

// ============================================================================
// TensorOptions Implementation
// ============================================================================
TensorOptions& TensorOptions::set_device(DeviceType type, int device_id) {
    device_ = DeviceManager::Instance().get_device(type, device_id).get();
    return *this;
}

Tensor TensorOptions::create(const TensorShape& shape) const {
    SCI_ASSERT(device_ != nullptr, "Device not set");
    return Tensor(shape, dtype_, *device_);
}

} // namespace sci
