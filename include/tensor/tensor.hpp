#pragma once

#include "../core/common.hpp"
#include "../memory/buffer_handle.hpp"
#include <initializer_list>

namespace sci {

// ============================================================================
// Forward Declarations
// ============================================================================
class Tensor;
class TensorView;

// ============================================================================
// Tensor - Multi-dimensional array
// ============================================================================
class Tensor {
public:
    // Constructors
    Tensor();
    
    Tensor(const TensorShape& shape, DType dtype, Device& device);
    
    Tensor(const TensorShape& shape, DType dtype, Device& device,
           void* external_data, bool owns_data = false);
    
    Tensor(std::initializer_list<index_t> dims, DType dtype, Device& device);
    
    // Factory methods
    static Tensor Empty(const TensorShape& shape, DType dtype, Device& device);
    static Tensor Zeros(const TensorShape& shape, DType dtype, Device& device);
    static Tensor Ones(const TensorShape& shape, DType dtype, Device& device);
    static Tensor Full(const TensorShape& shape, DType dtype, Device& device, const void* value);

    // initializer_list convenience overloads (mirror PyTorch style)
    static Tensor Empty(std::initializer_list<index_t> dims, DType dtype, Device& device) {
        return Empty(TensorShape(dims), dtype, device);
    }
    static Tensor Zeros(std::initializer_list<index_t> dims, DType dtype, Device& device) {
        return Zeros(TensorShape(dims), dtype, device);
    }
    static Tensor Ones(std::initializer_list<index_t> dims, DType dtype, Device& device) {
        return Ones(TensorShape(dims), dtype, device);
    }
    static Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, const void* value) {
        return Full(TensorShape(dims), dtype, device, value);
    }
    // Scalar convenience overloads so callers can pass float/int without
    // an explicit cast to const void*.
    static Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, float value) {
        return Full(TensorShape(dims), dtype, device, &value);
    }
    static Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, double value) {
        return Full(TensorShape(dims), dtype, device, &value);
    }
    static Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, int32_t value) {
        return Full(TensorShape(dims), dtype, device, &value);
    }
    static Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, int64_t value) {
        return Full(TensorShape(dims), dtype, device, &value);
    }
    static Tensor Rand(const TensorShape& shape, DType dtype, Device& device);
    static Tensor Randn(const TensorShape& shape, DType dtype, Device& device);
    
    // Destructor
    ~Tensor();
    
    // Move semantics
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;
    
    // Copy is deep copy
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    
    // Shape
    const TensorShape& shape() const { return shape_; }
    TensorShape& shape() { return shape_; }
    index_t ndims() const { return shape_.ndims(); }
    index_t dim(size_t i) const { return shape_.dim(i); }
    index_t num_elements() const { return shape_.num_elements(); }
    
    // Data type and device
    DType dtype() const { return dtype_; }
    Device& device() const { return *device_; }
    DeviceType device_type() const { return device_->type(); }
    
    // Memory
    void* data() { return buffer_.data(); }
    const void* data() const { return buffer_.data(); }
    size_t num_bytes() const { return buffer_.bytes(); }
    bool is_contiguous() const { return layout_ == Layout::kContiguous; }
    Layout layout() const { return layout_; }
    
    // Strides (row-major)
    std::vector<index_t> strides() const { return shape_.strides(); }
    index_t stride(size_t i) const { return strides()[i]; }
    
    // View operations (non-copying)
    TensorView view() const;
    TensorView reshape(const TensorShape& new_shape) const;
    TensorView slice(size_t dim, index_t start, index_t end) const;
    TensorView transpose(size_t dim0, size_t dim1) const;
    
    // Copy operations
    Tensor clone() const;
    Tensor to(Device& device) const;
    Result<Tensor> to(DeviceType type, int device_id = 0) const;
    
    // Fill
    void fill(const void* value);
    void fill_zero();
    void fill_ones();
    
    // Copy data
    void copy_from(const Tensor& other);
    void copy_from(const void* data, size_t bytes);
    
    // Comparison
    bool is_same_shape(const Tensor& other) const;
    bool is_same(const Tensor& other) const;  // Same shape, dtype, device, data
    
    // Cast
    Result<Tensor> to_dtype(DType new_dtype) const;
    
    // Raw pointer access (careful!)
    template<typename T>
    T* data_ptr() { return static_cast<T*>(data()); }
    
    template<typename T>
    const T* data_ptr() const { return static_cast<const T*>(data()); }
    
    explicit operator bool() const { return buffer_.data() != nullptr; }
    
private:
    friend class TensorView;
    
    TensorShape shape_;
    DType dtype_ = DType::kFloat32;
    Layout layout_ = Layout::kContiguous;
    Device* device_ = nullptr;
    BufferHandle buffer_;
};

// ============================================================================
// TensorView - Non-owning tensor reference
// ============================================================================
class TensorView {
public:
    TensorView() = default;
    
    TensorView(const Tensor& tensor);
    
    TensorView(const void* data, const TensorShape& shape, DType dtype,
               DeviceType device, Layout layout = Layout::kContiguous);
    
    TensorView(const TensorShape& shape, DType dtype, DeviceType device,
               Layout layout, const std::vector<index_t>& strides,
               const void* data);
    
    // Shape
    const TensorShape& shape() const { return shape_; }
    TensorShape& shape() { return shape_; }
    index_t ndims() const { return shape_.ndims(); }
    index_t dim(size_t i) const { return shape_.dim(i); }
    index_t num_elements() const { return shape_.num_elements(); }
    
    // Data type and device
    DType dtype() const { return dtype_; }
    DeviceType device_type() const { return device_; }
    
    // Memory
    const void* data() const { return data_; }
    void* data() { return const_cast<void*>(data_); }
    size_t num_bytes() const { return num_elements() * kDTypeSize(dtype_); }
    Layout layout() const { return layout_; }
    
    // Strides
    const std::vector<index_t>& strides() const { return strides_; }
    index_t stride(size_t i) const { return strides_[i]; }
    
    // Indexing helpers
    template<typename IndexT>
    const void* at(const std::vector<IndexT>& indices) const;
    
    template<typename IndexT>
    void* at(const std::vector<IndexT>& indices);
    
    // Offset calculation
    template<typename IndexT>
    size_t offset(const std::vector<IndexT>& indices) const;
    
    // Convert to owning Tensor
    Tensor clone() const;
    
    bool is_contiguous() const;
    
    explicit operator bool() const { return data_ != nullptr; }
    
private:
    const void* data_ = nullptr;
    TensorShape shape_;
    DType dtype_ = DType::kFloat32;
    DeviceType device_ = DeviceType::kCPU;
    Layout layout_ = Layout::kContiguous;
    std::vector<index_t> strides_;
};

// ============================================================================
// Tensor Options - Builder pattern for tensor construction
// ============================================================================
class TensorOptions {
public:
    TensorOptions() = default;
    
    TensorOptions& set_dtype(DType dtype) { dtype_ = dtype; return *this; }
    TensorOptions& set_device(Device& device) { device_ = &device; return *this; }
    TensorOptions& set_device(DeviceType type, int device_id = 0);
    TensorOptions& set_layout(Layout layout) { layout_ = layout; return *this; }
    TensorOptions& set_requires_grad(bool value) { requires_grad_ = value; return *this; }
    
    DType dtype() const { return dtype_; }
    Device* device() const { return device_; }
    Layout layout() const { return layout_; }
    bool requires_grad() const { return requires_grad_; }
    
    // Factory method
    Tensor create(const TensorShape& shape) const;
    
private:
    DType dtype_ = DType::kFloat32;
    Device* device_ = nullptr;
    Layout layout_ = Layout::kContiguous;
    bool requires_grad_ = false;
};

// Helper functions
inline TensorOptions dtype(DType dtype) { return TensorOptions().set_dtype(dtype); }
inline TensorOptions device(Device& dev) { return TensorOptions().set_device(dev); }
inline TensorOptions device(DeviceType type, int id = 0) { return TensorOptions().set_device(type, id); }
inline TensorOptions layout(Layout l) { return TensorOptions().set_layout(l); }

// Scalar convenience overloads for Tensor::Full (initializer_list + value).
// These accept the common C++ scalar types without an explicit cast to
// const void*. They are wrapped into inline helpers rather than members
// because overload resolution prefers them over the const void* version.
inline Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, float value) {
    return Tensor::Full(TensorShape(dims), dtype, device, &value);
}
inline Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, double value) {
    return Tensor::Full(TensorShape(dims), dtype, device, &value);
}
inline Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, int32_t value) {
    return Tensor::Full(TensorShape(dims), dtype, device, &value);
}
inline Tensor Full(std::initializer_list<index_t> dims, DType dtype, Device& device, int64_t value) {
    return Tensor::Full(TensorShape(dims), dtype, device, &value);
}

}  // namespace sci
