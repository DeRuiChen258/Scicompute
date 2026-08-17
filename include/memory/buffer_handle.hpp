#pragma once

#include "../core/common.hpp"
#include "../device/device.hpp"

namespace sci {

class BufferHandle {
public:
    BufferHandle() = default;
    
    BufferHandle(void* data, size_t bytes, DeviceType device,
                 std::function<void(void*)> dealloc = nullptr);
    
    BufferHandle(Device* device, size_t bytes);
    
    ~BufferHandle();
    
    BufferHandle(BufferHandle&& other) noexcept;
    BufferHandle& operator=(BufferHandle&& other) noexcept;
    
    BufferHandle(const BufferHandle&) = delete;
    BufferHandle& operator=(const BufferHandle&) = delete;
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    size_t bytes() const { return bytes_; }
    DeviceType device_type() const { return device_; }
    Device* device() const { return device_ptr_; }
    bool owns_data() const { return owns_data_; }
    
    explicit operator bool() const { return data_ != nullptr; }
    
    void reset();
    void reset(void* data, size_t bytes, DeviceType device,
               std::function<void(void*)> dealloc);
    
    void* detach();
    
    BufferHandle clone() const;
    
protected:
    // Protected setters for derived classes (like TensorBuffer)
    void set_data_ptr(void* data) { data_ = data; }
    void set_bytes(size_t bytes) { bytes_ = bytes; }
    void set_device_info(DeviceType dtype, Device* dev) { device_ = dtype; device_ptr_ = dev; }
    void set_ownership(bool owns, std::function<void(void*)> dealloc) { 
        owns_data_ = owns; 
        dealloc_fn_ = dealloc; 
    }
    void call_deallocate() { deallocate(); }
    
private:
    void deallocate();
    
    void* data_ = nullptr;
    size_t bytes_ = 0;
    DeviceType device_ = DeviceType::kCPU;
    Device* device_ptr_ = nullptr;
    std::function<void(void*)> dealloc_fn_;
    bool owns_data_ = false;
};

class BufferView {
public:
    BufferView() = default;
    
    BufferView(const void* data, size_t bytes, DeviceType device)
        : data_(const_cast<void*>(data)), bytes_(bytes), device_(device) {}
    
    BufferView(const BufferHandle& handle)
        : data_(const_cast<void*>(handle.data())), bytes_(handle.bytes()),
          device_(handle.device_type()) {}
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    size_t bytes() const { return bytes_; }
    DeviceType device_type() const { return device_; }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(data_); }
    
    template<typename T>
    T* as() { return static_cast<T*>(data_); }
    
    explicit operator bool() const { return data_ != nullptr; }

private:
    void* data_ = nullptr;
    size_t bytes_ = 0;
    DeviceType device_ = DeviceType::kCPU;
};

class TensorBuffer : public BufferHandle {
public:
    TensorBuffer() = default;
    
    TensorBuffer(const TensorShape& shape, DType dtype, Device* device);
    
    void resize(const TensorShape& shape, DType dtype, Device* device);
    
    TensorShape shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    size_t num_elements() const { return shape_.num_elements(); }
    size_t num_bytes() const { return num_elements() * kDTypeSize(dtype_); }
    
private:
    TensorShape shape_;
    DType dtype_ = DType::kFloat32;
};

} // namespace sci
