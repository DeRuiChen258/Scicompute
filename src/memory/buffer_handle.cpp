#include "memory/buffer_handle.hpp"
#include "device/device.hpp"

namespace sci {

BufferHandle::BufferHandle(void* data, size_t bytes, DeviceType device,
                           std::function<void(void*)> dealloc)
    : data_(data), bytes_(bytes), device_(device),
      dealloc_fn_(dealloc), owns_data_(true) {}

BufferHandle::BufferHandle(Device* device, size_t bytes)
    : bytes_(bytes), device_(device->type()), device_ptr_(device), owns_data_(true) {
    data_ = device->allocate(bytes);
    dealloc_fn_ = [device](void* ptr) { device->deallocate(ptr); };
}

BufferHandle::~BufferHandle() {
    deallocate();
}

BufferHandle::BufferHandle(BufferHandle&& other) noexcept
    : data_(other.data_), bytes_(other.bytes_), device_(other.device_),
      device_ptr_(other.device_ptr_), dealloc_fn_(std::move(other.dealloc_fn_)),
      owns_data_(other.owns_data_) {
    other.data_ = nullptr;
    other.owns_data_ = false;
}

BufferHandle& BufferHandle::operator=(BufferHandle&& other) noexcept {
    if (this != &other) {
        deallocate();
        data_ = other.data_;
        bytes_ = other.bytes_;
        device_ = other.device_;
        device_ptr_ = other.device_ptr_;
        dealloc_fn_ = std::move(other.dealloc_fn_);
        owns_data_ = other.owns_data_;
        other.data_ = nullptr;
        other.owns_data_ = false;
    }
    return *this;
}

void BufferHandle::deallocate() {
    if (owns_data_ && data_ && dealloc_fn_) {
        dealloc_fn_(data_);
    }
    data_ = nullptr;
    owns_data_ = false;
}

void BufferHandle::reset() {
    deallocate();
}

void BufferHandle::reset(void* data, size_t bytes, DeviceType device,
                        std::function<void(void*)> dealloc) {
    deallocate();
    data_ = data;
    bytes_ = bytes;
    device_ = device;
    dealloc_fn_ = dealloc;
    owns_data_ = true;
}

void* BufferHandle::detach() {
    owns_data_ = false;
    return data_;
}

BufferHandle BufferHandle::clone() const {
    BufferHandle result(device_ptr_, bytes_);
    if (device_ptr_ && data_) {
        device_ptr_->copy_to_device(result.data_, data_, bytes_);
    }
    return result;
}

TensorBuffer::TensorBuffer(const TensorShape& shape, DType dtype, Device* device)
    : shape_(shape), dtype_(dtype) {
    size_t bytes = num_bytes();
    set_data_ptr(device->allocate(bytes));
    set_bytes(bytes);
    set_device_info(device->type(), device);
    set_ownership(true, [device](void* ptr) { device->deallocate(ptr); });
}

void TensorBuffer::resize(const TensorShape& shape, DType dtype, Device* device) {
    shape_ = shape;
    dtype_ = dtype;
    size_t bytes = num_bytes();
    if (bytes > TensorBuffer::bytes()) {
        call_deallocate();
        set_data_ptr(device->allocate(bytes));
        set_bytes(bytes);
        set_device_info(device->type(), device);
        set_ownership(true, [device](void* ptr) { device->deallocate(ptr); });
    }
}

} // namespace sci
