#pragma once
/**
 * @file cuda_device.hpp
 * @brief CUDA 设备实现 - Device 抽象接口的 GPU 后端
 * @date 2026-08-18
 */

#include "device.hpp"
#include <memory>

namespace sci {

class CudaDevice final : public Device {
public:
    // 运行时检测设备是否可用 (无驱动/无设备时返回 false)
    static bool available(int device_id = 0);
    // 创建设备实例; 设备不可用时返回 nullptr
    static std::shared_ptr<CudaDevice> Create(int device_id = 0);

    DeviceType type() const override { return DeviceType::kCUDA; }
    int id() const override { return device_id_; }
    std::string name() const override;

    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    void* allocate_host(size_t bytes) override;
    void deallocate_host(void* ptr) override;

    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    void copy_within(void* dst, const void* src, size_t bytes) override;
    void copy_from_device(void* dst, const void* src, size_t bytes) override;
    void copy_async(void* dst, const void* src, size_t bytes, Stream& stream) override;

    void memset(void* ptr, int value, size_t bytes) override;
    void memset_async(void* ptr, int value, size_t bytes, Stream& stream) override;

    void synchronize() override;

    size_t total_memory() const override;
    size_t free_memory() const override;

    bool is_available() const override { return available(device_id_); }
    bool supports_unified_memory() const override;

private:
    explicit CudaDevice(int device_id);

    int device_id_;
};

} // namespace sci
