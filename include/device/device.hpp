#pragma once

#include "../core/common.hpp"
#include "stream.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <unordered_map>

namespace sci {

class Device;
class Stream;
class Event;

// ============================================================================
// 设备事件类型
// ============================================================================
enum class DeviceEvent {
    kDeviceAdded,
    kDeviceRemoved,
    kDeviceError
};

// ============================================================================
// Device 基类
// ============================================================================
class Device {
public:
    virtual ~Device() = default;
    
    virtual DeviceType type() const = 0;
    virtual int id() const { return 0; }
    virtual std::string name() const = 0;
    
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual void* allocate_host(size_t bytes) { return nullptr; }
    virtual void deallocate_host(void* ptr) {}
    
    virtual void copy_to_device(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_to_host(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_from_device(void* dst, const void* src, size_t bytes) {
        copy_to_host(dst, src, bytes);
    }
    virtual void copy_async(void* dst, const void* src, size_t bytes, Stream& stream);
    
    virtual void memset(void* ptr, int value, size_t bytes) = 0;
    virtual void memset_async(void* ptr, int value, size_t bytes, Stream& stream);
    
    virtual void synchronize() = 0;
    
    virtual size_t total_memory() const { return 0; }
    virtual size_t free_memory() const { return 0; }
    
    virtual bool is_available() const { return true; }
    virtual bool supports_unified_memory() const { return false; }

protected:
    Device() = default;
    Device(const Device&) = default;
    Device& operator=(const Device&) = default;
};

// ============================================================================
// CPUDevice
// ============================================================================
class CPUDevice final : public Device {
public:
    static CPUDevice& Instance();
    
    DeviceType type() const override { return DeviceType::kCPU; }
    int id() const override { return -1; }
    std::string name() const override { return "CPU"; }
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
    void copy_to_device(void* dst, const void* src, size_t bytes) override;
    void copy_to_host(void* dst, const void* src, size_t bytes) override;
    
    void memset(void* ptr, int value, size_t bytes) override;
    
    void synchronize() override {}
    
    size_t total_memory() const override;
    size_t free_memory() const override;
    
    CPUDevice() = default;
};

// ============================================================================
// 设备变化回调
// ============================================================================
using DeviceCallback = std::function<void(DeviceEvent, std::shared_ptr<Device>)>;
using DeviceId = uint32_t;

// ============================================================================
// DeviceManager (支持热插拔)
// ============================================================================
class DeviceManager {
public:
    static DeviceManager& Instance();
    
    // 设备获取
    std::shared_ptr<Device> get_device(DeviceType type, int id = 0);
    std::shared_ptr<Device> get_default_device();
    
    // 默认设备设置
    void set_default_device(DeviceType type, int id = 0);
    
    // 热插拔设备管理
    DeviceId RegisterDevice(std::shared_ptr<Device> device);
    bool UnregisterDevice(DeviceId id);
    std::shared_ptr<Device> GetDevice(DeviceId id);
    
    // 设备枚举
    std::vector<std::shared_ptr<Device>> all_devices();
    size_t num_devices(DeviceType type);
    
    // 事件回调
    DeviceId AddCallback(DeviceCallback callback);
    bool RemoveCallback(DeviceId callback_id);
    
    // 自动检测 (扫描可用设备)
    void ScanDevices();
    
private:
    DeviceManager();
    
    DeviceId next_device_id_ = 1;
    DeviceId next_callback_id_ = 1;
    
    std::shared_ptr<Device> cpu_device;
    std::shared_ptr<Device> default_device;
    DeviceType default_type_ = DeviceType::kCPU;
    int default_id_ = 0;
    
    std::unordered_map<DeviceId, std::shared_ptr<Device>> devices_;
    std::vector<DeviceCallback> callbacks_;
    
    void NotifyDeviceEvent(DeviceEvent event, std::shared_ptr<Device> device);
};

std::shared_ptr<Device> GetCPUDevice();

} // namespace sci

// ============================================================================
// 设备状态回调接口
