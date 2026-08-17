#pragma once

#include "../core/common.hpp"
#include "stream.hpp"
#include <memory>
#include <vector>

namespace sci {

class Device;
class Stream;
class Event;

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

class DeviceManager {
public:
    static DeviceManager& Instance();
    
    std::shared_ptr<Device> get_device(DeviceType type, int id = 0);
    std::shared_ptr<Device> get_default_device();
    void set_default_device(DeviceType type, int id = 0);
    void register_device(std::shared_ptr<Device> device);
    std::vector<std::shared_ptr<Device>> all_devices();
    size_t num_devices(DeviceType type);

private:
    DeviceManager();
    
    std::shared_ptr<Device> cpu_device;
};

std::shared_ptr<Device> GetCPUDevice();

} // namespace sci
