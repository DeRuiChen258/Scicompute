#include "device/device.hpp"
#include <cstdlib>
#include <cstring>
#include <sys/sysinfo.h>

namespace sci {

CPUDevice& CPUDevice::Instance() {
    static CPUDevice instance;
    return instance;
}

void* CPUDevice::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    void* ptr = nullptr;
    
#if defined(__linux__)
    if (posix_memalign(&ptr, kDefaultAlignment, bytes) != 0) {
        return nullptr;
    }
#else
    ptr = std::malloc(bytes);
#endif
    
    if (ptr) {
        std::memset(ptr, 0, bytes);
    }
    return ptr;
}

void CPUDevice::deallocate(void* ptr) {
    if (ptr) {
        std::free(ptr);
    }
}

void CPUDevice::copy_to_device(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void CPUDevice::copy_to_host(void* dst, const void* src, size_t bytes) {
    std::memcpy(dst, src, bytes);
}

void CPUDevice::memset(void* ptr, int value, size_t bytes) {
    std::memset(ptr, value, bytes);
}

size_t CPUDevice::total_memory() const {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.totalram;
    }
    return 0;
}

size_t CPUDevice::free_memory() const {
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        return info.freeram;
    }
    return 0;
}

DeviceManager::DeviceManager() : cpu_device(std::make_shared<CPUDevice>()) {}

DeviceManager& DeviceManager::Instance() {
    static DeviceManager instance;
    return instance;
}

std::shared_ptr<Device> DeviceManager::get_device(DeviceType type, int id) {
    if (type == DeviceType::kCPU) {
        return cpu_device;
    }
    return nullptr;
}

std::shared_ptr<Device> DeviceManager::get_default_device() {
    return get_device(DeviceType::kCPU, 0);
}

void DeviceManager::set_default_device(DeviceType type, int id) {}

void DeviceManager::register_device(std::shared_ptr<Device> device) {}

std::vector<std::shared_ptr<Device>> DeviceManager::all_devices() {
    std::vector<std::shared_ptr<Device>> devices;
    devices.push_back(get_device(DeviceType::kCPU, 0));
    return devices;
}

size_t DeviceManager::num_devices(DeviceType type) {
    if (type == DeviceType::kCPU) return 1;
    return 0;
}

std::shared_ptr<Device> GetCPUDevice() {
    return DeviceManager::Instance().get_device(DeviceType::kCPU, 0);
}

// Default implementations for async operations (CPU)
void Device::copy_async(void* dst, const void* src, size_t bytes, Stream& stream) {
    std::memcpy(dst, src, bytes);
}

void Device::memset_async(void* ptr, int value, size_t bytes, Stream& stream) {
    std::memset(ptr, value, bytes);
}

} // namespace sci
