#include "device/device.hpp"
#include "device/cuda_device.hpp"
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

// ============================================================================
// DeviceManager 实现 (热插拔支持)
// ============================================================================

DeviceManager::DeviceManager() 
    : cpu_device(std::make_shared<CPUDevice>())
    , default_device(cpu_device)
{
    // 注册CPU设备
    devices_[next_device_id_++] = cpu_device;
}

DeviceManager& DeviceManager::Instance() {
    static DeviceManager instance;
    return instance;
}

std::shared_ptr<Device> DeviceManager::get_device(DeviceType type, int id) {
    if (type == DeviceType::kCPU) {
        return cpu_device;
    }
    // 查找其他类型设备
    for (auto& [dev_id, device] : devices_) {
        if (device->type() == type && device->id() == id) {
            return device;
        }
    }
    return nullptr;
}

std::shared_ptr<Device> DeviceManager::get_default_device() {
    return default_device;
}

void DeviceManager::set_default_device(DeviceType type, int id) {
    default_type_ = type;
    default_id_ = id;
    default_device = get_device(type, id);
}

DeviceId DeviceManager::RegisterDevice(std::shared_ptr<Device> device) {
    if (!device) return 0;
    
    DeviceId id = next_device_id_++;
    devices_[id] = device;
    
    NotifyDeviceEvent(DeviceEvent::kDeviceAdded, device);
    
    return id;
}

bool DeviceManager::UnregisterDevice(DeviceId id) {
    auto it = devices_.find(id);
    if (it == devices_.end() || it->second == cpu_device) {
        // 不能注销CPU设备
        return false;
    }
    
    auto device = it->second;
    devices_.erase(it);
    
    NotifyDeviceEvent(DeviceEvent::kDeviceRemoved, device);
    
    if (default_device == device) {
        default_device = cpu_device;
        default_type_ = DeviceType::kCPU;
        default_id_ = 0;
    }
    
    return true;
}

std::shared_ptr<Device> DeviceManager::GetDevice(DeviceId id) {
    auto it = devices_.find(id);
    if (it != devices_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Device>> DeviceManager::all_devices() {
    std::vector<std::shared_ptr<Device>> result;
    for (auto& [id, device] : devices_) {
        result.push_back(device);
    }
    return result;
}

size_t DeviceManager::num_devices(DeviceType type) {
    size_t count = 0;
    for (auto& [id, device] : devices_) {
        if (device->type() == type) {
            count++;
        }
    }
    return count;
}

DeviceId DeviceManager::AddCallback(DeviceCallback callback) {
    DeviceId id = next_callback_id_++;
    callbacks_.push_back(std::move(callback));
    return id;
}

bool DeviceManager::RemoveCallback(DeviceId callback_id) {
    if (callback_id > 0 && callback_id < next_callback_id_) {
        callbacks_.erase(callbacks_.begin() + (callback_id - 1));
        return true;
    }
    return false;
}

void DeviceManager::ScanDevices() {
    // 自动检测可用设备
    // 在支持CUDA的环境中会检测GPU设备
#ifdef SCI_USE_CUDA
    if (num_devices(DeviceType::kCUDA) == 0) {
        for (int i = 0; i < 64; ++i) {
            auto device = CudaDevice::Create(i);
            if (!device) {
                break;
            }
            RegisterDevice(std::move(device));
        }
    }
#endif
}

void DeviceManager::NotifyDeviceEvent(DeviceEvent event, std::shared_ptr<Device> device) {
    for (auto& callback : callbacks_) {
        callback(event, device);
    }
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
