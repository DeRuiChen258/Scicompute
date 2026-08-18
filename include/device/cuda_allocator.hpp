#pragma once
/**
 * @file cuda_allocator.hpp
 * @brief CUDA 设备内存分配器 - sci::Allocator 接口的 GPU 实现
 * @date 2026-08-18
 */

#include "../memory/allocator.hpp"
#include "../gpu/gpu_memory.hpp"
#include "cuda_device.hpp"

namespace sci {

class CudaDeviceAllocator final : public Allocator {
public:
    static CudaDeviceAllocator& Instance(int device_id = 0) {
        static CudaDeviceAllocator instance(device_id);
        return instance;
    }

    static bool available(int device_id = 0) {
        return CudaDevice::available(device_id);
    }

    void* allocate(size_t bytes) override {
        if (!available(device_id_)) return nullptr;
        return gpu::CudaAllocator::Instance(device_id_).allocate_device(bytes);
    }

    void deallocate(void* ptr) override {
        if (!ptr || !available(device_id_)) return;
        gpu::CudaAllocator::Instance(device_id_).deallocate_device(ptr);
    }

    size_t allocated_bytes() const override {
        if (!available(device_id_)) return 0;
        return gpu::CudaAllocator::Instance(device_id_).stats().current_bytes;
    }

    size_t total_bytes() const override {
        if (!available(device_id_)) return 0;
        return gpu::CudaAllocator::Instance(device_id_).stats().total_allocated_bytes;
    }

    void clear() override {
        // 分配的内存由 BufferHandle 等 RAII 封装管理, 释放时自动归还;
        // CudaAllocator 自身维护分配统计, 无需整体清空。
    }

    Stats stats() const override {
        if (!available(device_id_)) return {};
        auto s = gpu::CudaAllocator::Instance(device_id_).stats();
        Stats out{};
        out.num_allocations = s.num_allocations;
        out.num_deallocations = s.num_deallocations;
        out.current_bytes = s.current_bytes;
        out.peak_bytes = s.peak_bytes;
        out.total_bytes = s.total_allocated_bytes;
        return out;
    }

private:
    explicit CudaDeviceAllocator(int device_id) : device_id_(device_id) {}

    int device_id_;
};

} // namespace sci
