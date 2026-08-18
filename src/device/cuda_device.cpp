/**
 * @file cuda_device.cpp
 * @brief CUDA 设备实现 - 基于 gpu::CudaAllocator 的 Device 后端
 * @date 2026-08-18
 */

#include "device/cuda_device.hpp"

#ifdef SCI_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <string>

namespace sci {

namespace {

bool cuda_available(int device_id) {
#ifdef SCI_USE_CUDA
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= device_id) {
        return false;
    }
    cudaDeviceProp prop{};
    return cudaGetDeviceProperties(&prop, device_id) == cudaSuccess;
#else
    (void)device_id;
    return false;
#endif
}

} // namespace

bool CudaDevice::available(int device_id) {
    return cuda_available(device_id);
}

std::shared_ptr<CudaDevice> CudaDevice::Create(int device_id) {
    if (!available(device_id)) {
        return nullptr;
    }
    return std::shared_ptr<CudaDevice>(new CudaDevice(device_id));
}

CudaDevice::CudaDevice(int device_id) : device_id_(device_id) {}

std::string CudaDevice::name() const {
#ifdef SCI_USE_CUDA
    if (available(device_id_)) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id_) == cudaSuccess) {
            return "CUDA:" + std::to_string(device_id_) + ":" + prop.name;
        }
    }
#endif
    return "CUDA:" + std::to_string(device_id_);
}

void* CudaDevice::allocate(size_t bytes) {
    if (!available(device_id_) || bytes == 0) return nullptr;
#ifdef SCI_USE_CUDA
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
#else
    return nullptr;
#endif
}

void CudaDevice::deallocate(void* ptr) {
    if (!ptr) return;
#ifdef SCI_USE_CUDA
    cudaFree(ptr);
#endif
}

void* CudaDevice::allocate_host(size_t bytes) {
    if (!available(device_id_) || bytes == 0) return nullptr;
#ifdef SCI_USE_CUDA
    void* ptr = nullptr;
    if (cudaMallocHost(&ptr, bytes) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
#else
    return nullptr;
#endif
}

void CudaDevice::deallocate_host(void* ptr) {
    if (!ptr) return;
#ifdef SCI_USE_CUDA
    cudaFreeHost(ptr);
#endif
}

void CudaDevice::copy_to_device(void* dst, const void* src, size_t bytes) {
#ifdef SCI_USE_CUDA
    if (!available(device_id_) || !dst || !src) return;
    cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
#else
    (void)dst; (void)src; (void)bytes;
#endif
}

void CudaDevice::copy_to_host(void* dst, const void* src, size_t bytes) {
#ifdef SCI_USE_CUDA
    if (!available(device_id_) || !dst || !src) return;
    cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
#else
    (void)dst; (void)src; (void)bytes;
#endif
}

void CudaDevice::copy_from_device(void* dst, const void* src, size_t bytes) {
    // 语义与 Tensor::to/to_device 对齐: 从 CPU(或其它设备) 拷贝进入本设备
    copy_to_device(dst, src, bytes);
}

void CudaDevice::copy_async(void* dst, const void* src, size_t bytes, Stream& stream) {
#ifdef SCI_USE_CUDA
    if (!available(device_id_) || !dst || !src) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream.handle());
    if (s) {
        cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
    } else {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
#else
    (void)dst; (void)src; (void)bytes; (void)stream;
#endif
}

void CudaDevice::memset(void* ptr, int value, size_t bytes) {
#ifdef SCI_USE_CUDA
    if (!available(device_id_) || !ptr) return;
    cudaMemset(ptr, value, bytes);
#else
    (void)ptr; (void)value; (void)bytes;
#endif
}

void CudaDevice::memset_async(void* ptr, int value, size_t bytes, Stream& stream) {
#ifdef SCI_USE_CUDA
    if (!available(device_id_) || !ptr) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream.handle());
    if (s) {
        cudaMemsetAsync(ptr, value, bytes, s);
    } else {
        cudaMemset(ptr, value, bytes);
    }
#else
    (void)ptr; (void)value; (void)bytes; (void)stream;
#endif
}

void CudaDevice::synchronize() {
#ifdef SCI_USE_CUDA
    if (!available(device_id_)) return;
    cudaDeviceSynchronize();
#endif
}

size_t CudaDevice::total_memory() const {
#ifdef SCI_USE_CUDA
    if (available(device_id_)) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_id_) == cudaSuccess) {
            return prop.totalGlobalMem;
        }
    }
#endif
    return 0;
}

size_t CudaDevice::free_memory() const {
#ifdef SCI_USE_CUDA
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (available(device_id_) &&
        cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        return free_bytes;
    }
#endif
    return 0;
}

bool CudaDevice::supports_unified_memory() const {
#ifdef SCI_USE_CUDA
    if (available(device_id_)) {
        int managed = 0;
        if (cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory,
                                   device_id_) == cudaSuccess) {
            return managed != 0;
        }
    }
#endif
    return false;
}

} // namespace sci
