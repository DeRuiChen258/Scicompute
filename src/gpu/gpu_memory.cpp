/**
 * @file gpu_memory.cpp - GPU内存分配器实现 (统一共享内存)
 */

#include "gpu/gpu_memory.hpp"
#include <cstring>
#include <cstdio>

namespace sci {
namespace gpu {

#ifdef SCI_USE_CUDA

CudaAllocator::CudaAllocator(int device_id) : device_id_(device_id) {
    set_device();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    device_info_.device_id = device_id_;
    strncpy(device_info_.name, prop.name, 255);
    device_info_.total_global_mem = prop.totalGlobalMem;
    device_info_.total_const_mem = prop.totalConstMem;
    device_info_.shared_mem_per_block = prop.sharedMemPerBlock;
    device_info_.warp_size = prop.warpSize;
    device_info_.max_threads_per_block = prop.maxThreadsPerBlock;
    device_info_.multiprocessor_count = prop.multiProcessorCount;
    device_info_.compute_version_major = prop.major;
    device_info_.compute_version_minor = prop.minor;
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, stream_));
    printf("[INFO] GPU %d initialized: %s (%lu MB)\n", device_id_, prop.name, prop.totalGlobalMem / 1024 / 1024);
}

CudaAllocator::~CudaAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (cublas_handle_) cublasDestroy(cublas_handle_);
    if (stream_) cudaStreamDestroy(stream_);
    for (auto& pair : managed_map_) cudaFree(pair.first);
    for (auto& pair : device_map_) cudaFree(pair.first);
    for (auto& pair : pinned_map_) cudaFreeHost(pair.first);
}

void CudaAllocator::set_device() const {
    int current_device;
    cudaGetDevice(&current_device);
    if (current_device != device_id_) cudaSetDevice(device_id_);
}

void CudaAllocator::update_stats(size_t bytes, bool allocate) {
    if (allocate) {
        stats_.num_allocations++;
        stats_.current_bytes += bytes;
        stats_.total_allocated_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    } else {
        stats_.num_deallocations++;
        stats_.current_bytes -= bytes;
    }
}

void* CudaAllocator::allocate_managed(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_device();
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA managed malloc failed: %lu bytes - %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    managed_map_[ptr] = bytes;
    update_stats(bytes, true);
    return ptr;
}

void CudaAllocator::deallocate_managed(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    set_device();
    auto it = managed_map_.find(ptr);
    if (it != managed_map_.end()) {
        cudaFree(ptr);
        update_stats(it->second, false);
        managed_map_.erase(it);
    } else {
        fprintf(stderr, "[WARNING] Double free or invalid managed pointer: %p\n", ptr);
    }
}

void* CudaAllocator::allocate_device(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    set_device();
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA malloc failed: %lu bytes - %s\n", bytes, cudaGetErrorString(err));
        return nullptr;
    }
    device_map_[ptr] = bytes;
    update_stats(bytes, true);
    return ptr;
}

void CudaAllocator::deallocate_device(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    set_device();
    auto it = device_map_.find(ptr);
    if (it != device_map_.end()) {
        cudaFree(ptr);
        update_stats(it->second, false);
        device_map_.erase(it);
    }
}

void* CudaAllocator::allocate_pinned(size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] CUDA pinned malloc failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }
    pinned_map_[ptr] = bytes;
    update_stats(bytes, true);
    return ptr;
}

void CudaAllocator::deallocate_pinned(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pinned_map_.find(ptr);
    if (it != pinned_map_.end()) {
        cudaFreeHost(ptr);
        update_stats(it->second, false);
        pinned_map_.erase(it);
    } else {
        fprintf(stderr, "[WARNING] Double free or invalid pinned pointer: %p\n", ptr);
    }
}

void CudaAllocator::copy_h2d(void* dst, const void* src, size_t bytes) {
    set_device();
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void CudaAllocator::copy_d2h(void* dst, const void* src, size_t bytes) {
    set_device();
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void CudaAllocator::copy_d2d(void* dst, const void* src, size_t bytes) {
    set_device();
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

void CudaAllocator::copy_managed(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

void CudaAllocator::synchronize() {
    set_device();
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

CudaAllocator& CudaAllocator::Instance(int device_id) {
    static CudaAllocator instance(device_id);
    return instance;
}

#endif // SCI_USE_CUDA

} // namespace gpu
} // namespace sci
