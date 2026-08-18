#pragma once
/**
 * @file gpu_memory.hpp
 * @brief GPU内存分配器 - CUDA 13.2 GPU内存管理 (支持统一共享内存)
 * @author SciComputeInfra
 * @date 2026-08-17
 * 
 * 特性:
 * - 统一内存 (Managed Memory) - CPU/GPU共享访问
 * - 页锁定内存 (Pinned Memory) - 高带宽传输
 * - cuBLAS/cuSPARSE集成
 */

#include "../core/common.hpp"
#include "../core/logger.hpp"

#ifdef SCI_USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#endif

namespace sci {
namespace gpu {

// ============================================================================
// GPU设备信息
// ============================================================================
struct GpuDeviceInfo {
    int device_id = 0;
    char name[256] = {0};
    size_t total_global_mem = 0;
    size_t total_const_mem = 0;
    int shared_mem_per_block = 0;
    int warp_size = 0;
    int max_threads_per_block = 0;
    int multiprocessor_count = 0;
    int compute_version_major = 0;
    int compute_version_minor = 0;
    int clock_rate = 0;
};

// ============================================================================
// GPU内存分配器接口
// ============================================================================
class GpuAllocator {
public:
    virtual ~GpuAllocator() = default;
    
    // 统一内存 (共享内存) - CPU/GPU均可访问
    virtual void* allocate_managed(size_t bytes) = 0;
    virtual void deallocate_managed(void* ptr) = 0;
    
    // 设备内存 (仅GPU访问)
    virtual void* allocate_device(size_t bytes) = 0;
    virtual void deallocate_device(void* ptr) = 0;
    
    // 页锁定内存 (仅CPU访问，但传输更快)
    virtual void* allocate_pinned(size_t bytes) = 0;
    virtual void deallocate_pinned(void* ptr) = 0;
    
    // 数据传输
    virtual void copy_h2d(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_d2h(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_d2d(void* dst, const void* src, size_t bytes) = 0;
    virtual void copy_managed(void* dst, const void* src, size_t bytes) = 0;
    
    // 同步
    virtual void synchronize() = 0;
    
    // 属性
    virtual size_t total_mem() const = 0;
    virtual int device_id() const = 0;
    virtual GpuDeviceInfo get_device_info() const = 0;
    
protected:
    GpuAllocator() = default;
    SCI_DISALLOW_COPY_AND_MOVE(GpuAllocator);
};

// ============================================================================
// CUDA内存分配器实现
// ============================================================================
#ifdef SCI_USE_CUDA

class CudaAllocator final : public GpuAllocator {
public:
    static CudaAllocator& Instance(int device_id = 0);
    
    // 统一内存 (共享内存)
    void* allocate_managed(size_t bytes) override;
    void deallocate_managed(void* ptr) override;
    
    // 设备内存
    void* allocate_device(size_t bytes) override;
    void deallocate_device(void* ptr) override;
    
    // 页锁定内存
    void* allocate_pinned(size_t bytes) override;
    void deallocate_pinned(void* ptr) override;
    
    // 数据传输
    void copy_h2d(void* dst, const void* src, size_t bytes) override;
    void copy_d2h(void* dst, const void* src, size_t bytes) override;
    void copy_d2d(void* dst, const void* src, size_t bytes) override;
    void copy_managed(void* dst, const void* src, size_t bytes) override;
    
    // 同步
    void synchronize() override;
    
    // 属性
    size_t total_mem() const override { return device_info_.total_global_mem; }
    int device_id() const override { return device_info_.device_id; }
    GpuDeviceInfo get_device_info() const override { return device_info_; }
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle() { return cublas_handle_; }
    
    // CUDA流
    void* stream() { return static_cast<void*>(stream_); }
    void set_stream(void* stream) { stream_ = static_cast<cudaStream_t>(stream); }
    
    // 统计
    struct Stats {
        size_t num_allocations = 0;
        size_t num_deallocations = 0;
        size_t current_bytes = 0;
        size_t peak_bytes = 0;
        size_t total_allocated_bytes = 0;
    };
    
    Stats stats() const { return stats_; }
    void reset_stats() { stats_ = Stats{}; }
    
private:
    CudaAllocator(int device_id);
    ~CudaAllocator();
    
    void set_device() const;
    void update_stats(size_t bytes, bool allocate);
    
    int device_id_;
    GpuDeviceInfo device_info_;
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_handle_ = nullptr;
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> managed_map_;
    std::unordered_map<void*, size_t> device_map_;
    std::unordered_map<void*, size_t> pinned_map_;
    Stats stats_;
};

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#else // SCI_USE_CUDA

// 非CUDA环境下 cublasHandle_t 的占位类型 (与 CUDA 头文件中的指针语义一致)
using cublasHandle_t = void*;

// 非CUDA版本的stub实现
class CudaAllocator final : public GpuAllocator {
public:
    static CudaAllocator& Instance(int device_id = 0) {
        static CudaAllocator instance(device_id);
        return instance;
    }
    
    void* allocate_managed(size_t bytes) override { return nullptr; }
    void deallocate_managed(void* ptr) override {}
    void* allocate_device(size_t bytes) override { return nullptr; }
    void deallocate_device(void* ptr) override {}
    void* allocate_pinned(size_t bytes) override { return nullptr; }
    void deallocate_pinned(void* ptr) override {}
    void copy_h2d(void* dst, const void* src, size_t bytes) override {}
    void copy_d2h(void* dst, const void* src, size_t bytes) override {}
    void copy_d2d(void* dst, const void* src, size_t bytes) override {}
    void copy_managed(void* dst, const void* src, size_t bytes) override {}
    void synchronize() override {}
    size_t total_mem() const override { return 0; }
    int device_id() const override { return -1; }
    GpuDeviceInfo get_device_info() const override { return GpuDeviceInfo{}; }
    struct Stats {
        size_t num_allocations = 0;
        size_t num_deallocations = 0;
        size_t current_bytes = 0;
        size_t peak_bytes = 0;
        size_t total_allocated_bytes = 0;
    };
    Stats stats() const { return {}; }
    cublasHandle_t cublas_handle() { return nullptr; }
    void* stream() { return nullptr; }
    void set_stream(void*) {}
    
private:
    CudaAllocator(int) {}
};

#endif // SCI_USE_CUDA

} // namespace gpu
} // namespace sci
