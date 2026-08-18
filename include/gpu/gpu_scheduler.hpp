#pragma once
/**
 * @file gpu_scheduler.hpp
 * @brief GPU调度器 - 管理GPU计算任务
 * @date 2026-08-17
 */

#include "gpu_memory.hpp"
#include "gpu_tensor.hpp"
#include <deque>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>

#ifdef SCI_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace sci {
namespace gpu {

// ============================================================================
// GPU计算任务基类
// ============================================================================
class GpuTask {
public:
    enum class Type {
        kMemcpy,
        kKernel,
        kGemm,
        kSync,
        kCustom
    };
    
    GpuTask(Type type) : type_(type) {}
    virtual ~GpuTask() = default;
    
    virtual void execute() = 0;
    
    Type type() const { return type_; }
    void set_stream(void* stream) { stream_ = stream; }
    void* stream() const { return stream_; }
    
protected:
    Type type_;
    void* stream_{nullptr};
};

// ============================================================================
// 内存拷贝任务
// ============================================================================
class MemcpyTask : public GpuTask {
public:
    enum class Kind { HostToDevice, DeviceToHost, DeviceToDevice, HostToHost };
    
    MemcpyTask(void* dst, const void* src, size_t bytes, Kind kind);
    
    void execute() override;
    
private:
    void* dst_;
    const void* src_;
    size_t bytes_;
    Kind kind_;
};

// ============================================================================
// GPU Kernel任务
// ============================================================================
class KernelTask : public GpuTask {
public:
    using KernelFn = std::function<void(void*)>;
    
    KernelTask(KernelFn fn, const std::string& name = "Kernel");
    
    void execute() override;
    void set_kernel_fn(KernelFn fn) { kernel_fn_ = std::move(fn); }
    
private:
    KernelFn kernel_fn_;
    std::string name_;
};

// ============================================================================
// GEMM任务
// ============================================================================
class GemmTask : public GpuTask {
public:
    GemmTask(const GpuTensor& A, const GpuTensor& B, GpuTensor& C,
             float alpha = 1.0f, float beta = 0.0f,
             bool trans_a = false, bool trans_b = false);
    
    void execute() override;
    
private:
    const GpuTensor& A_;
    const GpuTensor& B_;
    GpuTensor& C_;
    float alpha_;
    float beta_;
    bool trans_a_;
    bool trans_b_;
};

// ============================================================================
// GPU调度器
// ============================================================================
class GpuScheduler {
public:
    static GpuScheduler& Instance();
    void initialize(int device_id = 0);
    void shutdown();
    void submit(std::shared_ptr<GpuTask> task);
    void synchronize();
    void* create_stream(int priority = 0);
    void destroy_stream(void* stream);
    void* current_stream() const { return current_stream_; }
    void set_stream(void* stream);
    const GpuDeviceInfo& device_info() const { return device_info_; }
    CudaAllocator& allocator();
    void memcpy_h2d(void* dst, const void* src, size_t bytes);
    void memcpy_d2h(void* dst, const void* src, size_t bytes);
    void memcpy_d2d(void* dst, const void* src, size_t bytes);
    
private:
    GpuScheduler();
    ~GpuScheduler();
    GpuScheduler(const GpuScheduler&) = delete;
    GpuScheduler& operator=(const GpuScheduler&) = delete;
    void process_queue();
    
    int device_id_{0};
    GpuDeviceInfo device_info_;
    void* current_stream_{nullptr};
    std::deque<std::shared_ptr<GpuTask>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::vector<void*> streams_;
    bool running_{false};
    std::thread worker_thread_;
    std::atomic<bool> initialized_{false};
};

} // namespace gpu
} // namespace sci
