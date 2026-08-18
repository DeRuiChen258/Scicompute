/**
 * @file gpu_scheduler.cpp
 * @brief GPU调度器实现 - 使用统一共享内存
 */

#include "gpu/gpu_scheduler.hpp"
#include "gpu/gpu_memory.hpp"
#include <thread>
#include <chrono>
#include <cstring>
#include <cstdio>

namespace sci {
namespace gpu {

#ifdef SCI_USE_CUDA

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

MemcpyTask::MemcpyTask(void* dst, const void* src, size_t bytes, Kind kind)
    : GpuTask(Type::kMemcpy), dst_(dst), src_(src), bytes_(bytes), kind_(kind) {}

void MemcpyTask::execute() {
    switch (kind_) {
        case Kind::HostToDevice:
            CUDA_CHECK(cudaMemcpy(dst_, src_, bytes_, cudaMemcpyHostToDevice)); break;
        case Kind::DeviceToHost:
            CUDA_CHECK(cudaMemcpy(dst_, src_, bytes_, cudaMemcpyDeviceToHost)); break;
        case Kind::DeviceToDevice:
            CUDA_CHECK(cudaMemcpy(dst_, src_, bytes_, cudaMemcpyDeviceToDevice)); break;
        case Kind::HostToHost:
            std::memcpy(dst_, src_, bytes_); break;
    }
}

KernelTask::KernelTask(KernelFn fn, const std::string& name)
    : GpuTask(Type::kKernel), kernel_fn_(std::move(fn)), name_(name) {}

void KernelTask::execute() {
    if (kernel_fn_) { kernel_fn_(stream_); }
}

GemmTask::GemmTask(const GpuTensor& A, const GpuTensor& B, GpuTensor& C,
                   float alpha, float beta, bool trans_a, bool trans_b)
    : GpuTask(Type::kGemm), A_(A), B_(B), C_(C),
      alpha_(alpha), beta_(beta), trans_a_(trans_a), trans_b_(trans_b) {}

void GemmTask::execute() {
    auto& allocator = CudaAllocator::Instance();
    auto handle = allocator.cublas_handle();
    int m = static_cast<int>(C_.shape()[0]);
    int n = static_cast<int>(C_.shape()[1]);
    int k = trans_a_ ? static_cast<int>(A_.shape()[0]) : static_cast<int>(A_.shape()[1]);
    cublasOperation_t transa = trans_a_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transb = trans_b_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLAS_CHECK(cublasSgemm(handle, transb, transa, n, m, k, &alpha_,
                             static_cast<const float*>(B_.data()), trans_b_ ? n : k,
                             static_cast<const float*>(A_.data()), trans_a_ ? k : m,
                             &beta_, static_cast<float*>(C_.data()), m));
}

GpuScheduler::GpuScheduler() : initialized_(false) {}
GpuScheduler::~GpuScheduler() { shutdown(); }

GpuScheduler& GpuScheduler::Instance() {
    static GpuScheduler instance;
    return instance;
}

void GpuScheduler::initialize(int device_id) {
    device_id_ = device_id;
    CUDA_CHECK(cudaSetDevice(device_id_));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id_));
    device_info_.device_id = device_id_;
    strncpy(device_info_.name, prop.name, 255);
    device_info_.total_global_mem = prop.totalGlobalMem;
    device_info_.compute_version_major = prop.major;
    device_info_.compute_version_minor = prop.minor;
    CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&current_stream_)));
    initialized_ = true;
    running_ = true;
    worker_thread_ = std::thread([this]() {
        while (running_) {
            process_queue();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    printf("[INFO] GPU Scheduler initialized on device %d: %s\n", device_id_, prop.name);
}

void GpuScheduler::shutdown() {
    if (running_) {
        running_ = false;
        synchronize();
        for (auto stream : streams_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        }
        streams_.clear();
        if (worker_thread_.joinable()) { worker_thread_.join(); }
        initialized_ = false;
        printf("[INFO] GPU Scheduler shutdown\n");
    }
}

void GpuScheduler::submit(std::shared_ptr<GpuTask> task) {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    task->set_stream(current_stream_);
    task_queue_.push_back(std::move(task));
}

void GpuScheduler::synchronize() {
    if (current_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(current_stream_)));
    }
}

void GpuScheduler::process_queue() {
    std::shared_ptr<GpuTask> task;
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!task_queue_.empty()) {
            task = std::move(task_queue_.front());
            task_queue_.pop_front();
        }
    }
    if (task) { task->execute(); }
}

void* GpuScheduler::create_stream(int priority) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
    streams_.push_back(static_cast<void*>(stream));
    return static_cast<void*>(stream);
}

void GpuScheduler::destroy_stream(void* stream) {
    if (stream) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        auto it = std::find(streams_.begin(), streams_.end(), stream);
        if (it != streams_.end()) { streams_.erase(it); }
    }
}

void GpuScheduler::set_stream(void* stream) { current_stream_ = stream; }
CudaAllocator& GpuScheduler::allocator() { return CudaAllocator::Instance(); }
void GpuScheduler::memcpy_h2d(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
void GpuScheduler::memcpy_d2h(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
void GpuScheduler::memcpy_d2d(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice));
}

#else // SCI_USE_CUDA

GpuScheduler::GpuScheduler() : initialized_(false) {}
GpuScheduler::~GpuScheduler() {}
GpuScheduler& GpuScheduler::Instance() { static GpuScheduler instance; return instance; }
void GpuScheduler::initialize(int) { initialized_ = true; }
void GpuScheduler::shutdown() { initialized_ = false; }
void GpuScheduler::submit(std::shared_ptr<GpuTask>) {}
void GpuScheduler::synchronize() {}
void GpuScheduler::process_queue() {}
void* GpuScheduler::create_stream(int) { return nullptr; }
void GpuScheduler::destroy_stream(void*) {}
void GpuScheduler::set_stream(void*) {}
CudaAllocator& GpuScheduler::allocator() { return CudaAllocator::Instance(); }
void GpuScheduler::memcpy_h2d(void*, const void*, size_t) {}
void GpuScheduler::memcpy_d2h(void*, const void*, size_t) {}
void GpuScheduler::memcpy_d2d(void*, const void*, size_t) {}

#endif // SCI_USE_CUDA

} // namespace gpu
} // namespace sci
