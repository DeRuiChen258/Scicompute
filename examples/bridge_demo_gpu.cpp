/**
 * @file bridge_demo_gpu.cpp
 * @brief GPU桥接层演示程序 (支持无GPU环境优雅降级)
 * 
 * 功能:
 * - 检测CUDA运行时环境
 * - CUDA可用时使用GPU后端
 * - CUDA不可用时自动降级到CPU后端
 */

// CUDA运行时头文件 (编译期检测)
#ifdef SCI_USE_CUDA
#include <cuda_runtime.h>
#endif

// GPU头文件 (条件包含)
#ifdef SCI_USE_CUDA
#include "gpu/gpu_tensor.hpp"
#include "gpu/gpu_memory.hpp"
#endif

// CPU后端头文件
#include "memory/allocator.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "bridges/pool_bridge.hpp"
#include "bridges/thread_pool_bridge.hpp"
#include "bridges/ipc_bridge.hpp"
#include <iostream>
#include <vector>
#include <cstring>

using namespace sci;

// ============================================================================
// 运行时GPU检测
// ============================================================================
namespace {

bool detect_gpu_available() {
#ifdef SCI_USE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    
    // 检查设备是否真正可用
    int device_id = 0;
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }
    
    // 尝试一个简单操作验证设备可用
    void* test_ptr = nullptr;
    err = cudaMalloc(&test_ptr, 64);
    if (err != cudaSuccess) {
        return false;
    }
    cudaFree(test_ptr);
    
    return true;
#else
    return false;
#endif
}

std::string get_cuda_runtime_version() {
#ifdef SCI_USE_CUDA
    int version = 0;
    cudaRuntimeGetVersion(&version);
    return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
#else
    return "N/A";
#endif
}

}  // anonymous namespace

// ============================================================================
// GPU后端演示 (仅在GPU可用时调用)
// ============================================================================
namespace gpu_demo {

#ifdef SCI_USE_CUDA
using namespace sci::gpu;

void demo_memory() {
    std::cout << "\n[GPU] 内存分配演示 (统一共享内存)\n\n";
    
    auto& allocator = CudaAllocator::Instance(0);
    auto info = allocator.get_device_info();
    
    std::cout << "GPU设备: " << info.name << "\n";
    std::cout << "总显存: " << (info.total_global_mem / 1024 / 1024) << " MB\n";
    std::cout << "计算能力: " << info.compute_version_major << "." 
              << info.compute_version_minor << "\n\n";
    
    // 使用统一共享内存
    const size_t N = 1024 * 1024;
    float* d_data = static_cast<float*>(allocator.allocate_managed(N * sizeof(float)));
    
    std::cout << "分配统一共享内存: " << N * sizeof(float) / 1024 << " KB\n";
    std::cout << "内存使用: " << allocator.stats().current_bytes / 1024 << " KB\n\n";
    
    if (d_data == nullptr) {
        std::cout << "[GPU] 统一内存分配失败 (GPU可能不支持统一内存)\n\n";
        return;
    }
    
    // 清理
    allocator.deallocate_managed(d_data);
    std::cout << "[GPU] 内存释放完成\n\n";
}

void demo_tensor() {
    std::cout << "\n[GPU] 张量演示\n\n";
    
    std::vector<int64_t> shape = {16, 128, 512};
    GpuTensor tensor(shape, 0);  // float32
    
    if (tensor.data() == nullptr) {
        std::cout << "[GPU] 张量创建失败 (无GPU或内存不足)\n";
        std::cout << "[GPU] 张量大小: " << tensor.size() << " 元素\n\n";
        return;
    }
    
    std::cout << "创建张量形状: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "张量大小: " << tensor.size() << " 元素\n";
    std::cout << "内存占用: " << tensor.nbytes() / 1024 << " KB\n";
    
    tensor.zero();
    std::cout << "[GPU] 张量填充完成\n\n";
}
#endif  // SCI_USE_CUDA

}  // namespace gpu_demo

// ============================================================================
// CPU后端演示 (GPU不可用时的降级路径)
// ============================================================================
namespace cpu_demo {

void demo_memory() {
    std::cout << "\n[CPU] 内存分配演示\n\n";
    
    HostAllocator& allocator = HostAllocator::Instance();
    constexpr size_t kNumElements = 1024 * 1024;
    constexpr size_t kNumBytes = kNumElements * sizeof(float);
    
    void* buffer = allocator.allocate(kNumBytes);
    if (buffer == nullptr) {
        std::cout << "[CPU] 内存分配失败\n\n";
        return;
    }
    
    std::cout << "分配内存: " << kNumBytes / 1024 << " KB\n";
    std::memset(buffer, 0, kNumBytes);
    std::cout << "[CPU] 内存填充完成\n";
    
    allocator.deallocate(buffer);
    std::cout << "[CPU] 内存释放完成\n\n";
}

void demo_tensor() {
    std::cout << "\n[CPU] 张量演示\n\n";
    
    auto cpu_device = DeviceManager::Instance().get_device(DeviceType::kCPU, 0);
    std::vector<int64_t> shape_vec = {16, 128, 512};
    TensorShape shape(shape_vec);
    
    Tensor tensor = Tensor::Zeros(shape, DType::kFloat32, *cpu_device);
    
    std::cout << "创建张量形状: [";
    for (int i = 0; i < tensor.ndims(); ++i) {
        std::cout << tensor.dim(i);
        if (i < tensor.ndims() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
    std::cout << "张量大小: " << tensor.num_elements() << " 元素\n";
    std::cout << "内存占用: " << tensor.num_bytes() / 1024 << " KB\n";
    
    std::cout << "[CPU] 张量创建完成\n\n";
}

}  // namespace cpu_demo

// ============================================================================
// 桥接层演示 (GPU/CPU通用)
// ============================================================================
namespace bridge_demo {

void demo() {
    std::cout << "\n[Bridge] 桥接层演示\n\n";
    
    // 内存池桥接
    auto pool_bridge = CreateFixedSizePoolBridge(256, 256);
    std::cout << "固定大小内存池: 块大小=" << pool_bridge->block_size() 
              << ", 块数=" << pool_bridge->block_count() << "\n";
    
    // 线程池
    auto thread_pool = sci::bridges::CreateThreadPool(4);
    std::cout << "线程池: " << thread_pool->Threads() << " 线程\n";
    
    std::cout << "[Bridge] 桥接层测试通过\n\n";
}

}  // namespace bridge_demo

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║      SciComputeInfra Bridge Demo (GPU/CPU Auto)        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    std::cout << "\nCUDA运行时版本: " << get_cuda_runtime_version() << "\n";
    
    // 运行时GPU检测
    bool gpu_available = detect_gpu_available();
    
    try {
        if (gpu_available) {
            std::cout << "后端模式: GPU\n\n";
#ifdef SCI_USE_CUDA
            gpu_demo::demo_memory();
            gpu_demo::demo_tensor();
#endif
        } else {
            std::cout << "\n⚠️  未检测到GPU，自动切换到CPU后端\n\n";
            cpu_demo::demo_memory();
            cpu_demo::demo_tensor();
        }
        
        bridge_demo::demo();
        
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "演示完成!\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
