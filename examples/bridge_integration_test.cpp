/**
 * @file bridge_integration_test.cpp
 * @brief 桥接层集成测试 - 验证 vllm/ipc/turbol 接口对接
 */

#include "bridges/ipc_bridge.hpp"
#include "bridges/thread_pool_bridge.hpp"
#include "bridges/pool_bridge.hpp"

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <string>

using namespace sci;
using namespace sci::bridges;

// ============================================================================
// 测试结果统计
// ============================================================================
struct TestStats {
    int passed = 0;
    int failed = 0;
    
    void check(bool condition, const std::string& name) {
        if (condition) {
            std::cout << "  [PASS] " << name << "\n";
            passed++;
        } else {
            std::cout << "  [FAIL] " << name << "\n";
            failed++;
        }
    }
};

TestStats g_stats;

// ============================================================================
// 1. IPC 通信桥接测试 (linux_cpp/ipc)
// ============================================================================
void test_ipc_bridge() {
    std::cout << "\n=== IPC 通信桥接测试 (linux_cpp/ipc) ===\n";
    
    // 测试 SPSC 队列
    {
        SPSCQueue<int> queue;
        g_stats.check(queue.Init(1024), "SPSC 队列 Init");
        
        if (queue.Init(1024)) {
            bool push_ok = queue.Push(42);
            g_stats.check(push_ok, "SPSC 队列 Push");
            
            int value = 0;
            bool pop_ok = queue.Pop(value);
            g_stats.check(pop_ok && value == 42, "SPSC 队列 Pop");
            
            g_stats.check(queue.Empty(), "SPSC 队列 Empty 检查");
        }
        queue.Destroy();
    }
    
    // 测试 MPSC 队列
    {
        MPSCQueue<int> queue;
        g_stats.check(queue.Init(1024), "MPSC 队列 Init");
        
        if (queue.Init(1024)) {
            queue.Push(100);
            g_stats.check(!queue.Empty(), "MPSC 队列非空检查");
            
            int value = 0;
            bool pop_ok = queue.Pop(value);
            g_stats.check(pop_ok && value == 100, "MPSC 队列 Pop");
        }
        queue.Destroy();
    }
    
    // 测试 SpinLock
    {
        SpinLock lock;
        g_stats.check(lock.TryLock(), "SpinLock TryLock");
        lock.Unlock();
        g_stats.check(true, "SpinLock Unlock");
    }
    
    // 测试共享内存段
    {
        std::string shm_name = "/sci_test_shm_" + std::to_string(getpid());
        auto shm = SharedMemorySegment::Create(shm_name, 4096);
        g_stats.check(shm.IsValid(), "共享内存段创建");
        
        if (shm.IsValid()) {
            int* data = static_cast<int*>(shm.BasePtr());
            data[0] = 0x12345678;
            g_stats.check(data[0] == 0x12345678, "共享内存写入");
            
            shm.Destroy();
            g_stats.check(true, "共享内存段销毁");
        }
    }
}

// ============================================================================
// 2. 线程池桥接测试 (linux_cpp/Pool)
// ============================================================================
void test_thread_pool_bridge() {
    std::cout << "\n=== 线程池桥接测试 (linux_cpp/Pool) ===\n";
    
    auto pool = CreateThreadPool(4);
    g_stats.check(pool != nullptr, "线程池创建");
    
    if (pool) {
        g_stats.check(pool->Threads() == 4, "线程池大小检查");
        
        std::atomic<int> counter{0};
        auto f1 = pool->Enqueue([&counter]() {
            counter.fetch_add(1);
            return 42;
        });
        
        g_stats.check(f1.get() == 42, "线程池任务提交和返回值");
        g_stats.check(counter.load() == 1, "线程池任务执行");
        
        // 多任务测试
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 10; ++i) {
            futures.push_back(pool->Enqueue([i]() { return i * i; }));
        }
        
        int sum = 0;
        for (auto& f : futures) {
            sum += f.get();
        }
        g_stats.check(sum == 285, "线程池多任务并发执行");
        
        pool->Shutdown();
        g_stats.check(true, "线程池关闭");
    }
}

// ============================================================================
// 3. 内存池桥接测试 (linux_cpp/Pool)
// ============================================================================
void test_memory_pool_bridge() {
    std::cout << "\n=== 内存池桥接测试 (linux_cpp/Pool) ===\n";
    
    // 固定大小内存池
    {
        auto pool = CreateFixedSizePoolBridge(256, 100);
        g_stats.check(pool != nullptr, "固定大小内存池创建");
        
        if (pool) {
            g_stats.check(pool->block_size() == 256, "块大小检查");
            g_stats.check(pool->block_count() > 0, "块数量检查");
            
            void* ptr1 = pool->allocate(256);
            g_stats.check(ptr1 != nullptr, "内存池分配");
            
            if (ptr1) {
                pool->deallocate(ptr1);
                g_stats.check(true, "内存池释放");
            }
        }
    }
    
    // Slab 内存池
    {
        auto pool = CreateSlabPoolBridge();
        g_stats.check(pool != nullptr, "Slab 内存池创建");
        
        if (pool) {
            void* ptr = pool->allocate(64);
            g_stats.check(ptr != nullptr, "Slab 分配 64B");
            
            if (ptr) {
                pool->deallocate(ptr);
                g_stats.check(true, "Slab 释放");
            }
        }
    }
}

// ============================================================================
// 4. SciComputeInfra 核心接口测试
// ============================================================================
void test_core_interfaces() {
    std::cout << "\n=== SciComputeInfra 核心接口测试 ===\n";
    
    // Device 管理器
    {
        auto device = DeviceManager::Instance().get_device(DeviceType::kCPU, 0);
        g_stats.check(device != nullptr, "Device 管理器获取 CPU 设备");
        
        if (device) {
            std::cout << "  [INFO] Device: " << device->name() << "\n";
        }
    }
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║      SciComputeInfra 桥接层集成测试                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n";
    
    test_ipc_bridge();
    test_thread_pool_bridge();
    test_memory_pool_bridge();
    test_core_interfaces();
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "测试结果: " << g_stats.passed << " 通过, " << g_stats.failed << " 失败\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    return g_stats.failed > 0 ? 1 : 0;
}
