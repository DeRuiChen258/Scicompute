/**
 * @file bridge_demo.cpp
 * @brief 桥接层演示程序
 *
 * 演示如何使用桥接层连接新旧项目
 */

#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include "bridges/pool_bridge.hpp"
#include "bridges/thread_pool_bridge.hpp"
#include "bridges/ipc_bridge.hpp"

using namespace sci::bridges;

void demo_memory_pool() {
    std::cout << "\n=== Memory Pool Bridge Demo ===\n";

    // 创建固定大小内存池
    auto pool = CreateDefaultTensorPool(256);

    // 分配测试
    void* ptr1 = pool->allocate(256);
    void* ptr2 = pool->allocate(256);

    std::cout << "Allocated ptr1: " << ptr1 << "\n";
    std::cout << "Allocated ptr2: " << ptr2 << "\n";

    // 释放
    pool->deallocate(ptr1, 256);
    pool->deallocate(ptr2, 256);

    std::cout << "Deallocated both blocks\n";

    // Slab 池测试
    auto slab = CreateSlabPool();
    void* p1 = slab->allocate(100);  // -> 128
    void* p2 = slab->allocate(300);  // -> 512

    std::cout << "Slab allocate 100 -> " << slab->get_slab_size(100) << " bytes\n";
    std::cout << "Slab allocate 300 -> " << slab->get_slab_size(300) << " bytes\n";

    slab->deallocate(p1, 100);
    slab->deallocate(p2, 300);
}

void demo_thread_pool() {
    std::cout << "\n=== Thread Pool Bridge Demo ===\n";

    // 创建线程池
    auto pool = CreateThreadPool(4);
    std::cout << "Created thread pool with " << pool->Threads() << " threads\n";

    // 提交任务
    std::vector<std::future<int>> results;
    for (int i = 0; i < 8; ++i) {
        auto fut = pool->Enqueue([i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            return i * i;
        });
        results.push_back(std::move(fut));
    }

    // 等待完成
    pool->WaitAll();
    std::cout << "All tasks completed. Results: ";
    for (auto& fut : results) {
        std::cout << fut.get() << " ";
    }
    std::cout << "\n";
    std::cout << "Total completed: " << pool->Completed() << "\n";
}

void demo_ipc() {
    std::cout << "\n=== IPC Bridge Demo ===\n";

    // 创建 SPSC 队列
    auto queue = CreateSPSCQueue<int>(100);
    queue->Init(100);
    std::cout << "Created SPSC queue with capacity 100\n";

    // 生产者
    for (int i = 0; i < 5; ++i) {
        queue->Push(i * 10);
    }
    std::cout << "Pushed 5 items\n";

    // 消费者
    int value;
    int count = 0;
    while (queue->Pop(value)) {
        std::cout << "Popped: " << value << "\n";
        count++;
    }
    std::cout << "Popped " << count << " items\n";
}

int main() {
    std::cout << "SciComputeInfra Bridge Demo\n";
    std::cout << "=========================\n";

    demo_memory_pool();
    demo_thread_pool();
    demo_ipc();

    std::cout << "\n=== All Demos Complete ===\n";
    return 0;
}
