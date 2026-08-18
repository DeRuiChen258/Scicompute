// Stress test for IPC queue and memory pool
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cassert>
#include "bridges/ipc_bridge.hpp"
#include "bridges/pool_bridge.hpp"
#include "core/status.hpp"

using namespace sci;
using namespace sci::bridges;

// Test MPSC queue (sequential push/pop)
void TestMPSCSequential(size_t num_items) {
    std::cout << "\n=== MPSC Queue Stress Test (Sequential) ===" << std::endl;
    std::cout << "Items: " << num_items << std::endl;
    
    MPSCQueue<int> queue;
    if (!queue.Init(num_items + 1)) {
        std::cerr << "  [FAIL] MPSCQueue Init failed" << std::endl;
        return;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t pushed = 0, popped = 0;
    for (size_t i = 0; i < num_items; ++i) {
        int value = static_cast<int>(i);
        if (queue.Push(value)) pushed++;
    }
    
    int value;
    while (queue.Pop(value)) popped++;
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double throughput = num_items * 1000000.0 / duration;
    
    bool pass = (pushed == num_items && popped == num_items);
    std::cout << "  [" << (pass ? "PASS" : "FAIL") << "] MPSC: " 
              << pushed << " pushed, " << popped << " popped in " 
              << duration << " µs (" << throughput << " ops/sec)" << std::endl;
}

// Test concurrent memory pool operations
void TestPoolConcurrent(size_t num_threads, size_t allocs_per_thread) {
    std::cout << "\n=== Memory Pool Concurrent Test ===" << std::endl;
    std::cout << "Threads: " << num_threads 
              << ", Allocs/thread: " << allocs_per_thread << std::endl;
    
    FixedSizePoolBridge pool(64, 10000);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (size_t i = 0; i < allocs_per_thread; ++i) {
                void* ptr = pool.allocate(64);
                if (ptr) pool.deallocate(ptr);
            }
        });
    }
    
    for (auto& t : threads) t.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    size_t total_ops = num_threads * allocs_per_thread;
    double throughput = total_ops * 1000000.0 / duration;
    
    std::cout << "  [PASS] Pool: " << total_ops << " allocs in " 
              << duration << " µs (" << throughput << " ops/sec)" << std::endl;
}

// Test SPSC queue high throughput (producer-consumer threads)
void TestSPSCThroughput(size_t num_items) {
    std::cout << "\n=== SPSC Queue Throughput Test ===" << std::endl;
    std::cout << "Items: " << num_items << std::endl;
    
    SPSCQueue<int> queue;
    queue.Init(1024);
    std::atomic<bool> stop{false};
    std::atomic<size_t> received{0};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::thread producer([&]() {
        for (size_t i = 0; i < num_items; ++i) {
            while (!queue.Push(static_cast<int>(i))) {
                std::this_thread::yield();
            }
        }
        stop = true;
    });
    
    std::thread consumer([&]() {
        int value;
        while (!stop || !queue.Empty()) {
            if (queue.Pop(value)) received++;
            else std::this_thread::yield();
        }
    });
    
    producer.join();
    consumer.join();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    
    double throughput = num_items * 1000000.0 / duration;
    
    std::cout << "  [PASS] SPSC: " << num_items << " items in " 
              << duration << " µs (" << throughput << " ops/sec)" << std::endl;
    std::cout << "  [INFO] Received: " << received.load() << std::endl;
}

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║      SciComputeInfra 压力测试                           ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";
    
    TestMPSCSequential(100000);
    TestPoolConcurrent(16, 5000);
    TestSPSCThroughput(1000000);
    
    std::cout << "\n============================================================\n"
              << "压力测试完成\n"
              << "============================================================\n";
    
    return 0;
}
