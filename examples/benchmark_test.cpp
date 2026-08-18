// Performance benchmark for IPC queue and memory pool
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include "bridges/ipc_bridge.hpp"
#include "bridges/pool_bridge.hpp"
#include "benchmark/benchmark_runner.hpp"

using namespace sci;
using namespace sci::bridges;
using namespace sci::benchmark;

struct BenchmarkResult {
    std::string name;
    double total_us;
    size_t iterations;
    double ops_per_sec;
    double ns_per_op;
};

// Benchmark SPSC queue
BenchmarkResult BenchmarkSPSCQueue(size_t num_items) {
    SPSCQueue<int> queue;
    queue.Init(1024);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_items; ++i) {
        while (!queue.Push(static_cast<int>(i))) { /* spin */ }
        int value;
        while (!queue.Pop(value)) { /* spin */ }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double ops_per_sec = num_items * 1000000.0 / total_us;
    double ns_per_op = total_us * 1000.0 / num_items;
    
    return {"SPSC Queue", total_us, num_items, ops_per_sec, ns_per_op};
}

// Benchmark MPSC queue
BenchmarkResult BenchmarkMPSCQueue(size_t num_items) {
    MPSCQueue<int> queue;
    queue.Init(num_items + 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_items; ++i) {
        queue.Push(static_cast<int>(i));
    }
    int value;
    while (queue.Pop(value)) { /* drain */ }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double ops_per_sec = num_items * 1000000.0 / total_us;
    double ns_per_op = total_us * 1000.0 / num_items;
    
    return {"MPSC Queue", total_us, num_items, ops_per_sec, ns_per_op};
}

// Benchmark memory pool
BenchmarkResult BenchmarkPool(size_t num_allocs) {
    FixedSizePoolBridge pool(64, num_allocs * 2);
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocs);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocs; ++i) {
        ptrs.push_back(pool.allocate(64));
    }
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double ops_per_sec = num_allocs * 1000000.0 / total_us;
    double ns_per_op = total_us * 1000.0 / num_allocs;
    
    return {"Memory Pool", total_us, num_allocs, ops_per_sec, ns_per_op};
}

// Benchmark new/delete baseline
BenchmarkResult BenchmarkNewDelete(size_t num_allocs) {
    std::vector<void*> ptrs;
    ptrs.reserve(num_allocs);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_allocs; ++i) {
        ptrs.push_back(::operator new(64));
    }
    for (void* ptr : ptrs) {
        ::operator delete(ptr);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double ops_per_sec = num_allocs * 1000000.0 / total_us;
    double ns_per_op = total_us * 1000.0 / num_allocs;
    
    return {"new/delete", total_us, num_allocs, ops_per_sec, ns_per_op};
}

void PrintResult(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(15) << r.name
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) 
              << r.total_us << " µs"
              << std::setw(15) << std::fixed << std::setprecision(0)
              << r.ops_per_sec << " ops/s"
              << std::setw(12) << std::fixed << std::setprecision(1)
              << r.ns_per_op << " ns/op"
              << std::endl;
}

int main() {
    const size_t NUM_ITEMS = 100000;
    
    std::cout << "\n╔══════════════════════════════════════════════════════════╗\n"
              << "║      SciComputeInfra 性能基准测试                       ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n";
    
    std::cout << "\n测试配置: " << NUM_ITEMS << " 次操作\n" << std::endl;
    std::cout << std::left << std::setw(15) << "Benchmark"
              << std::right << std::setw(14) << "Total Time"
              << std::setw(16) << "Throughput"
              << std::setw(14) << "Latency"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    PrintResult(BenchmarkSPSCQueue(NUM_ITEMS));
    PrintResult(BenchmarkMPSCQueue(NUM_ITEMS));
    PrintResult(BenchmarkPool(NUM_ITEMS));
    PrintResult(BenchmarkNewDelete(NUM_ITEMS));
    
    std::cout << "\n============================================================\n"
              << "基准测试完成\n"
              << "============================================================\n";
    
    return 0;
}
