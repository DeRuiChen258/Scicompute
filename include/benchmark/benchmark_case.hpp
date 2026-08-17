#pragma once

#include "../core/common.hpp"
#include <chrono>
#include <vector>
#include <string>
#include <functional>

namespace sci {
namespace benchmark {

// ============================================================================
// Benchmark Case
// ============================================================================
class BenchmarkCase {
public:
    using RunFunc = std::function<void(int iters)>;
    using SetupFunc = std::function<void()>;
    using TeardownFunc = std::function<void()>;
    
    BenchmarkCase(const std::string& name);
    ~BenchmarkCase() = default;
    
    // Configuration
    BenchmarkCase& set_name(const std::string& name);
    const std::string& name() const { return name_; }
    BenchmarkCase& set_unit(const std::string& unit);
    BenchmarkCase& set_range(size_t start, size_t end, size_t step);
    BenchmarkCase& set_iterations(size_t min_iters, size_t max_iters);
    BenchmarkCase& set_min_time_ms(size_t min_time_ms);
    BenchmarkCase& set_timeout_ms(size_t timeout_ms);
    
    // Callbacks
    BenchmarkCase& set_setup(SetupFunc func);
    BenchmarkCase& set_run(RunFunc func);
    BenchmarkCase& set_teardown(TeardownFunc func);
    
    // Run
    void run();
    
    // Results
    struct Result {
        std::string name;
        std::string unit;
        size_t num_items = 0;
        double cpu_time_ns = 0;
        double real_time_ns = 0;
        size_t iterations = 0;
        double items_per_second = 0;
        double bytes_per_second = 0;
        double mean = 0;
        double stddev = 0;
        double min = 0;
        double max = 0;
        std::vector<double> samples;
    };
    
    const std::vector<Result>& results() const { return results_; }
    
private:
    void run_once(Result& result);
    
    std::string name_;
    std::string unit_ = "items";
    std::vector<size_t> range_;
    size_t min_iters_ = 3;
    size_t max_iters_ = 10000;
    size_t min_time_ms_ = 100;
    size_t timeout_ms_ = 60000;
    
    SetupFunc setup_;
    RunFunc run_;
    TeardownFunc teardown_;
    
    std::vector<Result> results_;
};

// ============================================================================
// Benchmark Registry
// ============================================================================
class BenchmarkRegistry {
public:
    static BenchmarkRegistry& Instance();
    
    void register_case(std::shared_ptr<BenchmarkCase> case_ptr);
    void unregister(const std::string& name);
    std::vector<std::string> list_cases() const;
    void clear();
    
    template<typename T, typename... Args>
    T* register_and_get(const std::string& name, Args&&... args);
    
private:
    BenchmarkRegistry() = default;
    
    std::unordered_map<std::string, std::shared_ptr<BenchmarkCase>> cases_;
};

// Helper macro
#define SCI_BENCHMARK(name) \
    static ::sci::benchmark::BenchmarkCase* name##_bench = \
        ::sci::benchmark::BenchmarkRegistry::Instance().register_and_get< ::sci::benchmark::BenchmarkCase>(#name)

} // namespace benchmark
} // namespace sci
