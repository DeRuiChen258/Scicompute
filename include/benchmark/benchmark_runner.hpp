#pragma once

#include "benchmark_case.hpp"
#include <fstream>

namespace sci {
namespace benchmark {

// ============================================================================
// Benchmark Report
// ============================================================================
struct BenchmarkReport {
    std::string timestamp;
    std::string hostname;
    std::string cpu_info;
    std::string gpu_info;
    std::string compiler_version;
    std::string build_type;
    std::vector<BenchmarkCase::Result> cases;
};

// ============================================================================
// Benchmark Runner
// ============================================================================
class BenchmarkRunner {
public:
    BenchmarkRunner() = default;
    ~BenchmarkRunner() = default;
    
    // Configuration
    BenchmarkRunner& set_filter(const std::string& filter);
    BenchmarkRunner& set_output_format(const std::string& format);  // json, csv, markdown
    BenchmarkRunner& set_output_file(const std::string& path);
    BenchmarkRunner& set_min_time_ms(size_t min_time_ms);
    
    // Add benchmark cases
    void add_case(std::shared_ptr<BenchmarkCase> case_ptr);
    void add_case(BenchmarkCase* case_ptr);  // Takes ownership
    
    // Run all benchmarks
    void run();
    
    // Get report
    const BenchmarkReport& report() const { return report_; }
    
    // Export
    void export_json(const std::string& path) const;
    void export_csv(const std::string& path) const;
    void export_markdown(const std::string& path) const;
    
    // Print to console
    void print_results() const;
    
private:
    void collect_system_info();
    void filter_cases();
    
    std::string filter_;
    std::string output_format_ = "console";
    std::string output_file_;
    size_t min_time_ms_ = 100;
    
    std::vector<std::shared_ptr<BenchmarkCase>> cases_;
    BenchmarkReport report_;
};

// ============================================================================
// Benchmark Utilities
// ============================================================================
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    
    Timer() : start_(Clock::now()) {}
    
    void reset() { start_ = Clock::now(); }
    
    double elapsed_ns() const {
        return std::chrono::duration<double, std::nano>(
            Clock::now() - start_).count();
    }
    
    double elapsed_us() const { return elapsed_ns() / 1000.0; }
    double elapsed_ms() const { return elapsed_ns() / 1000000.0; }
    double elapsed_s() const { return elapsed_ns() / 1000000000.0; }
    
private:
    Clock::time_point start_;
};

// Memory bandwidth estimation
double estimate_bandwidth_gbps(size_t bytes, double time_ms);

// Throughput calculation
double calculate_throughput(size_t items, double time_ms, const std::string& unit);

// Statistical utilities
double mean(const std::vector<double>& values);
double stddev(const std::vector<double>& values);
double percentile(const std::vector<double>& values, double p);

} // namespace benchmark
} // namespace sci
