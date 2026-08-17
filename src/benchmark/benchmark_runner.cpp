#include "benchmark/benchmark_runner.hpp"
#include <fstream>
#include <iomanip>
#include <ctime>
#include <sys/time.h>
#include <iostream>
#include <cmath>

namespace sci {
namespace benchmark {

// ============================================================================
// BenchmarkCase Implementation
// ============================================================================
BenchmarkCase::BenchmarkCase(const std::string& name) : name_(name) {}

BenchmarkCase& BenchmarkCase::set_name(const std::string& name) {
    name_ = name;
    return *this;
}

BenchmarkCase& BenchmarkCase::set_unit(const std::string& unit) {
    unit_ = unit;
    return *this;
}

BenchmarkCase& BenchmarkCase::set_range(size_t start, size_t end, size_t step) {
    for (size_t i = start; i <= end; i += step) {
        range_.push_back(i);
    }
    return *this;
}

BenchmarkCase& BenchmarkCase::set_iterations(size_t min_iters, size_t max_iters) {
    min_iters_ = min_iters;
    max_iters_ = max_iters;
    return *this;
}

BenchmarkCase& BenchmarkCase::set_min_time_ms(size_t min_time_ms) {
    min_time_ms_ = min_time_ms;
    return *this;
}

BenchmarkCase& BenchmarkCase::set_timeout_ms(size_t timeout_ms) {
    timeout_ms_ = timeout_ms;
    return *this;
}

BenchmarkCase& BenchmarkCase::set_setup(SetupFunc func) {
    setup_ = std::move(func);
    return *this;
}

BenchmarkCase& BenchmarkCase::set_run(RunFunc func) {
    run_ = std::move(func);
    return *this;
}

BenchmarkCase& BenchmarkCase::set_teardown(TeardownFunc func) {
    teardown_ = std::move(func);
    return *this;
}

void BenchmarkCase::run() {
    results_.clear();
    
    if (range_.empty()) {
        // Single run
        Result result;
        result.name = name_;
        result.unit = unit_;
        result.num_items = 1;
        run_once(result);
        results_.push_back(result);
    } else {
        // Range runs
        for (size_t n : range_) {
            Result result;
            result.name = name_;
            result.unit = unit_;
            result.num_items = n;
            run_once(result);
            results_.push_back(result);
        }
    }
}

void BenchmarkCase::run_once(Result& result) {
    if (setup_) setup_();
    
    // Warmup
    if (run_) {
        run_(1);
    }
    
    // Timing loop
    std::vector<double> times;
    size_t iters = min_iters_;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iters; ++i) {
        if (run_) run_(1);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    
    // Adjust iterations based on elapsed time
    double target_time = min_time_ms_ * 1000000.0;
    if (elapsed < target_time && iters < max_iters_) {
        iters = std::min(iters * 2, max_iters_);
    }
    
    // Final measurement
    times.clear();
    for (size_t i = 0; i < iters; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (run_) run_(1);
        auto t_end = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::nano>(t_end - t_start).count());
    }
    
    // Calculate statistics
    result.iterations = iters;
    result.cpu_time_ns = times.empty() ? 0 : mean(times);
    result.real_time_ns = result.cpu_time_ns;
    result.mean = result.cpu_time_ns;
    result.min = times.empty() ? 0 : *std::min_element(times.begin(), times.end());
    result.max = times.empty() ? 0 : *std::max_element(times.begin(), times.end());
    result.stddev = stddev(times);
    result.items_per_second = result.num_items / (result.mean / 1e9);
    
    result.samples = times;
    
    if (teardown_) teardown_();
}

// ============================================================================
// BenchmarkRegistry Implementation
// ============================================================================
BenchmarkRegistry& BenchmarkRegistry::Instance() {
    static BenchmarkRegistry instance;
    return instance;
}

void BenchmarkRegistry::register_case(std::shared_ptr<BenchmarkCase> case_ptr) {
    cases_[case_ptr->name()] = case_ptr;
}

void BenchmarkRegistry::unregister(const std::string& name) {
    cases_.erase(name);
}

std::vector<std::string> BenchmarkRegistry::list_cases() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : cases_) {
        names.push_back(name);
    }
    return names;
}

void BenchmarkRegistry::clear() {
    cases_.clear();
}

template<typename T, typename... Args>
T* BenchmarkRegistry::register_and_get(const std::string& name, Args&&... args) {
    auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
    ptr->set_name(name);
    cases_[name] = ptr;
    return ptr.get();
}

// ============================================================================
// BenchmarkRunner Implementation
// ============================================================================
BenchmarkRunner& BenchmarkRunner::set_filter(const std::string& filter) {
    filter_ = filter;
    return *this;
}

BenchmarkRunner& BenchmarkRunner::set_output_format(const std::string& format) {
    output_format_ = format;
    return *this;
}

BenchmarkRunner& BenchmarkRunner::set_output_file(const std::string& path) {
    output_file_ = path;
    return *this;
}

BenchmarkRunner& BenchmarkRunner::set_min_time_ms(size_t min_time_ms) {
    min_time_ms_ = min_time_ms;
    return *this;
}

void BenchmarkRunner::add_case(std::shared_ptr<BenchmarkCase> case_ptr) {
    cases_.push_back(case_ptr);
}

void BenchmarkRunner::add_case(BenchmarkCase* case_ptr) {
    cases_.push_back(std::shared_ptr<BenchmarkCase>(case_ptr));
}

void BenchmarkRunner::run() {
    collect_system_info();
    filter_cases();
    
    for (auto& case_ptr : cases_) {
        case_ptr->run();
        for (const auto& result : case_ptr->results()) {
            report_.cases.push_back(result);
        }
    }
    
    if (output_format_ == "console" || output_format_.empty()) {
        print_results();
    } else if (output_format_ == "json") {
        export_json(output_file_.empty() ? "benchmark_results.json" : output_file_);
    } else if (output_format_ == "csv") {
        export_csv(output_file_.empty() ? "benchmark_results.csv" : output_file_);
    } else if (output_format_ == "markdown") {
        export_markdown(output_file_.empty() ? "benchmark_results.md" : output_file_);
    }
}

void BenchmarkRunner::collect_system_info() {
    // Timestamp
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    report_.timestamp = std::ctime(&now_c);
    
    // Hostname
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        report_.hostname = hostname;
    }
    
    report_.build_type = 
#if defined(SCI_DEBUG)
        "Debug";
#else
        "Release";
#endif
}

void BenchmarkRunner::filter_cases() {
    if (filter_.empty()) return;
    
    // Simple prefix matching filter
    std::vector<std::shared_ptr<BenchmarkCase>> filtered;
    for (auto& c : cases_) {
        if (c->name().find(filter_) != std::string::npos) {
            filtered.push_back(c);
        }
    }
    cases_ = filtered;
}

void BenchmarkRunner::print_results() const {
    std::cout << "\n========== Benchmark Results ==========\n";
    std::cout << "Host: " << report_.hostname;
    std::cout << "Build: " << report_.build_type << "\n";
    std::cout << "========================================\n\n";
    
    for (const auto& result : report_.cases) {
        std::cout << result.name << "\n";
        std::cout << "  Mean:   " << std::fixed << std::setprecision(3)
                  << result.mean / 1000 << " us\n";
        std::cout << "  StdDev: " << result.stddev / 1000 << " us\n";
        std::cout << "  Min:    " << result.min / 1000 << " us\n";
        std::cout << "  Max:    " << result.max / 1000 << " us\n";
        std::cout << "  Items/s: " << std::scientific << result.items_per_second << "\n";
        std::cout << "\n";
    }
}

void BenchmarkRunner::export_json(const std::string& path) const {
    std::ofstream file(path);
    file << "{\n";
    file << "  \"timestamp\": \"" << report_.timestamp << "\",\n";
    file << "  \"hostname\": \"" << report_.hostname << "\",\n";
    file << "  \"build_type\": \"" << report_.build_type << "\",\n";
    file << "  \"results\": [\n";
    
    for (size_t i = 0; i < report_.cases.size(); ++i) {
        const auto& r = report_.cases[i];
        file << "    {\n";
        file << "      \"name\": \"" << r.name << "\",\n";
        file << "      \"mean_ns\": " << r.mean << ",\n";
        file << "      \"stddev_ns\": " << r.stddev << ",\n";
        file << "      \"min_ns\": " << r.min << ",\n";
        file << "      \"max_ns\": " << r.max << ",\n";
        file << "      \"items_per_second\": " << r.items_per_second << "\n";
        file << "    }" << (i + 1 < report_.cases.size() ? "," : "") << "\n";
    }
    
    file << "  ]\n";
    file << "}\n";
}

void BenchmarkRunner::export_csv(const std::string& path) const {
    std::ofstream file(path);
    file << "name,mean_us,stddev_us,min_us,max_us,items_per_second\n";
    
    for (const auto& r : report_.cases) {
        file << r.name << ","
             << r.mean / 1000 << ","
             << r.stddev / 1000 << ","
             << r.min / 1000 << ","
             << r.max / 1000 << ","
             << r.items_per_second << "\n";
    }
}

void BenchmarkRunner::export_markdown(const std::string& path) const {
    std::ofstream file(path);
    file << "# Benchmark Results\n\n";
    file << "**Host:** " << report_.hostname << "\n";
    file << "**Build:** " << report_.build_type << "\n\n";
    
    file << "| Benchmark | Mean (us) | StdDev (us) | Min (us) | Max (us) | Items/s |\n";
    file << "|-----------|-----------|-------------|----------|----------|---------|\n";
    
    for (const auto& r : report_.cases) {
        file << "| " << r.name
             << " | " << r.mean / 1000
             << " | " << r.stddev / 1000
             << " | " << r.min / 1000
             << " | " << r.max / 1000
             << " | " << std::scientific << r.items_per_second
             << " |\n";
    }
}

// ============================================================================
// Utility Functions
// ============================================================================
double estimate_bandwidth_gbps(size_t bytes, double time_ms) {
    return (bytes / 1e9) / (time_ms / 1000.0);
}

double calculate_throughput(size_t items, double time_ms, const std::string& unit) {
    return items / (time_ms / 1000.0);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) return 0;
    double sum = 0;
    for (double v : values) sum += v;
    return sum / values.size();
}

double stddev(const std::vector<double>& values) {
    if (values.size() < 2) return 0;
    double m = mean(values);
    double sum_sq = 0;
    for (double v : values) {
        double diff = v - m;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / (values.size() - 1));
}

double percentile(const std::vector<double>& values, double p) {
    if (values.empty()) return 0;
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    size_t idx = static_cast<size_t>(p * sorted.size());
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    return sorted[idx];
}

} // namespace benchmark
} // namespace sci
