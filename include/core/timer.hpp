#pragma once

#include "common.hpp"
#include <chrono>
#include <limits>

namespace sci {

// ============================================================================
// High-Resolution Timer
// ============================================================================
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::nano>;
    
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
    TimePoint start_;
};

// ============================================================================
// Scoped Measurement
// ============================================================================
#define SCI_SCOPED_TIMER(var) ::sci::Timer var##_timer

// ============================================================================
// Performance Counter
// ============================================================================
class PerfCounter {
public:
    PerfCounter() { reset(); }
    
    void reset() {
        count_ = 0;
        total_ns_ = 0;
        min_ns_ = std::numeric_limits<uint64_t>::max();
        max_ns_ = 0;
    }
    
    void record(uint64_t ns) {
        count_++;
        total_ns_ += ns;
        min_ns_ = std::min(min_ns_, ns);
        max_ns_ = std::max(max_ns_, ns);
    }
    
    void record(const Timer& timer) {
        record(static_cast<uint64_t>(timer.elapsed_ns()));
    }
    
    size_t count() const { return count_; }
    uint64_t total_ns() const { return total_ns_; }
    double mean_ns() const { return count_ ? static_cast<double>(total_ns_) / count_ : 0; }
    uint64_t min_ns() const { return min_ns_; }
    uint64_t max_ns() const { return max_ns_; }
    
    double mean_ms() const { return mean_ns() / 1000000.0; }
    double throughput(size_t items) const { 
        double mean_s = mean_ns() / 1000000000.0;
        return count_ ? items / mean_s : 0; 
    }
    
private:
    size_t count_;
    uint64_t total_ns_;
    uint64_t min_ns_;
    uint64_t max_ns_;
};

} // namespace sci
