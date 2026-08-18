/**
 * @file thread_pool_bridge.cpp - 线程池桥接层实现
 */

#include "bridges/thread_pool_bridge.hpp"

namespace sci {
namespace bridges {

void LegacyThreadPool::Start(size_t n) {
    workers_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back([this] { WorkerLoop(); });
    }
}

void LegacyThreadPool::WorkerLoop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return stop_.load() || !tasks_.empty() || pause_.load(); });
            if (stop_.load() && tasks_.empty()) return;
            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }
        if (task) {
            active_.fetch_add(1);
            queued_.fetch_sub(1);
            task();
            completed_.fetch_add(1);
            active_.fetch_sub(1);
            done_cv_.notify_all();
        }
    }
}

LegacyThreadPool::LegacyThreadPool(size_t num_threads) : config_{}, stop_(false), pause_(false) {
    Start(num_threads > 0 ? num_threads : std::thread::hardware_concurrency());
}

LegacyThreadPool::LegacyThreadPool(Config cfg) : config_(cfg), stop_(false), pause_(false) {
    Start(config_.min_threads);
}

LegacyThreadPool::~LegacyThreadPool() {
    Shutdown();
}

void LegacyThreadPool::WaitAll() {
    std::unique_lock<std::mutex> lock(mtx_);
    done_cv_.wait(lock, [this] { return tasks_.empty() && active_.load() == 0; });
}

void LegacyThreadPool::Resize(size_t n) { Start(n); }

void LegacyThreadPool::Shutdown() {
    stop_.store(true);
    cv_.notify_all();
    for (auto& worker : workers_) { if (worker.joinable()) worker.join(); }
    workers_.clear();
}

std::unique_ptr<LegacyThreadPool> CreateThreadPool(size_t num_threads) {
    return std::make_unique<LegacyThreadPool>(num_threads);
}

std::unique_ptr<LegacyThreadPool> CreateThreadPool(LegacyThreadPool::Config config) {
    return std::make_unique<LegacyThreadPool>(config);
}

} // namespace bridges
} // namespace sci
