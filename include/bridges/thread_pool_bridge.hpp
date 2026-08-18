#pragma once
/**
 * @file thread_pool_bridge.hpp
 * @brief 线程池桥接层 - 连接 linux_cpp/Pool 到 SciComputeInfra
 */

#include "../core/common.hpp"
#include <functional>
#include <future>
#include <queue>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace sci {
namespace bridges {

class LegacyThreadPool {
public:
    struct Config {
        size_t min_threads = 2;
        size_t max_threads = 8;
        size_t idle_timeout_ms = 30000;
    };

    explicit LegacyThreadPool(size_t num_threads = 0);
    explicit LegacyThreadPool(Config cfg);
    ~LegacyThreadPool();

    LegacyThreadPool(const LegacyThreadPool&) = delete;
    LegacyThreadPool& operator=(const LegacyThreadPool&) = delete;

    template<typename F, typename... Args>
    auto Enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    void WaitAll();
    void Pause() { pause_.store(true); }
    void Resume() { pause_.store(false); cv_.notify_all(); }
    void Resize(size_t n);
    void Shutdown();

    size_t Active() const noexcept { return active_.load(); }
    size_t Queued() const noexcept { return queued_.load(); }
    size_t Threads() const noexcept { return workers_.size(); }
    size_t Completed() const noexcept { return completed_.load(); }

private:
    void Start(size_t n);
    void WorkerLoop();

    Config config_;
    std::vector<std::thread> workers_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::queue<std::function<void()>> tasks_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> pause_{false};
    std::atomic<size_t> active_{0};
    std::atomic<size_t> queued_{0};
    std::atomic<size_t> completed_{0};
};

template<typename F, typename... Args>
auto LegacyThreadPool::Enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using RetType = std::invoke_result_t<F, Args...>;
    auto task = std::make_shared<std::packaged_task<RetType()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto future = task->get_future();
    {
        std::lock_guard<std::mutex> lock(mtx_);
        tasks_.push([task]() { (*task)(); });
        queued_.fetch_add(1);
    }
    cv_.notify_one();
    return future;
}

std::unique_ptr<LegacyThreadPool> CreateThreadPool(size_t num_threads = 0);
std::unique_ptr<LegacyThreadPool> CreateThreadPool(LegacyThreadPool::Config config);

} // namespace bridges
} // namespace sci
