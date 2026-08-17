#pragma once

#include "../core/common.hpp"
#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <condition_variable>
#include <atomic>

namespace sci {

class ThreadPool {
public:
    struct Config {
        size_t min_threads = 2;
        size_t max_threads = 8;
        size_t idle_timeout_ms = 30000;
    };
    
    explicit ThreadPool(size_t num_threads);
    explicit ThreadPool(const Config& config);
    ~ThreadPool();
    
    SCI_DISALLOW_COPY_AND_MOVE(ThreadPool);
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;
    
    void wait_all();
    void pause();
    void resume();
    void resize(size_t n);
    
    size_t active() const { return active_.load(); }
    size_t queued() const { return queued_.load(); }
    size_t threads() const { return workers_.size(); }
    size_t completed() const { return completed_.load(); }
    
private:
    struct TaskBase {
        virtual ~TaskBase() = default;
        virtual void run() = 0;
    };
    
    template<typename F>
    struct Task final : TaskBase {
        F func;
        explicit Task(F&& f) : func(std::forward<F>(f)) {}
        void run() override { func(); }
    };
    
    void start(size_t n);
    void worker_loop();
    void shutdown();
    
    Config config_;
    std::vector<std::thread> workers_;
    
    std::mutex mtx_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::queue<std::unique_ptr<TaskBase>> tasks_;
    
    std::atomic<bool> stop_{false};
    std::atomic<bool> pause_{false};
    std::atomic<size_t> active_{0};
    std::atomic<size_t> queued_{0};
    std::atomic<size_t> completed_{0};
};

template<typename F, typename... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using Ret = std::invoke_result_t<F, Args...>;
    
    auto pkg = std::make_shared<std::packaged_task<Ret()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto fut = pkg->get_future();
    
    auto task = std::make_unique<Task<std::function<void()>>>(
        [pkg = std::move(pkg)]() { (*pkg)(); });
    
    {
        std::lock_guard lock(mtx_);
        if (stop_) throw std::runtime_error("ThreadPool: stopped");
        tasks_.push(std::move(task));
        queued_.fetch_add(1, std::memory_order_relaxed);
    }
    cv_.notify_one();
    return fut;
}

ThreadPool& GetGlobalThreadPool();

} // namespace sci
