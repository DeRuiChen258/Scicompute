#include "scheduler/thread_pool.hpp"

namespace sci {

ThreadPool::ThreadPool(size_t num_threads) : config_{num_threads, num_threads} {
    start(num_threads);
}

ThreadPool::ThreadPool(const Config& config) : config_(config) {
    start(config.min_threads);
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::start(size_t n) {
    for (size_t i = 0; i < n; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::worker_loop() {
    while (true) {
        std::unique_ptr<TaskBase> task;
        {
            std::unique_lock lock(mtx_);
            cv_.wait(lock, [this] {
                return stop_.load(std::memory_order_acquire) ||
                       (!pause_.load(std::memory_order_acquire) && !tasks_.empty());
            });
            if (stop_ && tasks_.empty()) return;
            if (pause_ && tasks_.empty()) continue;

            task = std::move(tasks_.front());
            tasks_.pop();
            queued_.fetch_sub(1, std::memory_order_relaxed);
            active_.fetch_add(1, std::memory_order_relaxed);
        }

        try {
            task->run();
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[ThreadPool] unhandled exception: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "[ThreadPool] unhandled unknown exception\n");
        }

        active_.fetch_sub(1, std::memory_order_relaxed);
        completed_.fetch_add(1, std::memory_order_relaxed);
        done_cv_.notify_all();
    }
}

void ThreadPool::wait_all() {
    std::unique_lock lock(mtx_);
    done_cv_.wait(lock, [this] {
        return tasks_.empty() && active_.load() == 0;
    });
}

void ThreadPool::pause() {
    pause_ = true;
}

void ThreadPool::resume() {
    pause_ = false;
    cv_.notify_all();
}

void ThreadPool::resize(size_t n) {
    std::lock_guard lock(mtx_);
    if (n <= workers_.size()) return;
    for (size_t i = workers_.size(); i < n; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::shutdown() {
    stop_ = true;
    cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

ThreadPool& GetGlobalThreadPool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}

} // namespace sci
