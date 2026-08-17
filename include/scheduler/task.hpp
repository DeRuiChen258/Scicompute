#pragma once

#include "../core/common.hpp"
#include "../device/stream.hpp"
#include <memory>
#include <vector>
#include <functional>
#include <future>

namespace sci {

// ============================================================================
// Forward Declarations
// ============================================================================
class Task;
class ThreadPool;

// ============================================================================
// Task - Base class for executable tasks
// ============================================================================
class Task {
public:
    using Id = uint64_t;
    
    enum class Priority : int {
        kLow = 0,
        kNormal = 1,
        kHigh = 2,
    };
    
    enum class Status {
        kPending,
        kReady,
        kRunning,
        kCompleted,
        kFailed,
        kCancelled,
    };
    
    virtual ~Task() = default;
    
    // Task identification
    Id id() const { return id_; }
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }
    
    // Priority
    Priority priority() const { return priority_; }
    void set_priority(Priority p) { priority_ = p; }
    
    // Status
    Status status() const { return status_.load(); }
    bool is_completed() const { return status_.load() == Status::kCompleted; }
    bool is_failed() const { return status_.load() == Status::kFailed; }
    
    // Dependencies
    void add_dependency(Id dep);
    void add_dependency(const std::vector<Id>& deps);
    size_t num_dependencies() const { return dependencies_.size(); }
    const std::vector<Id>& dependencies() const { return dependencies_; }
    
    // Execute
    virtual Status execute() = 0;
    
    // Callbacks
    void set_callback(std::function<void(Status)> cb);
    void set_error_handler(std::function<void(const Status&)> handler);
    
protected:
    Task();
    explicit Task(const std::string& name);
    
    void set_status(Status s);
    void notify_complete();
    
    Id id_;
    std::string name_;
    Priority priority_ = Priority::kNormal;
    std::atomic<Status> status_{Status::kPending};
    std::vector<Id> dependencies_;
    std::function<void(Status)> callback_;
    std::function<void(const Status&)> error_handler_;
    
    static constexpr bool kIsAbstract = true;
    
private:
    friend class TaskGraph;
    
    static Id next_id();
    static std::atomic<Id> next_task_id_;
};

// ============================================================================
// Functional Task - Task with std::function
// ============================================================================
class FunctionalTask : public Task {
public:
    using Func = std::function<Status()>;
    
    explicit FunctionalTask(Func func, const std::string& name = {});
    
    Status execute() override;
    
    void set_func(Func f) { func_ = std::move(f); }
    
private:
    Func func_;
};

// ============================================================================
// Task Future - Async task wrapper
// ============================================================================
template<typename T>
class TaskFuture {
public:
    TaskFuture() = default;
    explicit TaskFuture(std::future<T> fut) : fut_(std::move(fut)) {}
    
    T get() { return fut_.get(); }
    void wait() { fut_.wait(); }
    bool is_ready() const { return fut_.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
    
    explicit operator bool() const { return fut_.valid(); }
    
private:
    std::future<T> fut_;
};

} // namespace sci
