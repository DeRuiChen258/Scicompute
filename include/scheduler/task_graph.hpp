#pragma once

#include "task.hpp"
#include "thread_pool.hpp"
#include <unordered_map>
#include <unordered_set>

namespace sci {

// ============================================================================
// TaskGraph - DAG of tasks with dependencies
// ============================================================================
class TaskGraph {
public:
    TaskGraph() = default;
    ~TaskGraph() = default;
    
    // Node operations
    Task::Id add_task(std::shared_ptr<Task> task);
    std::shared_ptr<Task> get_task(Task::Id id) const;
    bool has_task(Task::Id id) const;
    void remove_task(Task::Id id);
    
    // Edge operations
    void add_edge(Task::Id from, Task::Id to);
    void add_edges(Task::Id from, const std::vector<Task::Id>& tos);
    
    // Dependency queries
    bool depends_on(Task::Id a, Task::Id b) const;
    std::vector<Task::Id> get_dependencies(Task::Id id) const;
    std::vector<Task::Id> get_dependents(Task::Id id) const;
    
    // Topological order
    std::vector<Task::Id> topological_sort() const;
    
    // Ready tasks (no unmet dependencies)
    std::vector<Task::Id> get_ready_tasks() const;
    
    // Stats
    size_t num_tasks() const { return tasks_.size(); }
    size_t num_edges() const { return edges_.size(); }
    bool empty() const { return tasks_.empty(); }
    void clear();
    
    // Validation
    bool is_dag() const;
    bool is_valid() const;
    
    // Iteration
    template<typename Func>
    void foreach_task(Func&& func) const;
    
    template<typename Func>
    void foreach_edge(Func&& func) const;
    
private:
    friend class GraphExecutor;
    
    bool has_cycle_dfs(Task::Id id, std::unordered_set<Task::Id>& visited,
                       std::unordered_set<Task::Id>& rec_stack) const;
    
    std::unordered_map<Task::Id, std::shared_ptr<Task>> tasks_;
    std::unordered_map<Task::Id, std::vector<Task::Id>> edges_;
    std::unordered_map<Task::Id, std::vector<Task::Id>> rev_edges_;
};

template<typename Func>
void TaskGraph::foreach_task(Func&& func) const {
    for (const auto& [id, task] : tasks_) {
        func(id, task);
    }
}

template<typename Func>
void TaskGraph::foreach_edge(Func&& func) const {
    for (const auto& [from, tos] : edges_) {
        for (auto to : tos) {
            func(from, to);
        }
    }
}

// ============================================================================
// GraphExecutor - Executes a task graph
// ============================================================================
class GraphExecutor {
public:
    GraphExecutor() = default;
    explicit GraphExecutor(std::shared_ptr<ThreadPool> thread_pool);
    ~GraphExecutor();
    
    // Execute entire graph
    Status execute(TaskGraph& graph);
    
    // Execute with callbacks
    using ProgressCallback = std::function<void(size_t completed, size_t total)>;
    Status execute(TaskGraph& graph, ProgressCallback progress_cb);
    
    // Partial execution
    Status execute(TaskGraph& graph, const std::vector<Task::Id>& targets);
    
    // Cancel
    void cancel();
    
    // Stats
    size_t num_running() const { return running_.load(); }
    size_t num_completed() const { return completed_.load(); }
    
private:
    void worker_loop();
    void process_ready_tasks(TaskGraph& graph);
    void mark_completed(Task::Id id);
    bool all_complete(const TaskGraph& graph) const;
    
    std::shared_ptr<ThreadPool> thread_pool_;
    std::unordered_set<Task::Id> running_tasks_;
    std::unordered_set<Task::Id> completed_tasks_;
    std::atomic<size_t> running_{0};
    std::atomic<size_t> completed_{0};
    std::atomic<bool> cancelled_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_thread_;
};

} // namespace sci
