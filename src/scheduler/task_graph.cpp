#include "scheduler/task_graph.hpp"
#include "scheduler/thread_pool.hpp"

namespace sci {

// ============================================================================
// TaskGraph Implementation
// ============================================================================
Task::Id TaskGraph::add_task(std::shared_ptr<Task> task) {
    Task::Id id = task->id();
    tasks_[id] = std::move(task);
    return id;
}

std::shared_ptr<Task> TaskGraph::get_task(Task::Id id) const {
    auto it = tasks_.find(id);
    if (it != tasks_.end()) {
        return it->second;
    }
    return nullptr;
}

bool TaskGraph::has_task(Task::Id id) const {
    return tasks_.find(id) != tasks_.end();
}

void TaskGraph::remove_task(Task::Id id) {
    tasks_.erase(id);
    edges_.erase(id);
    rev_edges_.erase(id);
    
    for (auto& [from, tos] : edges_) {
        tos.erase(std::remove(tos.begin(), tos.end(), id), tos.end());
    }
    for (auto& [to, froms] : rev_edges_) {
        froms.erase(std::remove(froms.begin(), froms.end(), id), froms.end());
    }
}

void TaskGraph::add_edge(Task::Id from, Task::Id to) {
    SCI_ASSERT(has_task(from), "Source task not found");
    SCI_ASSERT(has_task(to), "Target task not found");
    
    edges_[from].push_back(to);
    rev_edges_[to].push_back(from);
}

void TaskGraph::add_edges(Task::Id from, const std::vector<Task::Id>& tos) {
    for (auto to : tos) {
        add_edge(from, to);
    }
}

bool TaskGraph::depends_on(Task::Id a, Task::Id b) const {
    std::unordered_set<Task::Id> visited;
    std::queue<Task::Id> queue;
    
    auto it = rev_edges_.find(a);
    if (it == rev_edges_.end()) return false;
    
    for (auto dep : it->second) {
        queue.push(dep);
    }
    
    while (!queue.empty()) {
        Task::Id current = queue.front();
        queue.pop();
        
        if (current == b) return true;
        if (visited.count(current)) continue;
        visited.insert(current);
        
        auto it2 = rev_edges_.find(current);
        if (it2 != rev_edges_.end()) {
            for (auto dep : it2->second) {
                queue.push(dep);
            }
        }
    }
    
    return false;
}

std::vector<Task::Id> TaskGraph::get_dependencies(Task::Id id) const {
    auto it = rev_edges_.find(id);
    if (it != rev_edges_.end()) {
        return it->second;
    }
    return {};
}

std::vector<Task::Id> TaskGraph::get_dependents(Task::Id id) const {
    auto it = edges_.find(id);
    if (it != edges_.end()) {
        return it->second;
    }
    return {};
}

std::vector<Task::Id> TaskGraph::topological_sort() const {
    std::vector<Task::Id> result;
    std::unordered_set<Task::Id> visited;
    std::unordered_set<Task::Id> in_stack;
    
    std::function<bool(Task::Id)> dfs = [&](Task::Id id) -> bool {
        if (in_stack.count(id)) return false;
        if (visited.count(id)) return true;
        
        visited.insert(id);
        in_stack.insert(id);
        
        auto it = edges_.find(id);
        if (it != edges_.end()) {
            for (auto dep : it->second) {
                if (!dfs(dep)) return false;
            }
        }
        
        in_stack.erase(id);
        result.push_back(id);
        return true;
    };
    
    for (const auto& [id, _] : tasks_) {
        if (!visited.count(id)) {
            if (!dfs(id)) {
                return {};
            }
        }
    }
    
    return result;
}

std::vector<Task::Id> TaskGraph::get_ready_tasks() const {
    std::vector<Task::Id> ready;
    
    for (const auto& [id, task] : tasks_) {
        if (task->status() != Task::Status::kPending) continue;
        
        bool all_deps_complete = true;
        for (auto dep_id : get_dependencies(id)) {
            auto dep = get_task(dep_id);
            if (!dep || dep->status() != Task::Status::kCompleted) {
                all_deps_complete = false;
                break;
            }
        }
        
        if (all_deps_complete) {
            ready.push_back(id);
        }
    }
    
    return ready;
}

void TaskGraph::clear() {
    tasks_.clear();
    edges_.clear();
    rev_edges_.clear();
}

bool TaskGraph::is_dag() const {
    std::unordered_set<Task::Id> visited;
    std::unordered_set<Task::Id> rec_stack;
    
    std::function<bool(Task::Id)> dfs = [&](Task::Id id) -> bool {
        if (rec_stack.count(id)) return false;
        if (visited.count(id)) return true;
        
        visited.insert(id);
        rec_stack.insert(id);
        
        auto it = edges_.find(id);
        if (it != edges_.end()) {
            for (auto dep : it->second) {
                if (!dfs(dep)) return false;
            }
        }
        
        rec_stack.erase(id);
        return true;
    };
    
    for (const auto& [id, _] : tasks_) {
        if (!visited.count(id)) {
            if (!dfs(id)) return false;
        }
    }
    
    return true;
}

bool TaskGraph::is_valid() const {
    if (!is_dag()) return false;
    
    for (const auto& [from, tos] : edges_) {
        if (!has_task(from)) return false;
        for (auto to : tos) {
            if (!has_task(to)) return false;
        }
    }
    
    return true;
}

// ============================================================================
// GraphExecutor Implementation
// ============================================================================
GraphExecutor::GraphExecutor(std::shared_ptr<ThreadPool> thread_pool)
    : thread_pool_(thread_pool) {
    worker_thread_ = std::thread(&GraphExecutor::worker_loop, this);
}

GraphExecutor::~GraphExecutor() {
    cancel();
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void GraphExecutor::worker_loop() {
    while (!cancelled_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (cancelled_.load()) break;
    }
}

void GraphExecutor::process_ready_tasks(TaskGraph& graph) {
    auto ready = graph.get_ready_tasks();
    for (auto id : ready) {
        if (cancelled_.load()) break;
        
        auto task = graph.get_task(id);
        if (!task) continue;
        
        running_tasks_.insert(id);
        running_.fetch_add(1);
        
        if (thread_pool_) {
            thread_pool_->enqueue([this, id, task]() {
                Task::Status status = task->execute();
                {
                    std::lock_guard lock(mtx_);
                    mark_completed(id);
                }
                cv_.notify_all();
            });
        } else {
            task->execute();
            std::lock_guard lock(mtx_);
            mark_completed(id);
        }
    }
}

void GraphExecutor::mark_completed(Task::Id id) {
    running_tasks_.erase(id);
    completed_tasks_.insert(id);
    running_.fetch_sub(1);
    completed_.fetch_add(1);
}

bool GraphExecutor::all_complete(const TaskGraph& graph) const {
    return completed_.load() == graph.num_tasks();
}

Status GraphExecutor::execute(TaskGraph& graph) {
    cancelled_.store(false);
    completed_.store(0);
    
    if (!graph.is_valid()) {
        return Status::InvalidArgument("Invalid task graph");
    }
    
    while (!all_complete(graph) && !cancelled_.load()) {
        process_ready_tasks(graph);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    return cancelled_.load() ? Status::Error("Cancelled") : Status::Ok();
}

Status GraphExecutor::execute(TaskGraph& graph, ProgressCallback progress_cb) {
    Status result = execute(graph);
    if (progress_cb && graph.num_tasks() > 0) {
        progress_cb(completed_.load(), graph.num_tasks());
    }
    return result;
}

Status GraphExecutor::execute(TaskGraph& graph, const std::vector<Task::Id>& targets) {
    return execute(graph);
}

void GraphExecutor::cancel() {
    cancelled_.store(true);
}

} // namespace sci
