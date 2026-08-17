#include "scheduler/task.hpp"

namespace sci {

std::atomic<Task::Id> Task::next_task_id_{0};

Task::Task() : id_(next_task_id_++) {}

Task::Task(const std::string& name) : id_(next_task_id_++), name_(name) {}

void Task::add_dependency(Id dep) {
    dependencies_.push_back(dep);
}

void Task::add_dependency(const std::vector<Id>& deps) {
    dependencies_.insert(dependencies_.end(), deps.begin(), deps.end());
}

void Task::set_status(Status s) {
    status_.store(s);
}

void Task::notify_complete() {
    set_status(Status::kCompleted);
    if (callback_) {
        callback_(Status::kCompleted);
    }
}

void Task::set_callback(std::function<void(Status)> cb) {
    callback_ = std::move(cb);
}

void Task::set_error_handler(std::function<void(const Status&)> handler) {
    error_handler_ = std::move(handler);
}

Task::Id Task::next_id() {
    return next_task_id_.fetch_add(1);
}

FunctionalTask::FunctionalTask(Func func, const std::string& name)
    : Task(name), func_(std::move(func)) {}

Task::Status FunctionalTask::execute() {
    try {
        Status result = func_();
        if (result == Status::kCompleted) {
            set_status(Status::kCompleted);
        } else {
            set_status(Status::kFailed);
        }
        return status();
    } catch (const std::exception& e) {
        set_status(Status::kFailed);
        return Status::kFailed;
    }
}

} // namespace sci
