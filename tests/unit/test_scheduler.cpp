#include <gtest/gtest.h>
#include "scheduler/thread_pool.hpp"
#include "scheduler/task.hpp"
#include "scheduler/task_graph.hpp"

using namespace sci;

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        pool_ = std::make_shared<ThreadPool>(4);
    }
    
    std::shared_ptr<ThreadPool> pool_;
};

TEST_F(SchedulerTest, ThreadPoolEnqueue) {
    auto future = pool_->enqueue([]() { return 42; });
    EXPECT_EQ(future.get(), 42);
}

TEST_F(SchedulerTest, ThreadPoolMultipleTasks) {
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool_->enqueue([i]() { return i * i; }));
    }
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(futures[i].get(), i * i);
    }
}

TEST_F(SchedulerTest, ThreadPoolStats) {
    for (int i = 0; i < 5; ++i) {
        pool_->enqueue([]() { 
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    
    pool_->wait_all();
    EXPECT_EQ(pool_->completed(), 5);
}

TEST_F(SchedulerTest, TaskCreation) {
    auto task = std::make_shared<FunctionalTask>(
        []() { return Task::Status::kCompleted; },
        "test_task"
    );
    
    EXPECT_EQ(task->name(), "test_task");
    EXPECT_EQ(task->status(), Task::Status::kPending);
}

TEST_F(SchedulerTest, TaskExecution) {
    bool executed = false;
    auto task = std::make_shared<FunctionalTask>(
        [&executed]() {
            executed = true;
            return Task::Status::kCompleted;
        }
    );
    
    Task::Status status = task->execute();
    EXPECT_TRUE(executed);
    EXPECT_EQ(status, Task::Status::kCompleted);
}

TEST_F(SchedulerTest, TaskGraphBasic) {
    TaskGraph graph;
    
    auto task1 = std::make_shared<FunctionalTask>(
        []() { return Task::Status::kCompleted; }
    );
    auto task2 = std::make_shared<FunctionalTask>(
        []() { return Task::Status::kCompleted; }
    );
    
    Task::Id id1 = graph.add_task(task1);
    Task::Id id2 = graph.add_task(task2);
    
    EXPECT_EQ(graph.num_tasks(), 2);
    EXPECT_TRUE(graph.has_task(id1));
    EXPECT_TRUE(graph.has_task(id2));
}

TEST_F(SchedulerTest, TaskGraphEdges) {
    TaskGraph graph;
    
    auto task1 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    auto task2 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    auto task3 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    
    Task::Id id1 = graph.add_task(task1);
    Task::Id id2 = graph.add_task(task2);
    Task::Id id3 = graph.add_task(task3);
    
    // task3 depends on task1 and task2
    graph.add_edge(id1, id3);
    graph.add_edge(id2, id3);
    
    EXPECT_EQ(graph.num_edges(), 2);
    
    auto deps = graph.get_dependencies(id3);
    EXPECT_EQ(deps.size(), 2);
}

TEST_F(SchedulerTest, TaskGraphTopologicalSort) {
    TaskGraph graph;
    
    auto task1 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    auto task2 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    auto task3 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    
    Task::Id id1 = graph.add_task(task1);
    Task::Id id2 = graph.add_task(task2);
    Task::Id id3 = graph.add_task(task3);
    
    // Create DAG: 1 -> 2 -> 3
    graph.add_edge(id1, id2);
    graph.add_edge(id2, id3);
    
    auto sorted = graph.topological_sort();
    EXPECT_EQ(sorted.size(), 3);
    
    // Verify order: id1 before id2, id2 before id3
    size_t pos1 = std::find(sorted.begin(), sorted.end(), id1) - sorted.begin();
    size_t pos2 = std::find(sorted.begin(), sorted.end(), id2) - sorted.begin();
    size_t pos3 = std::find(sorted.begin(), sorted.end(), id3) - sorted.begin();
    
    EXPECT_LT(pos1, pos2);
    EXPECT_LT(pos2, pos3);
}

TEST_F(SchedulerTest, TaskGraphCycleDetection) {
    TaskGraph graph;
    
    auto task1 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    auto task2 = std::make_shared<FunctionalTask>([]() { return Task::Status::kCompleted; });
    
    Task::Id id1 = graph.add_task(task1);
    Task::Id id2 = graph.add_task(task2);
    
    // Create cycle: 1 -> 2 -> 1
    graph.add_edge(id1, id2);
    graph.add_edge(id2, id1);
    
    EXPECT_FALSE(graph.is_dag());
}
