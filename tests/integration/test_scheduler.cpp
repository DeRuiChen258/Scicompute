#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "scheduler/thread_pool.hpp"
#include "scheduler/task.hpp"
#include "scheduler/task_graph.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "math/elementwise.hpp"
#include "math/reduction.hpp"
#include "math/softmax.hpp"

using namespace sci;

// ============================================================================
// 线程池 + Tensor 运算流水线
// ============================================================================
TEST(IntegrationSchedulerTest, ThreadPoolTensorPipeline) {
    ThreadPool pool(4);
    auto device = GetCPUDevice();

    auto a = std::make_shared<Tensor>(
        Tensor::Full({128}, DType::kFloat32, *device, 2.0f));
    auto b = std::make_shared<Tensor>(
        Tensor::Full({128}, DType::kFloat32, *device, 3.0f));

    auto add_fut = pool.enqueue([a, b]() -> std::shared_ptr<Tensor> {
        auto r = math::add(*a, *b);
        return r.ok() ? std::make_shared<Tensor>(std::move(*r)) : nullptr;
    });
    auto mul_fut = pool.enqueue([a, b]() -> std::shared_ptr<Tensor> {
        auto r = math::mul(*a, *b);
        return r.ok() ? std::make_shared<Tensor>(std::move(*r)) : nullptr;
    });

    auto add_res = add_fut.get();
    auto mul_res = mul_fut.get();
    ASSERT_NE(add_res, nullptr);
    ASSERT_NE(mul_res, nullptr);

    const float* add_data = add_res->data_ptr<float>();
    const float* mul_data = mul_res->data_ptr<float>();
    for (size_t i = 0; i < 128; ++i) {
        EXPECT_FLOAT_EQ(add_data[i], 5.0f);
        EXPECT_FLOAT_EQ(mul_data[i], 6.0f);
    }

    auto sum_fut = pool.enqueue([add_res]() -> std::shared_ptr<Tensor> {
        auto r = math::sum(*add_res);
        return r.ok() ? std::make_shared<Tensor>(std::move(*r)) : nullptr;
    });
    auto sum_res = sum_fut.get();
    ASSERT_NE(sum_res, nullptr);
    EXPECT_FLOAT_EQ(*sum_res->data_ptr<float>(), 640.0f);

    pool.wait_all();
}

// ============================================================================
// 任务图 + 调度器 + 张量运算 DAG
// ============================================================================
TEST(IntegrationSchedulerTest, GraphExecutorTensorDag) {
    auto pool = std::make_shared<ThreadPool>(4);
    GraphExecutor executor(pool);
    TaskGraph graph;
    auto device = GetCPUDevice();

    auto a = std::make_shared<Tensor>(
        Tensor::Full({4, 8}, DType::kFloat32, *device, 1.0f));
    auto b = std::make_shared<Tensor>(
        Tensor::Full({4, 8}, DType::kFloat32, *device, 2.0f));
    std::shared_ptr<Tensor> sum_ab;
    std::shared_ptr<Tensor> softmax_out;

    auto t_add = std::make_shared<FunctionalTask>(
        [a, b, &sum_ab]() {
            auto r = math::add(*a, *b);
            if (!r.ok()) return Task::Status::kFailed;
            sum_ab = std::make_shared<Tensor>(std::move(*r));
            return Task::Status::kCompleted;
        },
        "add");
    auto t_softmax = std::make_shared<FunctionalTask>(
        [&sum_ab, &softmax_out]() {
            if (!sum_ab) return Task::Status::kFailed;
            auto r = math::softmax(*sum_ab, -1);
            if (!r.ok()) return Task::Status::kFailed;
            softmax_out = std::make_shared<Tensor>(std::move(*r));
            return Task::Status::kCompleted;
        },
        "softmax");

    graph.add_task(t_add);
    graph.add_task(t_softmax);
    graph.add_edge(t_add->id(), t_softmax->id());

    Status status = executor.execute(graph);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(t_add->status(), Task::Status::kCompleted);
    EXPECT_EQ(t_softmax->status(), Task::Status::kCompleted);
    ASSERT_NE(softmax_out, nullptr);

    const float* sm = softmax_out->data_ptr<float>();
    for (int r = 0; r < 4; ++r) {
        float row_sum = 0.0f;
        for (int c = 0; c < 8; ++c) {
            row_sum += sm[r * 8 + c];
            EXPECT_FLOAT_EQ(sm[r * 8 + c], 0.125f);
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

// ============================================================================
// 依赖顺序保证: 无论前置任务耗时长短, 后继任务必须在其后执行
// ============================================================================
TEST(IntegrationSchedulerTest, DependencyOrderGuaranteed) {
    auto pool = std::make_shared<ThreadPool>(4);
    GraphExecutor executor(pool);
    TaskGraph graph;

    std::mutex mtx;
    std::vector<int> log;

    auto make_task = [&](int tag, int delay_ms) {
        return std::make_shared<FunctionalTask>(
            [&, tag, delay_ms]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                std::lock_guard<std::mutex> lock(mtx);
                log.push_back(tag);
                return Task::Status::kCompleted;
            },
            "task_" + std::to_string(tag));
    };

    auto t_a = make_task(1, 30);
    auto t_b = make_task(2, 10);
    auto t_c = make_task(3, 0);

    graph.add_task(t_a);
    graph.add_task(t_b);
    graph.add_task(t_c);
    graph.add_edge(t_a->id(), t_c->id());
    graph.add_edge(t_b->id(), t_c->id());

    Status status = executor.execute(graph);
    ASSERT_TRUE(status.ok()) << status.message();

    auto pos = [&](int tag) {
        for (size_t i = 0; i < log.size(); ++i) {
            if (log[i] == tag) return i;
        }
        return log.size();
    };

    // 修复重复入队问题后, 每个任务只应执行一次
    EXPECT_EQ(log.size(), 3u);
    EXPECT_LT(pos(1), pos(3));
    EXPECT_LT(pos(2), pos(3));
}

// ============================================================================
// 同一执行器重复运行任务图 (状态应可复用)
// ============================================================================
TEST(IntegrationSchedulerTest, GraphExecutorReusable) {
    auto pool = std::make_shared<ThreadPool>(4);
    GraphExecutor executor(pool);

    for (int round = 0; round < 3; ++round) {
        TaskGraph graph;
        std::atomic<int> counter{0};

        auto t1 = std::make_shared<FunctionalTask>(
            [&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
                return Task::Status::kCompleted;
            },
            "t1");
        auto t2 = std::make_shared<FunctionalTask>(
            [&counter]() {
                counter.fetch_add(1, std::memory_order_relaxed);
                return Task::Status::kCompleted;
            },
            "t2");

        graph.add_task(t1);
        graph.add_task(t2);

        Status status = executor.execute(graph);
        ASSERT_TRUE(status.ok()) << status.message();
        EXPECT_EQ(counter.load(), 2);
    }
}
