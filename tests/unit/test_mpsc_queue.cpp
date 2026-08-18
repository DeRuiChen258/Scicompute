// Multi-producer / single-consumer concurrency regression tests for sci::MPSCQueue.

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <thread>
#include <vector>

#include "bridges/impl/ipc_queue_impl.hpp"

namespace {

struct TaggedItem {
    int producer;
    uint64_t seq;
};

TEST(MPSCQueueTest, BasicPushPop) {
    sci::bridges::MPSCQueue<int> q;
    ASSERT_TRUE(q.Init(64));

    EXPECT_TRUE(q.Push(1));
    EXPECT_TRUE(q.Push(2));
    EXPECT_TRUE(q.Push(3));

    int v = 0;
    EXPECT_TRUE(q.Pop(v));
    EXPECT_EQ(v, 1);
    EXPECT_TRUE(q.Pop(v));
    EXPECT_EQ(v, 2);
    EXPECT_TRUE(q.Pop(v));
    EXPECT_EQ(v, 3);
    EXPECT_FALSE(q.Pop(v));
    EXPECT_TRUE(q.Empty());
}

TEST(MPSCQueueTest, ReserveGrowsCapacity) {
    sci::bridges::MPSCQueue<int> q;
    ASSERT_TRUE(q.Init(8));
    EXPECT_EQ(q.capacity(), 8u);

    EXPECT_TRUE(q.Reserve(64));
    EXPECT_EQ(q.capacity(), 64u);

    EXPECT_TRUE(q.Push(42));
    int v = 0;
    EXPECT_TRUE(q.Pop(v));
    EXPECT_EQ(v, 42);
}

TEST(MPSCQueueTest, ReportsFullAndRejectsLiveResize) {
    sci::bridges::MPSCQueue<int> q;
    EXPECT_FALSE(q.Init(1));
    ASSERT_TRUE(q.Init(4));

    for (int value = 0; value < 4; ++value) {
        EXPECT_TRUE(q.Push(value));
    }
    EXPECT_FALSE(q.Push(4));
    EXPECT_EQ(q.SizeApprox(), 4u);
    EXPECT_FALSE(q.Reserve(8));

    for (int expected = 0; expected < 4; ++expected) {
        int value = -1;
        ASSERT_TRUE(q.Pop(value));
        EXPECT_EQ(value, expected);
    }
    EXPECT_TRUE(q.Empty());
    EXPECT_TRUE(q.Reserve(8));
}

TEST(MPSCQueueTest, ConcurrentProducersNoLoss) {
    constexpr int kProducers = 4;
    constexpr uint64_t kItemsPerProducer = 1000;
    sci::bridges::MPSCQueue<TaggedItem> q;
    ASSERT_TRUE(q.Init(1024));

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (uint64_t i = 0; i < kItemsPerProducer; ++i) {
                TaggedItem item{p, i};
                while (!q.Push(item)) {
                    std::this_thread::yield();
                }
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kProducers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    std::vector<std::vector<bool>> seen(
        kProducers, std::vector<bool>(kItemsPerProducer, false));
    TaggedItem item{0, 0};
    const size_t expected = static_cast<size_t>(kProducers) * kItemsPerProducer;
    size_t consumed = 0;
    while (consumed < expected) {
        if (q.Pop(item)) {
            ASSERT_GE(item.producer, 0);
            ASSERT_LT(item.producer, kProducers);
            ASSERT_LT(item.seq, kItemsPerProducer);
            EXPECT_FALSE(seen[item.producer][item.seq]);
            seen[item.producer][item.seq] = true;
            ++consumed;
        } else {
            std::this_thread::yield();
        }
    }

    for (auto& t : producers) t.join();

    for (int p = 0; p < kProducers; ++p) {
        EXPECT_TRUE(std::all_of(seen[p].begin(), seen[p].end(),
                                [](bool value) { return value; }))
            << "producer " << p << " lost items";
    }
    EXPECT_TRUE(q.Empty());
}

TEST(MPSCQueueTest, ConcurrentProducersBatch) {
    constexpr int kProducers = 4;
    constexpr uint64_t kItemsPerProducer = 1000;
    sci::bridges::MPSCQueue<TaggedItem> q;
    ASSERT_TRUE(q.Init(256));

    std::atomic<int> ready{0};
    std::atomic<bool> start{false};

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; ++p) {
        producers.emplace_back([&, p]() {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::vector<TaggedItem> batch;
            batch.reserve(64);
            uint64_t i = 0;
            while (i < kItemsPerProducer) {
                batch.clear();
                const uint64_t chunk = std::min<uint64_t>(32, kItemsPerProducer - i);
                for (uint64_t k = 0; k < chunk; ++k) {
                    batch.push_back({p, i + k});
                }
                size_t pushed = 0;
                while (pushed < batch.size()) {
                    pushed += q.PushBatch(batch.data() + pushed,
                                          batch.size() - pushed);
                    if (pushed < batch.size()) std::this_thread::yield();
                }
                i += chunk;
            }
        });
    }

    while (ready.load(std::memory_order_acquire) < kProducers) {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    std::vector<std::vector<bool>> seen(
        kProducers, std::vector<bool>(kItemsPerProducer, false));
    std::vector<TaggedItem> out(64);
    const size_t expected = static_cast<size_t>(kProducers) * kItemsPerProducer;
    size_t consumed = 0;
    while (consumed < expected) {
        size_t got = q.PopBatch(out.data(), out.size());
        for (size_t i = 0; i < got; ++i) {
            ASSERT_GE(out[i].producer, 0);
            ASSERT_LT(out[i].producer, kProducers);
            ASSERT_LT(out[i].seq, kItemsPerProducer);
            EXPECT_FALSE(seen[out[i].producer][out[i].seq]);
            seen[out[i].producer][out[i].seq] = true;
            ++consumed;
        }
        if (got == 0) std::this_thread::yield();
    }

    for (auto& t : producers) t.join();

    for (int p = 0; p < kProducers; ++p) {
        EXPECT_TRUE(std::all_of(seen[p].begin(), seen[p].end(),
                                [](bool value) { return value; }));
    }
}

}  // namespace
