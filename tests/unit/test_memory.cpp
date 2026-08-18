// =====================================================================
// Tests for the memory layer.
//
// These tests exercise the current public API surface:
//   - HostAllocator (default system allocator)
//   - FixedSizePoolBridge (fixed-size memory pool bridge)
//   - SlabPoolBridge (multi-size slab bridge)
//   - BufferHandle (RAII buffer wrapper)
//
// Note: behavior under concurrent allocation is validated in a separate
// stress test rather than here, because gtest's filter set is meant to
// stay fast and deterministic.
// =====================================================================

#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>
#include <cstring>
#include <random>

#include "memory/allocator.hpp"
#include "memory/buffer_handle.hpp"
#include "bridges/pool_bridge.hpp"
#include "device/device.hpp"

using namespace sci;

namespace {
constexpr size_t kBlockSize = 64;
constexpr size_t kBlocksPerChunk = 128;
}

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        host_alloc_ = &HostAllocator::Instance();
    }

    HostAllocator* host_alloc_ = nullptr;
};

TEST_F(MemoryTest, HostAllocatorAllocateDeallocate) {
    constexpr size_t kBytes = 1024;
    void* ptr = host_alloc_->allocate(kBytes);
    ASSERT_NE(ptr, nullptr);
    std::memset(ptr, 0xAB, kBytes);
    host_alloc_->deallocate(ptr);
}

TEST_F(MemoryTest, HostAllocatorMany) {
    std::vector<void*> ptrs;
    ptrs.reserve(100);
    for (int i = 0; i < 100; ++i) {
        void* p = host_alloc_->allocate(512);
        ASSERT_NE(p, nullptr);
        ptrs.push_back(p);
    }
    for (void* p : ptrs) {
        host_alloc_->deallocate(p);
    }
    EXPECT_EQ(host_alloc_->allocated_bytes(), 0u);
}

TEST_F(MemoryTest, FixedSizePoolBridgeBasic) {
    bridges::FixedSizePoolBridge pool(kBlockSize, kBlocksPerChunk);
    std::vector<void*> ptrs;
    for (int i = 0; i < 256; ++i) {
        void* p = pool.allocate(kBlockSize);
        ASSERT_NE(p, nullptr);
        ptrs.push_back(p);
    }
    EXPECT_GE(pool.block_count(), 1u);
    for (void* p : ptrs) {
        pool.deallocate(p);
    }
}

TEST_F(MemoryTest, FixedSizePoolBridgeRejectsWrongSize) {
    bridges::FixedSizePoolBridge pool(kBlockSize, kBlocksPerChunk);
    // A request for a different size should be rejected.
    EXPECT_EQ(pool.allocate(7), nullptr);
    void* p = pool.allocate(kBlockSize);
    EXPECT_NE(p, nullptr);
    pool.deallocate(p);
}

TEST_F(MemoryTest, SlabPoolBridgeRoundTrip) {
    bridges::SlabPoolBridge slab;
    void* p64 = slab.allocate(64);
    void* p256 = slab.allocate(256);
    void* p1024 = slab.allocate(1024);
    ASSERT_NE(p64, nullptr);
    ASSERT_NE(p256, nullptr);
    ASSERT_NE(p1024, nullptr);
    EXPECT_NE(p64, p256);
    EXPECT_NE(p256, p1024);
    slab.deallocate(p64);
    slab.deallocate(p256);
    slab.deallocate(p1024);
}

TEST_F(MemoryTest, SlabPoolBridgeSizeHeader) {
    bridges::SlabPoolBridge slab;
    void* p = slab.allocate(128);
    ASSERT_NE(p, nullptr);
    // The bridge should accept deallocate without an explicit size
    // because the size used at allocation is now stored in the header.
    slab.deallocate(p);
}

TEST_F(MemoryTest, FixedSizePoolBridgeConcurrent) {
    bridges::FixedSizePoolBridge pool(kBlockSize, kBlocksPerChunk);
    constexpr int kThreads = 16;
    constexpr int kIters = 1000;
    std::atomic<size_t> live{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&] {
            std::mt19937 rng(static_cast<unsigned>(t));
            for (int i = 0; i < kIters; ++i) {
                void* p = pool.allocate(kBlockSize);
                if (p == nullptr) continue;
                live.fetch_add(1);
                if ((rng() & 0x3) == 0) std::this_thread::yield();
                pool.deallocate(p);
                live.fetch_sub(1);
            }
        });
    }
    for (auto& th : workers) th.join();
    EXPECT_EQ(live.load(), 0u);
}

TEST_F(MemoryTest, BufferHandleBasic) {
    auto device = GetCPUDevice();
    BufferHandle handle(device.get(), 1024);
    ASSERT_NE(handle.data(), nullptr);
    EXPECT_EQ(handle.bytes(), 1024u);
    EXPECT_TRUE(handle.owns_data());
    std::memset(handle.data(), 0xCD, 1024);
}

TEST_F(MemoryTest, BufferHandleMove) {
    auto device = GetCPUDevice();
    BufferHandle handle1(device.get(), 512);
    void* data1 = handle1.data();
    BufferHandle handle2 = std::move(handle1);
    EXPECT_EQ(handle2.data(), data1);
    EXPECT_FALSE(static_cast<bool>(handle1));
}

TEST_F(MemoryTest, BufferHandleClone) {
    auto device = GetCPUDevice();
    BufferHandle handle1(device.get(), 256);
    std::memset(handle1.data(), 0xAB, 256);
    BufferHandle handle2 = handle1.clone();
    EXPECT_NE(handle2.data(), handle1.data());
    EXPECT_EQ(handle2.bytes(), handle1.bytes());
    const uint8_t* d1 = static_cast<const uint8_t*>(handle1.data());
    const uint8_t* d2 = static_cast<const uint8_t*>(handle2.data());
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(d1[i], d2[i]);
    }
}
