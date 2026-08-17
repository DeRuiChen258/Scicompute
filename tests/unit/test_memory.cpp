#include <gtest/gtest.h>
#include "memory/allocator.hpp"
#include "memory/memory_pool.hpp"
#include "memory/buffer_handle.hpp"

using namespace sci;

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        host_alloc_ = &HostAllocator::Instance();
    }
    
    HostAllocator* host_alloc_;
};

TEST_F(MemoryTest, AllocateDeallocate) {
    size_t size = 1024;
    void* ptr = host_alloc_->allocate(size);
    ASSERT_NE(ptr, nullptr);
    
    // Fill with data
    std::memset(ptr, 0xAB, size);
    
    host_alloc_->deallocate(ptr);
}

TEST_F(MemoryTest, MultipleAllocations) {
    std::vector<void*> ptrs;
    size_t size = 512;
    
    // Allocate many blocks
    for (int i = 0; i < 100; ++i) {
        void* ptr = host_alloc_->allocate(size);
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    // Deallocate all
    for (void* ptr : ptrs) {
        host_alloc_->deallocate(ptr);
    }
}

TEST_F(MemoryTest, MemoryPool) {
    MemoryPool::Config config;
    config.block_size = 64;
    config.blocks_per_slab = 16;
    config.max_slabs = 4;
    config.enable_tls_cache = false;
    
    MemoryPool pool(config);
    
    std::vector<void*> ptrs;
    for (int i = 0; i < 32; ++i) {
        void* ptr = pool.allocate();
        ASSERT_NE(ptr, nullptr);
        ptrs.push_back(ptr);
    }
    
    for (void* ptr : ptrs) {
        pool.deallocate(ptr);
    }
    
    EXPECT_GT(pool.num_slabs(), 0);
}

TEST_F(MemoryTest, BufferHandle) {
    auto device = GetCPUDevice();
    BufferHandle handle(*device, 1024);
    
    ASSERT_NE(handle.data(), nullptr);
    EXPECT_EQ(handle.bytes(), 1024);
    EXPECT_TRUE(handle.owns_data());
    
    // Fill with data
    std::memset(handle.data(), 0xCD, 1024);
}

TEST_F(MemoryTest, BufferHandleMove) {
    auto device = GetCPUDevice();
    BufferHandle handle1(*device, 512);
    void* data1 = handle1.data();
    
    BufferHandle handle2 = std::move(handle1);
    
    EXPECT_EQ(handle2.data(), data1);
    EXPECT_FALSE(handle1);  // Should be null after move
}

TEST_F(MemoryTest, BufferHandleClone) {
    auto device = GetCPUDevice();
    BufferHandle handle1(*device, 256);
    std::memset(handle1.data(), 0xAB, 256);
    
    BufferHandle handle2 = handle1.clone();
    
    EXPECT_NE(handle2.data(), handle1.data());
    EXPECT_EQ(handle2.bytes(), handle1.bytes());
    
    // Verify content
    const uint8_t* d1 = static_cast<const uint8_t*>(handle1.data());
    const uint8_t* d2 = static_cast<const uint8_t*>(handle2.data());
    for (size_t i = 0; i < 256; ++i) {
        EXPECT_EQ(d1[i], d2[i]);
    }
}
