/**
 * @file pool_bridge.cpp - 内存池桥接层实现 (动态扩容)
 */

#include "bridges/pool_bridge.hpp"
#include <cstring>

namespace sci {
namespace bridges {
namespace legacy_pool {


void LockFreeStack::Push(void* ptr) noexcept {
    FreeNode* node = static_cast<FreeNode*>(ptr);
    node->next = head_.load(std::memory_order_relaxed);
    while (!head_.compare_exchange_weak(node->next, node, std::memory_order_release, std::memory_order_relaxed)) {}
}

void* LockFreeStack::Pop() noexcept {
    FreeNode* node = head_.load(std::memory_order_acquire);
    while (node && !head_.compare_exchange_weak(node, node->next, std::memory_order_acquire, std::memory_order_relaxed)) {}
    return node;
}

bool LockFreeStack::Empty() const noexcept { 
    return head_.load(std::memory_order_acquire) == nullptr; 
}

// ============================================================================
// FixedSizeMemoryPool 实现 (动态扩容)
// ============================================================================

FixedSizeMemoryPool::FixedSizeMemoryPool(size_t block_size, size_t blocks_per_chunk)
    : block_size_(align_up(block_size, kDefaultAlignment))
    , blocks_per_chunk_(blocks_per_chunk)
    , max_chunks_(0)  // 0 = 无限制
    , grow_factor_(1.0)
    , is_expanding_(false)
    , total_allocated_(0)
    , total_freed_(0)
{
    Expand();
}

FixedSizeMemoryPool::~FixedSizeMemoryPool() {
    std::lock_guard<std::mutex> lock(mtx_);
    for (auto* chunk : chunks_) { delete[] chunk; }
}

void FixedSizeMemoryPool::Expand() {
    // 检查是否达到最大限制
    {
        std::lock_guard<std::mutex> lock(mtx_);
        if (max_chunks_ > 0 && chunks_.size() >= max_chunks_) {
            return;  // 达到上限，不扩容
        }
    }
    
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 双重检查
    if (max_chunks_ > 0 && chunks_.size() >= max_chunks_) {
        return;
    }
    
    // 计算本次扩容的块数
    size_t blocks = blocks_per_chunk_;
    if (grow_factor_ > 1.0 && !chunks_.empty()) {
        blocks = static_cast<size_t>(blocks_per_chunk_ * grow_factor_);
    }
    
    char* chunk = new char[block_size_ * blocks];
    chunks_.push_back(chunk);
    
    for (size_t i = 0; i < blocks; ++i) {
        free_list_.Push(chunk + i * block_size_);
        total_allocated_.fetch_add(block_size_, std::memory_order_relaxed);
    }
}

void* FixedSizeMemoryPool::Allocate() {
    void* ptr = free_list_.Pop();
    
    // 无空闲块，尝试扩容
    if (!ptr) {
        // 使用原子操作防止多线程同时扩容
        bool expected = false;
        if (is_expanding_.compare_exchange_strong(expected, true)) {
            Expand();
            is_expanding_.store(false);
        }
        ptr = free_list_.Pop();
    }
    
    if (ptr) {
        total_freed_.fetch_sub(block_size_, std::memory_order_relaxed);  // 近似值
    }
    
    return ptr;
}

void FixedSizeMemoryPool::Deallocate(void* ptr) noexcept {
    if (ptr) {
        free_list_.Push(ptr);
        total_freed_.fetch_add(block_size_, std::memory_order_relaxed);
    }
}

size_t FixedSizeMemoryPool::TotalAllocated() noexcept {
    // Use the maintained counter so callers see a consistent value
    // even if another thread is mid-Expand.
    return total_allocated_.load(std::memory_order_relaxed);
}

// ============================================================================
// SlabAllocator 实现
// ============================================================================

SlabAllocator::SlabAllocator() {
    for (size_t i = 0; i < kSlabCount; ++i) {
        slabs_[i].block_size = kSizes[i];
        slabs_[i].blocks_per_chunk = 256;
    }
}

void* SlabAllocator::Allocate(size_t size) noexcept {
    if (!CanAllocate(size)) return nullptr;
    size_t idx = IndexFor(size);
    auto& slab = slabs_[idx];
    
    void* ptr = slab.free_list.Pop();
    if (!ptr) {
        std::lock_guard<std::mutex> lock(slab.mtx);
        ptr = slab.free_list.Pop();
        if (!ptr) {
            // Slab自动扩容
            char* chunk = new char[slab.block_size * slab.blocks_per_chunk];
            slab.chunks.push_back(chunk);
            for (size_t i = 0; i < slab.blocks_per_chunk; ++i) {
                slab.free_list.Push(chunk + i * slab.block_size);
            }
            ptr = slab.free_list.Pop();
        }
    }
    return ptr;
}

void SlabAllocator::Deallocate(void* ptr, size_t size) noexcept {
    if (!ptr) return;
    slabs_[IndexFor(size)].free_list.Push(ptr);
}

size_t SlabAllocator::IndexFor(size_t size) const noexcept {
    if (size > kSizes[kSlabCount - 1]) return kSlabCount - 1;
    for (size_t i = 0; i < kSlabCount; ++i) { if (size <= kSizes[i]) return i; }
    return kSlabCount - 1;
}

size_t SlabAllocator::GetSlabSize(size_t size) const noexcept { return kSizes[IndexFor(size)]; }
bool SlabAllocator::CanAllocate(size_t size) const noexcept { return size > 0 && size <= kMaxSlabSize; }

// ============================================================================
// TlsCache 实现
// ============================================================================

TlsCache::TlsCache(AllocFn alloc, FreeFn free) : alloc_(std::move(alloc)), free_(std::move(free)) { 
    cache_.reserve(kBatchSize); 
    Refill(); 
}

void* TlsCache::Get() {
    if (cache_.empty()) Refill();
    if (!cache_.empty()) { void* ptr = cache_.back(); cache_.pop_back(); return ptr; }
    return alloc_();
}

void TlsCache::Put(void* ptr) {
    cache_.push_back(ptr);
    if (cache_.size() >= kBatchSize * 2) Flush();
}

size_t TlsCache::CacheSize() const { return cache_.size(); }

void TlsCache::Refill() {
    for (size_t i = 0; i < kBatchSize; ++i) {
        void* ptr = alloc_();
        if (ptr) cache_.push_back(ptr); else break;
    }
}

void TlsCache::Flush() {
    for (void* ptr : cache_) free_(ptr);
    cache_.clear();
    cache_.reserve(kBatchSize);
}

} // namespace legacy_pool

} // namespace bridges

} // namespace sci
