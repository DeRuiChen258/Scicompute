#pragma once

#include "../core/common.hpp"
#include <vector>
#include <array>
#include <atomic>
#include <mutex>
#include <functional>
#include <optional>
#include <cstddef>
#include <new>

namespace sci {
namespace memory {

// ============================================================================
// Alignment Utilities
// ============================================================================
constexpr size_t kDefaultAlignment = 16;

inline size_t align_up(size_t size, size_t alignment) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

// ============================================================================
// LockFreeStack - Lock-free Treiber Stack
// ============================================================================
struct FreeNode {
    FreeNode* next;
};

class LockFreeStack {
public:
    void Push(void* ptr) noexcept {
        auto* node = static_cast<FreeNode*>(ptr);
        FreeNode* head = head_.load(std::memory_order_relaxed);
        do {
            node->next = head;
        } while (!head_.compare_exchange_weak(
            head, node, std::memory_order_release, std::memory_order_relaxed));
    }

    void* Pop() noexcept {
        FreeNode* head = head_.load(std::memory_order_acquire);
        while (head) {
            FreeNode* next = head->next;
            if (head_.compare_exchange_weak(
                    head, next, std::memory_order_acquire, std::memory_order_relaxed)) {
                return head;
            }
        }
        return nullptr;
    }

    bool Empty() const noexcept { return head_.load(std::memory_order_acquire) == nullptr; }

private:
    std::atomic<FreeNode*> head_{nullptr};
};

// ============================================================================
// FixedSizeMemoryPool - Single-size memory pool (integrated from linux_cpp/Pool)
// ============================================================================
class FixedSizeMemoryPool {
public:
    FixedSizeMemoryPool(size_t block_size, size_t blocks_per_chunk = 256)
        : block_size_(std::max(block_size, sizeof(FreeNode)))
        , blocks_per_chunk_(blocks_per_chunk) {}

    ~FixedSizeMemoryPool() {
        for (char* page : chunks_) { ::operator delete[](page); }
    }

    void* Allocate() {
        void* ptr = free_list_.Pop();
        if (!ptr) {
            Expand();
            ptr = free_list_.Pop();
        }
        return ptr;
    }

    void Deallocate(void* ptr) noexcept {
        if (ptr) free_list_.Push(ptr);
    }

    size_t BlockSize() const noexcept { return block_size_; }
    size_t ChunkCount() const noexcept { return chunks_.size(); }
    size_t TotalAllocated() const noexcept { return chunks_.size() * block_size_ * blocks_per_chunk_; }

private:
    void Expand() {
        size_t bytes = block_size_ * blocks_per_chunk_;
        char* page = static_cast<char*>(::operator new[](bytes));
        {
            std::lock_guard lock(mtx_);
            chunks_.push_back(page);
        }
        for (size_t i = 0; i < blocks_per_chunk_; ++i) {
            free_list_.Push(page + i * block_size_);
        }
    }

    LockFreeStack       free_list_;
    std::vector<char*>  chunks_;
    size_t              block_size_;
    size_t              blocks_per_chunk_;
    std::mutex          mtx_;
};

// ============================================================================
// SlabAllocator - Multi-size slab allocator (integrated from linux_cpp/Pool)
// ============================================================================
class SlabAllocator {
    struct Slab {
        size_t         block_size;
        size_t         blocks_per_chunk;
        std::vector<char*> chunks;
        LockFreeStack  free_list;
        std::mutex     mtx;
    };

    static constexpr size_t kMaxSlabSize = 4096;
    static constexpr size_t kSlabCount   = 8;

    static constexpr std::array<size_t, kSlabCount> kSizes = {
        16, 32, 64, 128, 256, 512, 1024, 4096
    };

public:
    SlabAllocator() {
        for (size_t i = 0; i < kSlabCount; ++i) {
            slabs_[i].block_size       = kSizes[i];
            slabs_[i].blocks_per_chunk = std::max<size_t>(1, kMaxSlabSize / kSizes[i]);
        }
    }

    void* Allocate(size_t size) noexcept {
        auto idx = IndexFor(size);
        if (!idx) return nullptr;
        auto& slab = slabs_[*idx];

        void* ptr = slab.free_list.Pop();
        if (!ptr) {
            std::lock_guard lock(slab.mtx);
            ptr = slab.free_list.Pop();
            if (!ptr) {
                size_t bytes = slab.block_size * slab.blocks_per_chunk;
                char* page = static_cast<char*>(::operator new[](bytes));
                slab.chunks.push_back(page);
                for (size_t i = 0; i < slab.blocks_per_chunk; ++i) {
                    slab.free_list.Push(page + i * slab.block_size);
                }
                ptr = slab.free_list.Pop();
            }
        }
        return ptr;
    }

    void Deallocate(void* ptr, size_t size) noexcept {
        if (!ptr) return;
        auto idx = IndexFor(size);
        if (!idx) return;
        slabs_[*idx].free_list.Push(ptr);
    }

    // Check if a size can be allocated
    bool CanAllocate(size_t size) const noexcept {
        return IndexFor(size).has_value();
    }

    // Get slab size for a given allocation size
    size_t GetSlabSize(size_t size) const noexcept {
        auto idx = IndexFor(size);
        return idx ? kSizes[*idx] : size;
    }

private:
    std::optional<size_t> IndexFor(size_t size) const noexcept {
        for (size_t i = 0; i < kSlabCount; ++i) {
            if (size <= kSizes[i]) return i;
        }
        return std::nullopt;
    }

    std::array<Slab, kSlabCount> slabs_;
};

// ============================================================================
// TlsCache - Thread-local cache for batch operations (integrated from linux_cpp/Pool)
// ============================================================================
class TlsCache {
    static constexpr size_t kBatchSize = 16;
public:
    using AllocFn = std::function<void*()>;
    using FreeFn  = std::function<void(void*)>;

    TlsCache(AllocFn alloc, FreeFn free)
        : alloc_(std::move(alloc)), free_(std::move(free)) {}

    void* Get() {
        if (cache_.empty()) Refill();
        void* p = cache_.back();
        cache_.pop_back();
        return p;
    }

    void Put(void* ptr) {
        if (cache_.size() >= kBatchSize) Flush();
        cache_.push_back(ptr);
    }

    size_t CacheSize() const { return cache_.size(); }

    ~TlsCache() { Flush(); }

private:
    void Refill() {
        for (size_t i = 0; i < kBatchSize; ++i) {
            void* p = alloc_();
            if (!p) break;
            cache_.push_back(p);
        }
    }

    void Flush() {
        for (void* p : cache_) free_(p);
        cache_.clear();
    }

    AllocFn alloc_;
    FreeFn  free_;
    std::vector<void*> cache_;
};

// ============================================================================
// Global Memory Pool Manager
// ============================================================================
class MemoryPoolManager {
public:
    static MemoryPoolManager& Instance() {
        static MemoryPoolManager instance;
        return instance;
    }

    // Get or create a fixed-size pool
    FixedSizeMemoryPool* GetFixedPool(size_t block_size) {
        std::lock_guard lock(mtx_);
        auto it = fixed_pools_.find(block_size);
        if (it != fixed_pools_.end()) {
            return it->second.get();
        }
        auto pool = std::make_unique<FixedSizeMemoryPool>(block_size);
        auto* ptr = pool.get();
        fixed_pools_[block_size] = std::move(pool);
        return ptr;
    }

    // Access the slab allocator
    SlabAllocator* GetSlabAllocator() { return &slab_allocator_; }

    // Statistics
    size_t TotalMemoryUsage() const {
        size_t total = 0;
        for (const auto& [size, pool] : fixed_pools_) {
            total += pool->TotalAllocated();
        }
        return total;
    }

private:
    MemoryPoolManager() = default;

    std::mutex mtx_;
    std::map<size_t, std::unique_ptr<FixedSizeMemoryPool>> fixed_pools_;
    SlabAllocator slab_allocator_;
};

} // namespace memory
} // namespace sci
