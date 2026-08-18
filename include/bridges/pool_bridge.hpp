#pragma once
/**
 * @file pool_bridge.hpp
 * @brief 内存池桥接层 - 动态扩容实现
 *
 * The bridge adapts the legacy_pool implementations (LockFreeStack /
 * FixedSizeMemoryPool / SlabAllocator / TlsCache) to SciComputeInfra's
 * sci::Allocator interface. The legacy code is intentionally kept in
 * tree for ABI stability; the canonical copy lives in
 * linux_cpp/Pool and is meant to be the source of truth.
 */

#include "../core/common.hpp"
#include "../memory/allocator.hpp"
#include <unordered_map>
#include <memory>
#include <functional>
#include <vector>
#include <atomic>

namespace sci {
namespace bridges {

// ============================================================================
// 前向声明
// ============================================================================
namespace legacy { class FixedSizeMemoryPool; class SlabAllocator; }

// ============================================================================
// 旧版 Pool API (来自 linux_cpp/Pool/Memory_Pool.cpp)
// ============================================================================
namespace legacy_pool {

constexpr size_t kDefaultAlignment = 16;

/**
 * 对齐函数
 */
inline size_t align_up(size_t size, size_t alignment) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * LockFreeStack - 无锁 Treiber Stack
 */
class LockFreeStack {
public:
    void Push(void* ptr) noexcept;
    void* Pop() noexcept;
    bool Empty() const noexcept;

public:
    struct FreeNode {
        FreeNode* next;
    };
    
    std::atomic<FreeNode*> head_{nullptr};
};

/**
 * FixedSizeMemoryPool - 单尺寸内存池 (动态扩容)
 */
class FixedSizeMemoryPool {
public:
    FixedSizeMemoryPool(size_t block_size, size_t blocks_per_chunk = 256);
    ~FixedSizeMemoryPool();

    void* Allocate();
    void Deallocate(void* ptr) noexcept;
    size_t BlockSize() const noexcept { return block_size_; }
    size_t ChunkCount() const noexcept { return chunks_.size(); }
    size_t TotalAllocated() noexcept;
    
    // 扩容控制
    void Expand();
    void SetMaxChunks(size_t max) { max_chunks_ = max; }
    void SetGrowFactor(double factor) { grow_factor_ = factor; }
    size_t MaxChunks() const { return max_chunks_; }
    double GrowFactor() const { return grow_factor_; }
    
    // 统计
    size_t FreeBlockCount() const;
    size_t TotalAllocatedBytes() const { return total_allocated_.load(std::memory_order_relaxed); }
    size_t TotalFreedBytes() const { return total_freed_.load(std::memory_order_relaxed); }

private:
    LockFreeStack       free_list_;
    std::vector<char*>  chunks_;
    size_t              block_size_;
    size_t              blocks_per_chunk_;
    size_t              max_chunks_;           // 0 = 无限制
    double              grow_factor_;           // 扩容倍数
    std::atomic<bool>  is_expanding_;         // 防止并发扩容
    std::mutex          mtx_;
    // Atomic so Deallocate (lock-free path) and Expand (locked path)
    // can both update counters without a data race.
    std::atomic<size_t> total_allocated_;
    std::atomic<size_t> total_freed_;
};

/**
 * SlabAllocator - 多尺寸 Slab 分配器 (动态扩容)
 */
class SlabAllocator {
public:
    SlabAllocator();

    void* Allocate(size_t size) noexcept;
    void Deallocate(void* ptr, size_t size) noexcept;
    size_t GetSlabSize(size_t size) const noexcept;
    bool CanAllocate(size_t size) const noexcept;

private:
    size_t IndexFor(size_t size) const noexcept;

public:
    static constexpr size_t kMaxSlabSize = 4096;
    static constexpr size_t kSlabCount   = 8;
    static constexpr std::array<size_t, kSlabCount> kSizes = {
        16, 32, 64, 128, 256, 512, 1024, 4096
    };

    struct Slab {
        size_t         block_size;
        size_t         blocks_per_chunk;
        std::vector<char*> chunks;
        LockFreeStack  free_list;
        std::mutex     mtx;
    };

    std::array<Slab, kSlabCount> slabs_;
};

/**
 * TlsCache - 线程本地缓存
 */
class TlsCache {
public:
    using AllocFn = std::function<void*()>;
    using FreeFn  = std::function<void(void*)>;

    TlsCache(AllocFn alloc, FreeFn free);

    void* Get();
    void Put(void* ptr);
    size_t CacheSize() const;

private:
    void Refill();
    void Flush();

public:
    static constexpr size_t kBatchSize = 16;
    AllocFn alloc_;
    FreeFn  free_;
    std::vector<void*> cache_;
};

} // namespace legacy_pool

// ============================================================================
// 桥接器: 将旧版 Pool 适配到新版 Allocator 接口
// ============================================================================

class FixedSizePoolBridge : public Allocator {
public:
    static std::unique_ptr<FixedSizePoolBridge> Create(size_t block_size, size_t blocks_per_chunk = 256) {
        return std::make_unique<FixedSizePoolBridge>(block_size, blocks_per_chunk);
    }

    explicit FixedSizePoolBridge(size_t block_size, size_t blocks_per_chunk = 256)
        : pool_(block_size, blocks_per_chunk) {}

    void* allocate(size_t bytes) override {
        if (bytes != pool_.BlockSize()) {
            return nullptr;  // fixed-size pool rejects mismatched requests
        }
        return pool_.Allocate();
    }

    void deallocate(void* ptr) override {
        pool_.Deallocate(ptr);
    }

    size_t allocated_bytes() const override { 
        return pool_.TotalAllocated(); 
    }

    size_t total_bytes() const override { 
        return pool_.TotalAllocated(); 
    }

    size_t block_size() const { return pool_.BlockSize(); }
    size_t block_count() const { return pool_.ChunkCount(); }
    
    // 动态扩容控制
    void SetMaxChunks(size_t max) { pool_.SetMaxChunks(max); }
    void SetGrowFactor(double factor) { pool_.SetGrowFactor(factor); }
    
    // 统计
    size_t free_blocks() const { return pool_.FreeBlockCount(); }

public:
    mutable legacy_pool::FixedSizeMemoryPool pool_;
};

class SlabPoolBridge : public Allocator {
public:
    static std::unique_ptr<SlabPoolBridge> Create() {
        return std::make_unique<SlabPoolBridge>();
    }

    SlabPoolBridge() : allocator_() {}

    void* allocate(size_t bytes) override {
        if (bytes == 0 || bytes > legacy_pool::SlabAllocator::kMaxSlabSize) {
            return nullptr;
        }
        size_t rounded = allocator_.GetSlabSize(bytes);
        void* p = allocator_.Allocate(rounded);
        if (p) {
            std::lock_guard<std::mutex> lock(size_mtx_);
            size_map_[p] = rounded;
        }
        return p;
    }

    void deallocate(void* ptr) override {
        // Recover the original size from the per-block header so the
        // pointer is routed back to the correct slab bucket.
        if (!ptr) return;
        size_t size = 0;
        {
            std::lock_guard<std::mutex> lock(size_mtx_);
            auto it = size_map_.find(ptr);
            if (it == size_map_.end()) {
                // Unknown pointer; do nothing to avoid corrupting a slab.
                return;
            }
            size = it->second;
            size_map_.erase(it);
        }
        allocator_.Deallocate(ptr, size);
    }

    size_t allocated_bytes() const override { return 0; }
    size_t total_bytes() const override { return 0; }

    bool can_allocate(size_t bytes) const {
        return allocator_.CanAllocate(bytes);
    }

public:
    legacy_pool::SlabAllocator allocator_;
    std::unordered_map<void*, size_t> size_map_;
    mutable std::mutex size_mtx_;
};

} // namespace bridges
} // namespace sci

// ============================================================================
// 便捷工厂函数
// ============================================================================
inline std::unique_ptr<sci::bridges::FixedSizePoolBridge> CreateFixedSizePoolBridge(
    size_t block_size, size_t blocks_per_chunk = 256) {
    return std::make_unique<sci::bridges::FixedSizePoolBridge>(block_size, blocks_per_chunk);
}

inline std::unique_ptr<sci::bridges::SlabPoolBridge> CreateSlabPoolBridge() {
    return std::make_unique<sci::bridges::SlabPoolBridge>();
}
