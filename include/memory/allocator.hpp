#pragma once

#include "../core/common.hpp"
#include "../device/device.hpp"

namespace sci {

// ============================================================================
// Allocator Interface
// ============================================================================
class Allocator {
public:
    virtual ~Allocator() = default;
    
    virtual void* allocate(size_t bytes) = 0;
    virtual void deallocate(void* ptr) = 0;
    virtual size_t allocated_bytes() const = 0;
    virtual size_t total_bytes() const = 0;
    virtual void clear() {}
    
    // Allocation stats
    struct Stats {
        size_t num_allocations = 0;
        size_t num_deallocations = 0;
        size_t current_bytes = 0;
        size_t peak_bytes = 0;
        size_t total_bytes = 0;
    };
    
    virtual Stats stats() const { return {}; }
    
protected:
    Allocator() = default;
    SCI_DISALLOW_COPY_AND_MOVE(Allocator);
};

// ============================================================================
// Host Allocator
// ============================================================================
class HostAllocator final : public Allocator {
public:
    static HostAllocator& Instance();
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
    size_t allocated_bytes() const override { return stats_.current_bytes; }
    size_t total_bytes() const override { return stats_.total_bytes; }
    void clear() override;
    
    Stats stats() const override;
    
private:
    HostAllocator();
    
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocation_map_;
    Stats stats_;
};

// ============================================================================
// Aligned Allocator
// ============================================================================
class AlignedAllocator : public Allocator {
public:
    explicit AlignedAllocator(size_t alignment = kDefaultAlignment);
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
    size_t allocated_bytes() const override { return stats_.current_bytes; }
    size_t total_bytes() const override { return stats_.total_bytes; }
    void clear() override;
    
    Stats stats() const override { return stats_; }
    size_t alignment() const { return alignment_; }
    
private:
    size_t alignment_;
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocation_map_;
    Stats stats_;
};

// ============================================================================
// Pinned (Page-locked) Memory Allocator
// ============================================================================
class PinnedAllocator final : public Allocator {
public:
    static PinnedAllocator& Instance();
    
    void* allocate(size_t bytes) override;
    void deallocate(void* ptr) override;
    
    size_t allocated_bytes() const override { return stats_.current_bytes; }
    size_t total_bytes() const override { return stats_.total_bytes; }
    void clear() override;
    
    Stats stats() const override;
    
private:
    PinnedAllocator();
    
    mutable std::mutex mutex_;
    std::unordered_map<void*, size_t> allocation_map_;
    Stats stats_;
};

} // namespace sci
