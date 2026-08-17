#include "memory/allocator.hpp"
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace sci {

// ============================================================================
// Host Allocator Implementation
// ============================================================================
HostAllocator::HostAllocator() = default;

HostAllocator& HostAllocator::Instance() {
    static HostAllocator instance;
    return instance;
}

void* HostAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    
    void* ptr = nullptr;
#if defined(__linux__)
    if (posix_memalign(&ptr, kDefaultAlignment, bytes) != 0) {
        return nullptr;
    }
#else
    ptr = std::malloc(bytes);
#endif
    
    if (ptr) {
        std::memset(ptr, 0, bytes);
        std::lock_guard lock(mutex_);
        allocation_map_[ptr] = bytes;
        stats_.num_allocations++;
        stats_.current_bytes += bytes;
        stats_.total_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    }
    return ptr;
}

void HostAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard lock(mutex_);
    auto it = allocation_map_.find(ptr);
    if (it != allocation_map_.end()) {
        stats_.current_bytes -= it->second;
        stats_.num_deallocations++;
        allocation_map_.erase(it);
    }
    std::free(ptr);
}

void HostAllocator::clear() {
    std::lock_guard lock(mutex_);
    // Note: This only clears the tracking map, not actual allocations
    allocation_map_.clear();
}

Allocator::Stats HostAllocator::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

// ============================================================================
// Aligned Allocator Implementation
// ============================================================================
AlignedAllocator::AlignedAllocator(size_t alignment)
    : alignment_(alignment) {
    // Validate alignment (must be power of 2)
    SCI_ASSERT((alignment & (alignment - 1)) == 0, "Alignment must be power of 2");
}

void* AlignedAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    
    void* ptr = nullptr;
#if defined(__linux__)
    if (posix_memalign(&ptr, alignment_, bytes) != 0) {
        return nullptr;
    }
#else
    ptr = std::aligned_alloc(alignment_, bytes);
#endif
    
    if (ptr) {
        std::memset(ptr, 0, bytes);
        std::lock_guard lock(mutex_);
        allocation_map_[ptr] = bytes;
        stats_.num_allocations++;
        stats_.current_bytes += bytes;
        stats_.total_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    }
    return ptr;
}

void AlignedAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard lock(mutex_);
    auto it = allocation_map_.find(ptr);
    if (it != allocation_map_.end()) {
        stats_.current_bytes -= it->second;
        stats_.num_deallocations++;
        allocation_map_.erase(it);
    }
    std::free(ptr);
}

void AlignedAllocator::clear() {
    std::lock_guard lock(mutex_);
    allocation_map_.clear();
}

// ============================================================================
// Pinned Allocator Implementation
// ============================================================================
PinnedAllocator::PinnedAllocator() = default;

PinnedAllocator& PinnedAllocator::Instance() {
    static PinnedAllocator instance;
    return instance;
}

void* PinnedAllocator::allocate(size_t bytes) {
    if (bytes == 0) return nullptr;
    
    void* ptr = nullptr;
#if defined(__linux__)
    // Use mlock for page-locked memory
    if (posix_memalign(&ptr, kDefaultAlignment, bytes) == 0) {
        std::memset(ptr, 0, bytes);
        // Note: Actually pinning memory requires root or CAP_IPC_LOCK
        std::lock_guard lock(mutex_);
        allocation_map_[ptr] = bytes;
        stats_.num_allocations++;
        stats_.current_bytes += bytes;
        stats_.total_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    }
#else
    ptr = std::malloc(bytes);
    if (ptr) {
        std::memset(ptr, 0, bytes);
        std::lock_guard lock(mutex_);
        allocation_map_[ptr] = bytes;
        stats_.num_allocations++;
        stats_.current_bytes += bytes;
        stats_.total_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.current_bytes);
    }
#endif
    return ptr;
}

void PinnedAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard lock(mutex_);
    auto it = allocation_map_.find(ptr);
    if (it != allocation_map_.end()) {
        stats_.current_bytes -= it->second;
        stats_.num_deallocations++;
        allocation_map_.erase(it);
    }
    std::free(ptr);
}

void PinnedAllocator::clear() {
    std::lock_guard lock(mutex_);
    allocation_map_.clear();
}

Allocator::Stats PinnedAllocator::stats() const {
    std::lock_guard lock(mutex_);
    return stats_;
}

} // namespace sci
