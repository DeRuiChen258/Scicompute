#pragma once

// Communication Layer - IPC integration header
// This header provides access to IPC communication capabilities
// Original IPC library: linux_cpp/ipc/

#include "../core/common.hpp"

namespace sci {
namespace communication {

// Forward declarations
class SharedMemorySegment;
class MPSCQueue;
class SPSCQueue;
class SpinLock;

// ============================================================================
// IPC Configuration
// ============================================================================
struct IPCConfig {
    size_t segment_size = 1024 * 1024;  // 1MB default segment size
    bool use_huge_pages = false;
    bool use_cuda_ipc = false;
};

// ============================================================================
// Shared Memory Segment
// ============================================================================
class SharedMemorySegment {
public:
    SharedMemorySegment() = default;
    ~SharedMemorySegment();
    
    // Create a new shared memory segment
    bool create(const std::string& name, size_t size);
    
    // Open an existing shared memory segment
    bool open(const std::string& name);
    
    // Close the segment
    void close();
    
    // Get pointer to the shared memory
    void* data() { return data_; }
    const void* data() const { return data_; }
    
    // Get size of the segment
    size_t size() const { return size_; }
    
    // Check if segment is valid
    explicit operator bool() const { return data_ != nullptr; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
};

// ============================================================================
// Simple MPSC Queue (lock-free for single producer, multiple consumers)
// ============================================================================
template<typename T>
class MPSCQueue {
public:
    MPSCQueue() = default;
    ~MPSCQueue();
    
    bool init(size_t capacity);
    void destroy();
    
    // Push from producer (single producer, no locks needed)
    bool push(const T& item);
    
    // Pop from consumer (multi-consumer, needs coordination)
    bool pop(T& item);
    
    // Check if empty
    bool empty() const { return head_.load() >= tail_.load(); }
    
    // Get approximate size
    size_t size_approx() const;
    
private:
    struct Node {
        std::atomic<Node*> next{nullptr};
        T data;
    };
    
    std::atomic<Node*> head_{nullptr};
    std::atomic<Node*> tail_{nullptr};
    Node* node_pool_ = nullptr;
    size_t capacity_ = 0;
};

// ============================================================================
// Simple SPSC Queue (lock-free single producer, single consumer)
// ============================================================================
template<typename T>
class SPSCQueue {
public:
    SPSCQueue() = default;
    ~SPSCQueue();
    
    bool init(size_t capacity);
    void destroy();
    
    // Push from producer
    bool push(const T& item);
    
    // Pop from consumer
    bool pop(T& item);
    
    // Check if empty
    bool empty() const;
    
    // Check if full
    bool full() const;
    
    // Get approximate size
    size_t size_approx() const;

private:
    T* buffer_ = nullptr;
    size_t capacity_ = 0;
    std::atomic<size_t> write_pos_{0};
    std::atomic<size_t> read_pos_{0};
};

// ============================================================================
// Simple SpinLock
// ============================================================================
class SpinLock {
public:
    SpinLock() = default;
    
    void lock() {
        while (locked_.test_and_set(std::memory_order_acquire)) {
            // Spin - could add backoff here
        }
    }
    
    void unlock() {
        locked_.clear(std::memory_order_release);
    }
    
    bool try_lock() {
        return !locked_.test_and_set(std::memory_order_acquire);
    }

private:
    std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};

// ============================================================================
// Global IPC Manager
// ============================================================================
class IPCManager {
public:
    static IPCManager& Instance();
    
    // Get configuration
    IPCConfig& config() { return config_; }
    
    // Create/open shared memory segment
    SharedMemorySegment* get_segment(const std::string& name, size_t size);
    
    // Remove a shared memory segment
    bool remove_segment(const std::string& name);

private:
    IPCManager() = default;
    
    IPCConfig config_;
    std::map<std::string, std::unique_ptr<SharedMemorySegment>> segments_;
};

} // namespace communication
} // namespace sci
