#pragma once
/**
 * @file ipc_queue_impl.hpp
 * @brief IPC队列模板实现 - 无锁算法 + ABA防护
 */

#include "../../core/common.hpp"
#include <atomic>
#include <memory>
#include <cassert>
#include <thread>

namespace sci {
namespace bridges {

// ============================================================================
// SPSC 队列 (无锁单生产者单消费者) - 环形缓冲区实现
// ============================================================================
template<typename T>
class SPSCQueue {
public:
    SPSCQueue() : buffer_(nullptr), capacity_(0) {}
    ~SPSCQueue() { Destroy(); }
    
    bool Init(size_t capacity);
    void Destroy();
    bool Push(const T& item);
    bool Pop(T& item);
    bool Empty() const;
    bool Full() const;
    size_t SizeApprox() const;
    size_t capacity() const { return capacity_; }

private:
    T* buffer_ = nullptr;
    size_t capacity_ = 0;
    std::atomic<size_t> read_idx_{0};
    std::atomic<size_t> write_idx_{0};
};

// ============================================================================
// MPSC 队列 (多生产者单消费者) - 有界序列号环形缓冲区
// ============================================================================
template<typename T>
class MPSCQueue {
public:
    MPSCQueue() : pool_(nullptr), pool_capacity_(0) {}
    ~MPSCQueue() { Destroy(); }
    
    bool Init(size_t capacity);
    void Destroy();
    bool Push(const T& item);
    bool Pop(T& item);
    bool Empty() const;
    size_t SizeApprox() const;
    size_t capacity() const { return pool_capacity_; }
    
    // 批量操作接口
    size_t PushBatch(const T* items, size_t count);
    size_t PopBatch(T* items, size_t count);

    // Explicit capacity bump. The Push path no longer auto-expands;
    // callers that need more headroom must call Reserve first. This
    // avoids the data-loss class of bugs that arise from rewriting
    // the linked list while other threads are mid-flight.
    bool Reserve(size_t new_capacity);

private:
    struct Node {
        std::atomic<size_t> sequence{0};
        T data;
    };

    Node* pool_;
    size_t pool_capacity_;
    alignas(64) std::atomic<size_t> enqueue_pos_{0};
    alignas(64) std::atomic<size_t> dequeue_pos_{0};
};

// ============================================================================
// SPSC 队列实现
// ============================================================================
template<typename T>
bool SPSCQueue<T>::Init(size_t capacity) {
    Destroy();
    if (capacity == 0) return false;
    capacity_ = capacity;
    buffer_ = new T[capacity];
    read_idx_.store(0, std::memory_order_relaxed);
    write_idx_.store(0, std::memory_order_relaxed);
    return true;
}

template<typename T>
void SPSCQueue<T>::Destroy() {
    if (buffer_) {
        delete[] buffer_;
        buffer_ = nullptr;
    }
    capacity_ = 0;
}

template<typename T>
bool SPSCQueue<T>::Push(const T& item) {
    size_t write = write_idx_.load(std::memory_order_relaxed);
    size_t next = (write + 1) % capacity_;
    
    if (next == read_idx_.load(std::memory_order_acquire)) {
        return false;
    }
    
    buffer_[write] = item;
    write_idx_.store(next, std::memory_order_release);
    return true;
}

template<typename T>
bool SPSCQueue<T>::Pop(T& item) {
    size_t read = read_idx_.load(std::memory_order_relaxed);
    
    if (read == write_idx_.load(std::memory_order_acquire)) {
        return false;
    }
    
    item = buffer_[read];
    read_idx_.store((read + 1) % capacity_, std::memory_order_release);
    return true;
}

template<typename T>
bool SPSCQueue<T>::Empty() const {
    return read_idx_.load(std::memory_order_acquire) == 
           write_idx_.load(std::memory_order_acquire);
}

template<typename T>
bool SPSCQueue<T>::Full() const {
    size_t next = (write_idx_.load(std::memory_order_acquire) + 1) % capacity_;
    return next == read_idx_.load(std::memory_order_acquire);
}

template<typename T>
size_t SPSCQueue<T>::SizeApprox() const {
    size_t read = read_idx_.load(std::memory_order_relaxed);
    size_t write = write_idx_.load(std::memory_order_relaxed);
    if (write >= read) return write - read;
    return capacity_ - (read - write);
}

// ============================================================================
// MPSC 队列实现 (有界序列号环形缓冲区)
// ============================================================================

template<typename T>
bool MPSCQueue<T>::Reserve(size_t new_capacity) {
    if (new_capacity <= pool_capacity_) return true;

    // Resizing invalidates every slot address. Reserve is therefore only
    // valid while producers and the consumer are quiescent and the queue is
    // empty. Runtime expansion belongs in a segmented queue, not this type.
    if (enqueue_pos_.load(std::memory_order_acquire) !=
        dequeue_pos_.load(std::memory_order_acquire)) {
        return false;
    }

    Node* new_pool = new Node[new_capacity];
    for (size_t i = 0; i < new_capacity; ++i) {
        new_pool[i].sequence.store(i, std::memory_order_relaxed);
    }

    delete[] pool_;
    pool_ = new_pool;
    pool_capacity_ = new_capacity;
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    return true;
}

template<typename T>
bool MPSCQueue<T>::Init(size_t capacity) {
    Destroy();
    if (capacity < 2) return false;

    pool_capacity_ = capacity;
    pool_ = new Node[pool_capacity_];
    for (size_t i = 0; i < pool_capacity_; ++i) {
        pool_[i].sequence.store(i, std::memory_order_relaxed);
    }

    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
    return true;
}

template<typename T>
void MPSCQueue<T>::Destroy() {
    if (pool_) {
        delete[] pool_;
        pool_ = nullptr;
    }
    pool_capacity_ = 0;
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
}

template<typename T>
bool MPSCQueue<T>::Push(const T& item) {
    if (pool_ == nullptr) return false;

    size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
    Node* node = nullptr;
    while (true) {
        node = &pool_[pos % pool_capacity_];
        const size_t sequence = node->sequence.load(std::memory_order_acquire);
        const auto difference = static_cast<std::intptr_t>(sequence) -
                                static_cast<std::intptr_t>(pos);

        if (difference == 0) {
            if (enqueue_pos_.compare_exchange_weak(
                    pos, pos + 1, std::memory_order_relaxed,
                    std::memory_order_relaxed)) {
                break;
            }
        } else if (difference < 0) {
            return false;
        } else {
            pos = enqueue_pos_.load(std::memory_order_relaxed);
        }
    }

    node->data = item;
    node->sequence.store(pos + 1, std::memory_order_release);
    return true;
}

template<typename T>
bool MPSCQueue<T>::Pop(T& item) {
    if (pool_ == nullptr) return false;

    const size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    Node& node = pool_[pos % pool_capacity_];
    const size_t sequence = node.sequence.load(std::memory_order_acquire);
    const auto difference = static_cast<std::intptr_t>(sequence) -
                            static_cast<std::intptr_t>(pos + 1);
    if (difference != 0) return false;

    item = node.data;
    dequeue_pos_.store(pos + 1, std::memory_order_relaxed);
    node.sequence.store(pos + pool_capacity_, std::memory_order_release);
    return true;
}

template<typename T>
bool MPSCQueue<T>::Empty() const {
    if (pool_ == nullptr) return true;

    const size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
    const Node& node = pool_[pos % pool_capacity_];
    return node.sequence.load(std::memory_order_acquire) != pos + 1;
}

template<typename T>
size_t MPSCQueue<T>::SizeApprox() const {
    return enqueue_pos_.load(std::memory_order_relaxed) -
           dequeue_pos_.load(std::memory_order_relaxed);
}

template<typename T>
size_t MPSCQueue<T>::PushBatch(const T* items, size_t count) {
    size_t pushed = 0;
    for (size_t i = 0; i < count; ++i) {
        if (Push(items[i])) pushed++;
        else break;
    }
    return pushed;
}

template<typename T>
size_t MPSCQueue<T>::PopBatch(T* items, size_t count) {
    size_t popped = 0;
    for (size_t i = 0; i < count; ++i) {
        if (Pop(items[i])) popped++;
        else break;
    }
    return popped;
}

} // namespace bridges
} // namespace sci
