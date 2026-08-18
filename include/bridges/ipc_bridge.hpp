#pragma once
/**
 * @file ipc_bridge.hpp
 * @brief IPC 桥接层 - 队列声明
 */

#include "../core/common.hpp"
#include "impl/ipc_queue_impl.hpp"
#include <atomic>
#include <memory>
#include <string>

namespace sci {
namespace bridges {

// ============================================================================
// 枚举和类型定义
// ============================================================================
enum class ShmState { Created, Opened, Invalid };

// ============================================================================
// SharedMemorySegment
// ============================================================================
class SharedMemorySegment {
public:
    SharedMemorySegment() = default;
    SharedMemorySegment(const SharedMemorySegment&) = delete;
    SharedMemorySegment& operator=(const SharedMemorySegment&) = delete;
    SharedMemorySegment(SharedMemorySegment&& other) noexcept;
    SharedMemorySegment& operator=(SharedMemorySegment&& other) noexcept;
    ~SharedMemorySegment();
    
    static SharedMemorySegment Create(const std::string& name, size_t size);
    static SharedMemorySegment Open(const std::string& name);
    
    void AddRef() noexcept;
    bool Release() noexcept;
    void Destroy() noexcept;
    
    void* BasePtr() noexcept { return mapped_addr_; }
    const void* BasePtr() const noexcept { return mapped_addr_; }
    size_t Size() const noexcept { return mapped_size_; }
    bool IsValid() const noexcept { return mapped_addr_ != nullptr; }
    const std::string& Name() const noexcept { return name_; }
    ShmState State() const noexcept { return state_; }

private:
    std::string name_;
    int fd_ = -1;
    void* mapped_addr_ = nullptr;
    size_t mapped_size_ = 0;
    bool is_creator_ = false;
    ShmState state_ = ShmState::Invalid;
};

// ============================================================================
// SpinLock
// ============================================================================
class SpinLock {
public:
    SpinLock() = default;
    void Lock() { while (locked_.test_and_set(std::memory_order_acquire)) {} }
    void Unlock() { locked_.clear(std::memory_order_release); }
    bool TryLock() { return !locked_.test_and_set(std::memory_order_acquire); }

private:
    std::atomic_flag locked_ = ATOMIC_FLAG_INIT;
};

// ============================================================================
// SPSC/MPSC queues are defined in impl/ipc_queue_impl.hpp. They are
// exposed as sci::bridges::SPSCQueue / MPSCQueue directly; no
// backward-compat macros are needed.

} // namespace bridges
} // namespace sci
