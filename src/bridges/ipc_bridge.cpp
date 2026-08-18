/**
 * @file ipc_bridge.cpp
 * @brief IPC 桥接层实现 - 非模板部分
 */

#include "bridges/ipc_bridge.hpp"
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace sci {
namespace bridges {

// ============================================================================
// SharedMemorySegment 实现
// ============================================================================

SharedMemorySegment::SharedMemorySegment(SharedMemorySegment&& other) noexcept
    : name_(std::move(other.name_))
    , fd_(other.fd_)
    , mapped_addr_(other.mapped_addr_)
    , mapped_size_(other.mapped_size_)
    , is_creator_(other.is_creator_)
    , state_(other.state_) {
    other.fd_ = -1;
    other.mapped_addr_ = nullptr;
    other.state_ = ShmState::Invalid;
}

SharedMemorySegment& SharedMemorySegment::operator=(SharedMemorySegment&& other) noexcept {
    if (this != &other) {
        Destroy();
        name_ = std::move(other.name_);
        fd_ = other.fd_;
        mapped_addr_ = other.mapped_addr_;
        mapped_size_ = other.mapped_size_;
        is_creator_ = other.is_creator_;
        state_ = other.state_;
        other.fd_ = -1;
        other.mapped_addr_ = nullptr;
        other.state_ = ShmState::Invalid;
    }
    return *this;
}

SharedMemorySegment::~SharedMemorySegment() {
    Destroy();
}

SharedMemorySegment SharedMemorySegment::Create(const std::string& name, size_t size) {
    SharedMemorySegment seg;
    seg.fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (seg.fd_ == -1) return seg;
    if (ftruncate(seg.fd_, size) == -1) {
        close(seg.fd_);
        seg.fd_ = -1;
        return seg;
    }
    seg.mapped_addr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, seg.fd_, 0);
    if (seg.mapped_addr_ == MAP_FAILED) {
        close(seg.fd_);
        seg.fd_ = -1;
        seg.mapped_addr_ = nullptr;
        return seg;
    }
    seg.name_ = name;
    seg.mapped_size_ = size;
    seg.is_creator_ = true;
    seg.state_ = ShmState::Created;
    return seg;
}

SharedMemorySegment SharedMemorySegment::Open(const std::string& name) {
    SharedMemorySegment seg;
    seg.fd_ = shm_open(name.c_str(), O_RDWR, 0666);
    if (seg.fd_ == -1) return seg;
    struct stat stat_buf;
    if (fstat(seg.fd_, &stat_buf) == -1) {
        close(seg.fd_);
        seg.fd_ = -1;
        return seg;
    }
    seg.mapped_size_ = stat_buf.st_size;
    seg.mapped_addr_ = mmap(nullptr, seg.mapped_size_, PROT_READ | PROT_WRITE, MAP_SHARED, seg.fd_, 0);
    if (seg.mapped_addr_ == MAP_FAILED) {
        close(seg.fd_);
        seg.fd_ = -1;
        seg.mapped_addr_ = nullptr;
        return seg;
    }
    seg.name_ = name;
    seg.is_creator_ = false;
    seg.state_ = ShmState::Opened;
    return seg;
}

void SharedMemorySegment::AddRef() noexcept {}
bool SharedMemorySegment::Release() noexcept { return true; }

void SharedMemorySegment::Destroy() noexcept {
    if (mapped_addr_ != nullptr && mapped_addr_ != MAP_FAILED) {
        munmap(mapped_addr_, mapped_size_);
        mapped_addr_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    if (is_creator_) {
        shm_unlink(name_.c_str());
    }
    state_ = ShmState::Invalid;
}

// ============================================================================
// 便捷工厂实现
// ============================================================================

std::unique_ptr<SharedMemorySegment> CreateSharedMemory(
    const std::string& name, size_t size) {
    auto seg = SharedMemorySegment::Create(name, size);
    if (seg.IsValid()) {
        return std::make_unique<SharedMemorySegment>(std::move(seg));
    }
    return nullptr;
}

std::unique_ptr<SharedMemorySegment> OpenSharedMemory(const std::string& name) {
    auto seg = SharedMemorySegment::Open(name);
    if (seg.IsValid()) {
        return std::make_unique<SharedMemorySegment>(std::move(seg));
    }
    return nullptr;
}

} // namespace bridges
} // namespace sci
