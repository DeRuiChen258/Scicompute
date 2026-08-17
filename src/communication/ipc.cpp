#include "communication/ipc.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace sci {
namespace communication {

// ============================================================================
// SharedMemorySegment Implementation
// ============================================================================
SharedMemorySegment::~SharedMemorySegment() {
    close();
}

bool SharedMemorySegment::create(const std::string& name, size_t size) {
    // Create shared memory file
    fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ == -1) {
        return false;
    }
    
    // Set size
    if (ftruncate(fd_, size) == -1) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    // Map the memory
    data_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        data_ = nullptr;
        return false;
    }
    
    size_ = size;
    return true;
}

bool SharedMemorySegment::open(const std::string& name) {
    fd_ = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd_ == -1) {
        return false;
    }
    
    // Get size
    struct stat stat_buf;
    if (fstat(fd_, &stat_buf) == -1) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    size_ = stat_buf.st_size;
    
    // Map the memory
    data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        data_ = nullptr;
        return false;
    }
    
    return true;
}

void SharedMemorySegment::close() {
    if (data_ != nullptr && data_ != MAP_FAILED) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

// ============================================================================
// IPCManager Implementation
// ============================================================================
IPCManager& IPCManager::Instance() {
    static IPCManager instance;
    return instance;
}

SharedMemorySegment* IPCManager::get_segment(const std::string& name, size_t size) {
    auto it = segments_.find(name);
    if (it != segments_.end()) {
        return it->second.get();
    }
    
    auto segment = std::make_unique<SharedMemorySegment>();
    if (!segment->create(name, size)) {
        // Try to open existing
        if (!segment->open(name)) {
            return nullptr;
        }
    }
    
    auto* ptr = segment.get();
    segments_[name] = std::move(segment);
    return ptr;
}

bool IPCManager::remove_segment(const std::string& name) {
    segments_.erase(name);
    return shm_unlink(name.c_str()) == 0;
}

} // namespace communication
} // namespace sci
