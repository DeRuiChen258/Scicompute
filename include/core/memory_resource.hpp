#pragma once
/**
 * @file memory_resource.hpp
 * @brief RAII内存管理封装
 */

#include <memory>
#include <cstdlib>
#include <cstring>

namespace sci {

// ============================================================================
// RAII内存块包装器
// ============================================================================
class MemoryBlock {
public:
    MemoryBlock() : data_(nullptr), size_(0) {}
    
    explicit MemoryBlock(size_t size) : size_(size) {
        data_ = ::operator new(size);
    }
    
    ~MemoryBlock() {
        reset();
    }
    
    // 禁止拷贝
    MemoryBlock(const MemoryBlock&) = delete;
    MemoryBlock& operator=(const MemoryBlock&) = delete;
    
    // 移动语义
    MemoryBlock(MemoryBlock&& other) noexcept 
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    MemoryBlock& operator=(MemoryBlock&& other) noexcept {
        if (this != &other) {
            reset();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }
    bool valid() const { return data_ != nullptr; }
    
    void reset() {
        if (data_) {
            ::operator delete(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }
    
    // 显式获取指针
    template<typename T>
    T* as() { return static_cast<T*>(data_); }
    
    template<typename T>
    const T* as() const { return static_cast<const T*>(data_); }

private:
    void* data_;
    size_t size_;
};

// ============================================================================
// 唯一拥有权内存指针
// ============================================================================
template<typename T>
using UniquePtr = std::unique_ptr<T, std::function<void(void*)>>;

template<typename T>
UniquePtr<T> make_unique_ptr(size_t size) {
    return UniquePtr<T>(static_cast<T*>(::operator new(size)), 
                        [](void* p) { ::operator delete(p); });
}

// ============================================================================
// 共享内存指针
// ============================================================================
template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
SharedPtr<T> make_shared_ptr(T* ptr) {
    return SharedPtr<T>(ptr, [](T* p) { delete p; });
}

} // namespace sci
