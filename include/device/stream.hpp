#pragma once

#include "../core/common.hpp"
#include <atomic>
#include <mutex>

namespace sci {

// ============================================================================
// Event (forward declared)
// ============================================================================
class Event;

// ============================================================================
// Stream
// ============================================================================
class Stream {
public:
    enum class Priority : int {
        kHigh = 0,
        kNormal = 1,
        kLow = 2,
    };
    
    Stream();
    Stream(void* stream_handle, DeviceType type);
    ~Stream();
    
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&& other) noexcept;
    Stream& operator=(Stream&& other) noexcept;
    
    void* handle() const { return stream_; }
    DeviceType device_type() const { return device_type_; }
    
    void wait(Event& event);
    void synchronize();
    bool is_query_complete() const;
    
    static Stream Create(DeviceType type, Priority priority = Priority::kNormal);
    static Stream& GetCurrent();
    static void SetCurrent(const Stream& stream);
    
    explicit operator bool() const { return stream_ != nullptr; }
    
private:
    void* stream_ = nullptr;
    DeviceType device_type_ = DeviceType::kCPU;
};

// ============================================================================
// Event
// ============================================================================
class Event {
public:
    Event();
    Event(void* event_handle);
    ~Event();
    
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    Event(Event&& other) noexcept;
    Event& operator=(Event&& other) noexcept;
    
    void* handle() const { return event_; }
    
    void record(Stream& stream);
    void synchronize();
    bool query() const;
    float elapsed_ms(const Event& other) const;
    
    static Event Create(DeviceType type);
    
    explicit operator bool() const { return event_ != nullptr; }
    
private:
    void* event_ = nullptr;
    DeviceType device_type_ = DeviceType::kCPU;
};

// ============================================================================
// Stream Pool
// ============================================================================
class StreamPool {
public:
    StreamPool(DeviceType type, size_t max_streams = 4);
    ~StreamPool();
    
    Stream get_stream();
    void return_stream(Stream&& stream);
    
    size_t size() const { return pool_size_; }
    DeviceType device_type() const { return device_type_; }
    
private:
    DeviceType device_type_;
    size_t pool_size_;
    std::vector<Stream> available_streams_;
    std::mutex mutex_;
};

} // namespace sci
