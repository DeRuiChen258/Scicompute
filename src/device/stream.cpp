#include "device/stream.hpp"
#include "core/macros.hpp"

namespace sci {

Stream::Stream() = default;

Stream::Stream(void* stream_handle, DeviceType type)
    : stream_(stream_handle), device_type_(type) {}

Stream::~Stream() = default;

Stream::Stream(Stream&& other) noexcept
    : stream_(other.stream_), device_type_(other.device_type_) {
    other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
    if (this != &other) {
        stream_ = other.stream_;
        device_type_ = other.device_type_;
        other.stream_ = nullptr;
    }
    return *this;
}

void Stream::wait(Event& event) {}

void Stream::synchronize() {}

bool Stream::is_query_complete() const {
    return true;
}

Stream Stream::Create(DeviceType type, Priority priority) {
    return Stream(nullptr, type);
}

Stream& Stream::GetCurrent() {
    thread_local Stream current;
    return current;
}

void Stream::SetCurrent(const Stream& stream) {
    GetCurrent() = std::move(const_cast<Stream&>(stream));
}

Event::Event() = default;

Event::Event(void* event_handle) : event_(event_handle) {}

Event::~Event() = default;

Event::Event(Event&& other) noexcept
    : event_(other.event_), device_type_(other.device_type_) {
    other.event_ = nullptr;
}

Event& Event::operator=(Event&& other) noexcept {
    if (this != &other) {
        event_ = other.event_;
        device_type_ = other.device_type_;
        other.event_ = nullptr;
    }
    return *this;
}

void Event::record(Stream& stream) {}

void Event::synchronize() {}

bool Event::query() const {
    return true;
}

float Event::elapsed_ms(const Event& other) const {
    return 0.0f;
}

Event Event::Create(DeviceType type) {
    return Event(nullptr);
}

StreamPool::StreamPool(DeviceType type, size_t max_streams)
    : device_type_(type), pool_size_(max_streams) {
    for (size_t i = 0; i < max_streams; ++i) {
        available_streams_.push_back(Stream::Create(type));
    }
}

StreamPool::~StreamPool() = default;

Stream StreamPool::get_stream() {
    std::lock_guard lock(mutex_);
    if (!available_streams_.empty()) {
        Stream s = std::move(available_streams_.back());
        available_streams_.pop_back();
        return s;
    }
    return Stream::Create(device_type_);
}

void StreamPool::return_stream(Stream&& stream) {
    std::lock_guard lock(mutex_);
    if (available_streams_.size() < pool_size_) {
        available_streams_.push_back(std::move(stream));
    }
}

} // namespace sci
