#pragma once

#include "types.hpp"
#include <string>
#include <utility>
#include <functional>
#include <variant>
#include <cassert>

namespace sci {

// ============================================================================
// Status Code
// ============================================================================
enum class StatusCode : int32_t {
    kOk = 0,
    kError = 1,
    kNotFound = 2,
    kAlreadyExists = 3,
    kInvalidArgument = 4,
    kNotImplemented = 5,
    kOutOfMemory = 6,
    kResourceExhausted = 7,
    kUnavAILABLE = 8,
    kPermissionDenied = 9,
    kInvalidOperation = 10,
    kDeadlineExceeded = 11,
    kInternal = 12,
    kCancelled = 13,
};

constexpr bool IsOk(StatusCode code) {
    return code == StatusCode::kOk;
}

constexpr const char* StatusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::kOk: return "OK";
        case StatusCode::kError: return "Error";
        case StatusCode::kNotFound: return "Not Found";
        case StatusCode::kAlreadyExists: return "Already Exists";
        case StatusCode::kInvalidArgument: return "Invalid Argument";
        case StatusCode::kNotImplemented: return "Not Implemented";
        case StatusCode::kOutOfMemory: return "Out of Memory";
        case StatusCode::kResourceExhausted: return "Resource Exhausted";
        case StatusCode::kUnavAILABLE: return "Unavailable";
        case StatusCode::kPermissionDenied: return "Permission Denied";
        case StatusCode::kInvalidOperation: return "Invalid Operation";
        case StatusCode::kDeadlineExceeded: return "Deadline Exceeded";
        case StatusCode::kInternal: return "Internal Error";
        case StatusCode::kCancelled: return "Cancelled";
        default: return "Unknown";
    }
}

// ============================================================================
// Status
// ============================================================================
class Status {
public:
    Status() : code_(StatusCode::kOk) {}
    
    explicit Status(StatusCode code, const std::string& message = {})
        : code_(code), message_(message) {}
    
    static Status Ok() { return Status(); }
    static Status Error(const std::string& msg = "Unknown error") {
        return Status(StatusCode::kError, msg);
    }
    static Status NotFound(const std::string& msg = "Not found") {
        return Status(StatusCode::kNotFound, msg);
    }
    static Status InvalidArgument(const std::string& msg = "Invalid argument") {
        return Status(StatusCode::kInvalidArgument, msg);
    }
    static Status NotImplemented(const std::string& msg = "Not implemented") {
        return Status(StatusCode::kNotImplemented, msg);
    }
    static Status OutOfMemory(const std::string& msg = "Out of memory") {
        return Status(StatusCode::kOutOfMemory, msg);
    }
    static Status InternalError(const std::string& msg = "Internal error") {
        return Status(StatusCode::kInternal, msg);
    }
    
    bool ok() const { return code_ == StatusCode::kOk; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }
    
    explicit operator bool() const { return ok(); }
    
    bool operator==(const Status& other) const {
        return code_ == other.code_;
    }
    bool operator!=(const Status& other) const {
        return !(*this == other);
    }
    
    std::string ToString() const {
        if (ok()) return "OK";
        return std::string(StatusCodeToString(code_)) + ": " + message_;
    }
    
    void IgnoreError() const {}

private:
    StatusCode code_;
    std::string message_;
};

// ============================================================================
// Expected<T, E> - Rust-style Result type
// ============================================================================
template<typename T>
class Unexpected {
public:
    explicit Unexpected(Status s) : error_(std::move(s)) {}
    const Status& error() const { return error_; }
private:
    Status error_;
};

template<typename T>
class Expected {
public:
    Expected() requires(std::is_default_constructible_v<T>)
        : has_value_(true) {}
    
    explicit Expected(T value) : has_value_(true) {
        new (&storage_) T(std::move(value));
    }
    
    explicit Expected(Unexpected<T> unexpected)
        : has_value_(false) {
        new (&storage_) Status(unexpected.error());
    }
    
    Expected(const Expected& other) : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(*other.get_ptr_());
        } else {
            new (&storage_) Status(*other.get_error_ptr_());
        }
    }
    
    Expected(Expected&& other) noexcept : has_value_(other.has_value_) {
        if (has_value_) {
            new (&storage_) T(std::move(*other.get_ptr_()));
        } else {
            new (&storage_) Status(std::move(*other.get_error_ptr_()));
        }
    }
    
    Expected& operator=(const Expected& other) {
        if (this != &other) {
            destroy_();
            has_value_ = other.has_value_;
            if (has_value_) {
                new (&storage_) T(*other.get_ptr_());
            } else {
                new (&storage_) Status(*other.get_error_ptr_());
            }
        }
        return *this;
    }
    
    Expected& operator=(Expected&& other) noexcept {
        if (this != &other) {
            destroy_();
            has_value_ = other.has_value_;
            if (has_value_) {
                new (&storage_) T(std::move(*other.get_ptr_()));
            } else {
                new (&storage_) Status(std::move(*other.get_error_ptr_()));
            }
        }
        return *this;
    }
    
    ~Expected() { destroy_(); }
    
    bool has_value() const { return has_value_; }
    explicit operator bool() const { return has_value(); }
    
    T& value() & {
        assert(has_value_ && "Expected has no value");
        return *get_ptr_();
    }
    
    const T& value() const & {
        assert(has_value_ && "Expected has no value");
        return *get_ptr_();
    }
    
    T&& value() && {
        assert(has_value_ && "Expected has no value");
        return std::move(*get_ptr_());
    }
    
    const T& operator*() const { return value(); }
    T& operator*() { return value(); }
    
    const T* operator->() const { return get_ptr_(); }
    T* operator->() { return get_ptr_(); }
    
    const Status& error() const {
        assert(!has_value_ && "Expected has value");
        return *get_error_ptr_();
    }
    
    T value_or(T default_value) const {
        return has_value_ ? *get_ptr_() : default_value;
    }
    
    template<typename U>
    T value_or(U&& default_value) const {
        return has_value_ ? *get_ptr_() : static_cast<T>(std::forward<U>(default_value));
    }

private:
    void destroy_() {
        if (has_value_) {
            get_ptr_()->~T();
        } else {
            get_error_ptr_()->~Status();
        }
    }
    
    T* get_ptr_() { return std::launder(reinterpret_cast<T*>(&storage_)); }
    const T* get_ptr_() const { return std::launder(reinterpret_cast<const T*>(&storage_)); }
    Status* get_error_ptr_() { return std::launder(reinterpret_cast<Status*>(&storage_)); }
    const Status* get_error_ptr_() const { return std::launder(reinterpret_cast<const Status*>(&storage_)); }
    
    alignas(T) alignas(Status) mutable std::byte storage_[sizeof(T) > sizeof(Status) ? sizeof(T) : sizeof(Status)];
    bool has_value_;
};

// Type alias for Result<T>
template<typename T>
using Result = Expected<T>;

// Helper for creating Expected
template<typename T>
Result<T> Ok(T value) {
    return Result<T>(std::move(value));
}

struct OkType {};
inline Result<OkType> Ok() {
    return Result<OkType>(OkType{});
}

template<typename T>
Expected<T> MakeUnexpected(Status status) {
    return Expected<T>(Unexpected<T>(std::move(status)));
}

} // namespace sci
