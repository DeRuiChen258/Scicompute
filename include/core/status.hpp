#pragma once

#include "types.hpp"
#include <string>
#include <utility>
#include <functional>
#include <variant>
#include <stdexcept>
#include <cassert>

namespace sci {

// ============================================================================
// Status Code
// ============================================================================
enum class StatusCode : int32_t {
    // 成功
    kOk = 0,
    
    // 通用错误
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
    
    // CUDA/GPU相关错误
    kCudaError = 100,
    kCudaNotAvailable = 101,
    kCudaOutOfMemory = 102,
    kCudaInvalidDevice = 103,
    kCudaKernelNotFound = 104,
    kCudaLaunchFailure = 105,
    kCudaIllegalAddress = 106,
    
    // IPC相关错误
    kIpcError = 200,
    kIpcSegmentNotFound = 201,
    kIpcSegmentExists = 202,
    kIpcPermissionDenied = 203,
    kIpcQueueFull = 204,
    kIpcQueueEmpty = 205,
    kIpcTimeout = 206,
    
    // 内存池相关错误
    kPoolError = 300,
    kPoolOutOfSlots = 301,
    kPoolInvalidBlock = 302,
    kPoolDoubleFree = 303,
    
    // 张量相关错误
    kTensorError = 400,
    kTensorInvalidShape = 401,
    kTensorInvalidDtype = 402,
    kTensorDeviceMismatch = 403,
    kTensorBroadcastFailed = 404,
    
    // 设备相关错误
    kDeviceError = 500,
    kDeviceNotFound = 501,
    kDeviceUnavailable = 502,
    kDeviceInvalidId = 503,
    
    // 调度器相关错误
    kSchedulerError = 600,
    kSchedulerFull = 601,
    kSchedulerClosed = 602,
    kSchedulerTimeout = 603,
};

constexpr bool IsOk(StatusCode code) { return code == StatusCode::kOk; }

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
        case StatusCode::kCudaError: return "CUDA Error";
        case StatusCode::kCudaNotAvailable: return "CUDA Not Available";
        case StatusCode::kCudaOutOfMemory: return "CUDA Out of Memory";
        case StatusCode::kCudaInvalidDevice: return "CUDA Invalid Device";
        case StatusCode::kCudaKernelNotFound: return "CUDA Kernel Not Found";
        case StatusCode::kCudaLaunchFailure: return "CUDA Launch Failure";
        case StatusCode::kCudaIllegalAddress: return "CUDA Illegal Address";
        case StatusCode::kIpcError: return "IPC Error";
        case StatusCode::kIpcSegmentNotFound: return "IPC Segment Not Found";
        case StatusCode::kIpcSegmentExists: return "IPC Segment Exists";
        case StatusCode::kIpcPermissionDenied: return "IPC Permission Denied";
        case StatusCode::kIpcQueueFull: return "IPC Queue Full";
        case StatusCode::kIpcQueueEmpty: return "IPC Queue Empty";
        case StatusCode::kIpcTimeout: return "IPC Timeout";
        case StatusCode::kPoolError: return "Pool Error";
        case StatusCode::kPoolOutOfSlots: return "Pool Out of Slots";
        case StatusCode::kPoolInvalidBlock: return "Pool Invalid Block";
        case StatusCode::kPoolDoubleFree: return "Pool Double Free";
        case StatusCode::kTensorError: return "Tensor Error";
        case StatusCode::kTensorInvalidShape: return "Tensor Invalid Shape";
        case StatusCode::kTensorInvalidDtype: return "Tensor Invalid Dtype";
        case StatusCode::kTensorDeviceMismatch: return "Tensor Device Mismatch";
        case StatusCode::kTensorBroadcastFailed: return "Tensor Broadcast Failed";
        case StatusCode::kDeviceError: return "Device Error";
        case StatusCode::kDeviceNotFound: return "Device Not Found";
        case StatusCode::kDeviceUnavailable: return "Device Unavailable";
        case StatusCode::kDeviceInvalidId: return "Device Invalid ID";
        case StatusCode::kSchedulerError: return "Scheduler Error";
        case StatusCode::kSchedulerFull: return "Scheduler Queue Full";
        case StatusCode::kSchedulerClosed: return "Scheduler Closed";
        case StatusCode::kSchedulerTimeout: return "Scheduler Timeout";
        default: return "Unknown";
    }
}

// ============================================================================
// Status
// ============================================================================
class Status {
public:
    Status() : code_(StatusCode::kOk) {}
    explicit Status(StatusCode code, const std::string& msg = {}) : code_(code), message_(msg) {}
    
    static Status Ok() { return Status(); }
    static Status Error(const std::string& msg = "Unknown error") { return Status(StatusCode::kError, msg); }
    static Status NotFound(const std::string& msg = "Not found") { return Status(StatusCode::kNotFound, msg); }
    static Status InvalidArgument(const std::string& msg = "Invalid argument") { return Status(StatusCode::kInvalidArgument, msg); }
    static Status NotImplemented(const std::string& msg = "Not implemented") { return Status(StatusCode::kNotImplemented, msg); }
    static Status OutOfMemory(const std::string& msg = "Out of memory") { return Status(StatusCode::kOutOfMemory, msg); }
    static Status InternalError(const std::string& msg = "Internal error") { return Status(StatusCode::kInternal, msg); }
    
    // CUDA错误
    static Status CudaError(const std::string& msg = "CUDA error") { return Status(StatusCode::kCudaError, msg); }
    static Status CudaNotAvailable(const std::string& msg = "CUDA not available") { return Status(StatusCode::kCudaNotAvailable, msg); }
    static Status CudaOutOfMemory(const std::string& msg = "CUDA out of memory") { return Status(StatusCode::kCudaOutOfMemory, msg); }
    
    // IPC错误
    static Status IpcError(const std::string& msg = "IPC error") { return Status(StatusCode::kIpcError, msg); }
    static Status IpcSegmentNotFound(const std::string& msg = "IPC segment not found") { return Status(StatusCode::kIpcSegmentNotFound, msg); }
    static Status IpcQueueFull(const std::string& msg = "IPC queue full") { return Status(StatusCode::kIpcQueueFull, msg); }
    static Status IpcQueueEmpty(const std::string& msg = "IPC queue empty") { return Status(StatusCode::kIpcQueueEmpty, msg); }
    static Status IpcTimeout(const std::string& msg = "IPC timeout") { return Status(StatusCode::kIpcTimeout, msg); }
    
    // 内存池错误
    static Status PoolError(const std::string& msg = "Pool error") { return Status(StatusCode::kPoolError, msg); }
    static Status PoolOutOfSlots(const std::string& msg = "Pool out of slots") { return Status(StatusCode::kPoolOutOfSlots, msg); }
    static Status PoolDoubleFree(const std::string& msg = "Pool double free") { return Status(StatusCode::kPoolDoubleFree, msg); }
    
    // 张量错误
    static Status TensorError(const std::string& msg = "Tensor error") { return Status(StatusCode::kTensorError, msg); }
    static Status TensorInvalidShape(const std::string& msg = "Tensor invalid shape") { return Status(StatusCode::kTensorInvalidShape, msg); }
    static Status TensorDeviceMismatch(const std::string& msg = "Tensor device mismatch") { return Status(StatusCode::kTensorDeviceMismatch, msg); }
    
    // 设备错误
    static Status DeviceError(const std::string& msg = "Device error") { return Status(StatusCode::kDeviceError, msg); }
    static Status DeviceNotFound(const std::string& msg = "Device not found") { return Status(StatusCode::kDeviceNotFound, msg); }
    
    // 调度器错误
    static Status SchedulerError(const std::string& msg = "Scheduler error") { return Status(StatusCode::kSchedulerError, msg); }
    static Status SchedulerFull(const std::string& msg = "Scheduler queue full") { return Status(StatusCode::kSchedulerFull, msg); }
    static Status SchedulerClosed(const std::string& msg = "Scheduler closed") { return Status(StatusCode::kSchedulerClosed, msg); }
    
    bool ok() const { return code_ == StatusCode::kOk; }
    StatusCode code() const { return code_; }
    const std::string& message() const { return message_; }
    explicit operator bool() const { return ok(); }
    bool operator==(const Status& o) const { return code_ == o.code_; }
    bool operator!=(const Status& o) const { return !(*this == o); }
    std::string ToString() const { return ok() ? std::string("OK") : std::string(StatusCodeToString(code_)) + ": " + message_; }

private:
    StatusCode code_;
    std::string message_;
};

// ============================================================================
// Expected / Result
// ============================================================================
struct OkType {};

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
    Expected() : has_value_(true) {}
    explicit Expected(T value) : has_value_(true) { new (&storage_) T(std::move(value)); }
    explicit Expected(Unexpected u) : has_value_(false) { new (&storage_) Status(u.error()); }
    
    Expected(const Expected& o) : has_value_(o.has_value_) {
        if (has_value_) new (&storage_) T(*o.get_ptr_()); else new (&storage_) Status(*o.get_err_());
    }
    
    Expected& operator=(const Expected& o) {
        if (this != &o) {
            if (has_value_) { if (!o.has_value_) get_ptr_()->~T(), new (&storage_) Status(*o.get_err_()); else *get_ptr_() = *o.get_ptr_(); }
            else { if (o.has_value_) get_err_()->~Status(), new (&storage_) T(*o.get_ptr_()); else *get_err_() = *o.get_err_(); }
            has_value_ = o.has_value_;
        }
        return *this;
    }
    
    ~Expected() { if (has_value_) get_ptr_()->~T(); else get_err_()->~Status(); }
    
    T& value() & { if (!has_value_) throw std::runtime_error(message()); return *get_ptr_(); }
    const T& value() const & { if (!has_value_) throw std::runtime_error(message()); return *get_ptr_(); }
    T&& value() && { if (!has_value_) throw std::runtime_error(message()); return std::move(*get_ptr_()); }
    
    bool has_value() const { return has_value_; }
    bool ok() const noexcept { return has_value_; }
    explicit operator bool() const noexcept { return has_value_; }
    T* operator->() { return get_ptr_(); }
    const T* operator->() const { return get_ptr_(); }
    T& operator*() { return *get_ptr_(); }
    const T& operator*() const { return *get_ptr_(); }
    Status error() const { return has_value_ ? Status::Ok() : *get_err_(); }
    const std::string& message() const { return has_value_ ? empty_str_ : get_err_()->message(); }

private:
    T* get_ptr_() { return std::launder(reinterpret_cast<T*>(&storage_)); }
    const T* get_ptr_() const { return std::launder(reinterpret_cast<const T*>(&storage_)); }
    Status* get_err_() { return std::launder(reinterpret_cast<Status*>(&storage_)); }
    const Status* get_err_() const { return std::launder(reinterpret_cast<const Status*>(&storage_)); }
    
    static constexpr size_t StorageSize = sizeof(T) > sizeof(Status) ? sizeof(T) : sizeof(Status);
    alignas(T) alignas(Status) mutable std::byte storage_[sizeof(T) > sizeof(Status) ? sizeof(T) : sizeof(Status)];
    bool has_value_;
    static const std::string empty_str_;
};

template<typename T> const std::string Expected<T>::empty_str_{};

template<typename T> using Result = Expected<T>;

template<typename T> Expected<T> MakeUnexpected(Status s) { return Expected<T>(Unexpected(std::move(s))); }

// 通用的 Ok 函数模板
template<typename T>
inline Result<T> Ok(T value) { return Result<T>(std::move(value)); }

} // namespace sci
