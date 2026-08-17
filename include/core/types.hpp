#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <type_traits>
#include <initializer_list>
#include <cstring>

namespace sci {

// ============================================================================
// Forward Declarations
// ============================================================================
class Tensor;
class Device;

// ============================================================================
// Fundamental Types
// ============================================================================
using int8 = int8_t;
using int16 = int16_t;
using int32 = int32_t;
using int64 = int64_t;
using uint8 = uint8_t;
using uint16 = uint16_t;
using uint32 = uint32_t;
using uint64 = uint64_t;
using index_t = int64_t;
using size_t = std::size_t;

// ============================================================================
// Data Types (DType)
// ============================================================================
enum class DType : uint8_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
    kInt8 = 3,
    kInt32 = 4,
    kInt64 = 5,
    kBool = 6,
    kFloat64 = 7,
    kUInt8 = 8,
};

constexpr size_t kDTypeSize(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return 4;
        case DType::kFloat16: return 2;
        case DType::kBFloat16: return 2;
        case DType::kInt8: return 1;
        case DType::kInt32: return 4;
        case DType::kInt64: return 8;
        case DType::kBool: return 1;
        case DType::kFloat64: return 8;
        case DType::kUInt8: return 1;
        default: return 0;
    }
}

constexpr const char* kDTypeName(DType dtype) {
    switch (dtype) {
        case DType::kFloat32: return "float32";
        case DType::kFloat16: return "float16";
        case DType::kBFloat16: return "bfloat16";
        case DType::kInt8: return "int8";
        case DType::kInt32: return "int32";
        case DType::kInt64: return "int64";
        case DType::kBool: return "bool";
        case DType::kFloat64: return "float64";
        case DType::kUInt8: return "uint8";
        default: return "unknown";
    }
}

// ============================================================================
// Layout Types
// ============================================================================
enum class Layout : uint8_t {
    kContiguous = 0,
    kColumnMajor = 1,
    kStrided = 2,
};

constexpr const char* kLayoutName(Layout layout) {
    switch (layout) {
        case Layout::kContiguous: return "contiguous";
        case Layout::kColumnMajor: return "column_major";
        case Layout::kStrided: return "strided";
        default: return "unknown";
    }
}

// ============================================================================
// Device Types
// ============================================================================
enum class DeviceType : uint8_t {
    kCPU = 0,
    kCUDA = 1,
    kCPUPinned = 2,
};

constexpr const char* kDeviceTypeName(DeviceType type) {
    switch (type) {
        case DeviceType::kCPU: return "cpu";
        case DeviceType::kCUDA: return "cuda";
        case DeviceType::kCPUPinned: return "cpu_pinned";
        default: return "unknown";
    }
}

// ============================================================================
// Memory Alignment
// ============================================================================
constexpr size_t kDefaultAlignment = 64;
constexpr size_t kSimdAlignment = 32;
constexpr size_t kMaxDim = 8;

// ============================================================================
// Tensor Shape
// ============================================================================
class TensorShape {
public:
    TensorShape() = default;
    
    explicit TensorShape(std::initializer_list<index_t> dims) {
        ndims_ = static_cast<index_t>(dims.size());
        size_t i = 0;
        for (auto d : dims) {
            if (i < kMaxDim) dims_[i++] = d;
        }
    }
    
    explicit TensorShape(const std::vector<index_t>& dims) {
        ndims_ = static_cast<index_t>(dims.size());
        for (index_t i = 0; i < ndims_; ++i) {
            dims_[i] = dims[i];
        }
    }
    
    explicit TensorShape(const std::array<index_t, kMaxDim>& dims, index_t ndims) {
        ndims_ = ndims;
        for (index_t i = 0; i < ndims; ++i) {
            dims_[i] = dims[i];
        }
    }
    
    index_t operator[](size_t i) const { return dims_[i]; }
    index_t& operator[](size_t i) { return dims_[i]; }
    
    index_t dim(size_t i) const { return dims_[i]; }
    index_t& dim(size_t i) { return dims_[i]; }
    
    index_t ndims() const { return ndims_; }
    index_t size() const { return ndims_; }
    bool empty() const { return ndims_ == 0; }
    
    index_t num_elements() const {
        index_t n = 1;
        for (index_t i = 0; i < ndims_; ++i) {
            n *= dims_[i];
        }
        return n;
    }
    
    index_t count(index_t start, index_t end) const {
        if (start < 0) start = ndims_ + start;
        if (end < 0) end = ndims_ + end;
        index_t n = 1;
        for (index_t i = start; i < end; ++i) {
            n *= dims_[i];
        }
        return n;
    }
    
    std::vector<index_t> strides() const {
        std::vector<index_t> s(ndims_, 1);
        for (index_t i = ndims_ - 2; i >= 0; --i) {
            s[i] = s[i + 1] * dims_[i + 1];
        }
        return s;
    }
    
    void set_ndims(index_t n) { ndims_ = n; }
    void resize(index_t n) { ndims_ = n; }
    
    std::vector<index_t> dims() const {
        return std::vector<index_t>(dims_.begin(), dims_.begin() + ndims_);
    }
    
    bool operator==(const TensorShape& other) const {
        if (ndims_ != other.ndims_) return false;
        for (index_t i = 0; i < ndims_; ++i) {
            if (dims_[i] != other.dims_[i]) return false;
        }
        return true;
    }
    
    bool operator!=(const TensorShape& other) const {
        return !(*this == other);
    }
    
private:
    std::array<index_t, kMaxDim> dims_ = {};
    index_t ndims_ = 0;
};

// ============================================================================
// Memory Descriptor
// ============================================================================
struct MemoryDesc {
    DeviceType device = DeviceType::kCPU;
    DType dtype = DType::kFloat32;
    Layout layout = Layout::kContiguous;
    size_t alignment = kDefaultAlignment;
    
    size_t bytes_per_element() const { return kDTypeSize(dtype); }
};

// ============================================================================
// Stream Type
// ============================================================================
using StreamHandle = void*;
using EventHandle = void*;

// ============================================================================
// Common Type Aliases
// ============================================================================
template<typename T>
using Vec = std::vector<T>;

template<typename T, size_t N>
using Array = std::array<T, N>;

using String = std::string;

// ============================================================================
// Compile-time Type Helpers
// ============================================================================
template<DType dtype>
struct DTypeToType {};

template<typename T>
struct TypeToDType {};

#define SCI_REGISTER_DTYPE(CPP_TYPE, ENUM_VALUE) \
    template<> struct DTypeToType<ENUM_VALUE> { using type = CPP_TYPE; }; \
    template<> struct TypeToDType<CPP_TYPE> { static constexpr DType value = ENUM_VALUE; };

SCI_REGISTER_DTYPE(float, DType::kFloat32)
SCI_REGISTER_DTYPE(double, DType::kFloat64)
SCI_REGISTER_DTYPE(int32_t, DType::kInt32)
SCI_REGISTER_DTYPE(int64_t, DType::kInt64)
SCI_REGISTER_DTYPE(int8_t, DType::kInt8)
SCI_REGISTER_DTYPE(uint8_t, DType::kUInt8)

#undef SCI_REGISTER_DTYPE

} // namespace sci
