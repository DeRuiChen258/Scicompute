#pragma once

#include <cstdlib>
#include <cstdio>
#include "types.hpp"

// ============================================================================
// Compile-time Configuration
// ============================================================================
#ifdef SCI_DEBUG
    #define SCI_DEBUG_MODE 1
#else
    #define SCI_DEBUG_MODE 0
#endif

// ============================================================================
// Assertion
// ============================================================================
#if SCI_DEBUG_MODE
    #define SCI_ASSERT(condition, msg) \
        do { \
            if (!(condition)) { \
                ::sci::detail::AssertionFailure( \
                    __FILE__, __LINE__, __FUNCTION__, #condition, msg); \
            } \
        } while (false)
#else
    #define SCI_ASSERT(condition, msg) ((void)0)
#endif

#define SCI_CHECK(condition, msg) \
    do { \
        if (!(condition)) { \
            ::sci::detail::CheckFailure( \
                __FILE__, __LINE__, __FUNCTION__, #condition, msg); \
        } \
    } while (false)

// ============================================================================
// Likely/Unlikely
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
    #define SCI_LIKELY(x) __builtin_expect(!!(x), 1)
    #define SCI_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define SCI_LIKELY(x) (x)
    #define SCI_UNLIKELY(x) (x)
#endif

// ============================================================================
// Disable Copy/Move
// ============================================================================
#define SCI_DISALLOW_COPY(TypeName) \
    TypeName(const TypeName&) = delete; \
    TypeName& operator=(const TypeName&) = delete;

#define SCI_DISALLOW_MOVE(TypeName) \
    TypeName(TypeName&&) = delete; \
    TypeName& operator=(TypeName&&) = delete;

#define SCI_DISALLOW_COPY_AND_MOVE(TypeName) \
    SCI_DISALLOW_COPY(TypeName) \
    SCI_DISALLOW_MOVE(TypeName)

// ============================================================================
// Maybe Unused
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
    #define SCI_UNUSED __attribute__((unused))
#else
    #define SCI_UNUSED
#endif

// ============================================================================
// Export Macros
// ============================================================================
#if defined(_WIN32)
    #define SCI_EXPORT __declspec(dllexport)
    #define SCI_IMPORT __declspec(dllimport)
#else
    #define SCI_EXPORT __attribute__((visibility("default")))
    #define SCI_IMPORT
#endif

#if defined(SCI_BUILDING_LIBRARY)
    #define SCI_API SCI_EXPORT
#else
    #define SCI_API SCI_IMPORT
#endif

// ============================================================================
// No Discard
// ============================================================================
#if defined(__GNUC__) || defined(__clang__)
    #define SCI_NODISCARD [[nodiscard]]
#elif defined(_MSC_VER)
    #define SCI_NODISCARD _Check_return_
#else
    #define SCI_NODISCARD [[nodiscard]]
#endif

namespace sci {
namespace detail {

[[noreturn]] inline void AssertionFailure(
    const char* file, int line, const char* func,
    const char* condition, const char* msg) {
    std::fprintf(stderr, "[SCI ASSERTION FAILED]\n");
    std::fprintf(stderr, "  File: %s:%d\n", file, line);
    std::fprintf(stderr, "  Function: %s\n", func);
    std::fprintf(stderr, "  Condition: %s\n", condition);
    if (msg) std::fprintf(stderr, "  Message: %s\n", msg);
    std::abort();
}

[[noreturn]] inline void CheckFailure(
    const char* file, int line, const char* func,
    const char* condition, const char* msg) {
    std::fprintf(stderr, "[SCI CHECK FAILED]\n");
    std::fprintf(stderr, "  File: %s:%d\n", file, line);
    std::fprintf(stderr, "  Function: %s\n", func);
    std::fprintf(stderr, "  Condition: %s\n", condition);
    if (msg) std::fprintf(stderr, "  Message: %s\n", msg);
    std::abort();
}

} // namespace detail
} // namespace sci
