#pragma once

// ============================================================================
// Platform Detection
// ============================================================================
#if defined(__linux__)
    #define SCI_PLATFORM_LINUX 1
    #define SCI_PLATFORM_NAME "Linux"
#elif defined(_WIN32)
    #define SCI_PLATFORM_WINDOWS 1
    #define SCI_PLATFORM_NAME "Windows"
#elif defined(__APPLE__)
    #define SCI_PLATFORM_MACOS 1
    #define SCI_PLATFORM_NAME "macOS"
#else
    #define SCI_PLATFORM_UNKNOWN 1
    #define SCI_PLATFORM_NAME "Unknown"
#endif

// ============================================================================
// Compiler Detection
// ============================================================================
#if defined(__clang__)
    #define SCI_COMPILER_CLANG 1
    #define SCI_COMPILER_NAME "Clang"
    #define SCI_COMPILER_VERSION __clang_major__ * 100 + __clang_minor__
#elif defined(__GNUC__)
    #define SCI_COMPILER_GCC 1
    #define SCI_COMPILER_NAME "GCC"
    #define SCI_COMPILER_VERSION __GNUC__ * 100 + __GNUC_MINOR__
#elif defined(_MSC_VER)
    #define SCI_COMPILER_MSVC 1
    #define SCI_COMPILER_NAME "MSVC"
    #define SCI_COMPILER_VERSION _MSC_VER
#else
    #define SCI_COMPILER_UNKNOWN 1
    #define SCI_COMPILER_NAME "Unknown"
#endif

// ============================================================================
// Architecture Detection
// ============================================================================
#if defined(__x86_64__) || defined(_M_X64)
    #define SCI_ARCH_X86_64 1
    #define SCI_ARCH_NAME "x86_64"
#elif defined(__aarch64__) || defined(_M_ARM64)
    #define SCI_ARCH_ARM64 1
    #define SCI_ARCH_NAME "ARM64"
#elif defined(__arm__) || defined(_M_ARM)
    #define SCI_ARCH_ARM 1
    #define SCI_ARCH_NAME "ARM"
#else
    #define SCI_ARCH_UNKNOWN 1
    #define SCI_ARCH_NAME "Unknown"
#endif

// ============================================================================
// Feature Detection
// ============================================================================

// SIMD
#if defined(__SSE__) || (defined(_M_X64) && !defined(__clang__))
    #define SCI_HAS_SSE 1
#endif

#if defined(__SSE2__) || (defined(_M_X64) && !defined(__clang__))
    #define SCI_HAS_SSE2 1
#endif

#if defined(__AVX__)
    #define SCI_HAS_AVX 1
#endif

#if defined(__AVX2__)
    #define SCI_HAS_AVX2 1
#endif

#if defined(__AVX512F__)
    #define SCI_HAS_AVX512 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    #define SCI_HAS_NEON 1
#endif

// C++ Standard
#if __cplusplus >= 202302L
    #define SCI_CPP23 1
#elif __cplusplus >= 202002L
    #define SCI_CPP20 1
#elif __cplusplus >= 201703L
    #define SCI_CPP17 1
#else
    #define SCI_CPP14 1
#endif

// ============================================================================
// Utility Macros
// ============================================================================
#define SCI_STRINGIFY_IMPL(x) #x
#define SCI_STRINGIFY(x) SCI_STRINGIFY_IMPL(x)

#define SCI_CONCAT_IMPL(a, b) a##b
#define SCI_CONCAT(a, b) SCI_CONCAT_IMPL(a, b)

#define SCI_HAS_CUDA (defined(SCI_HAS_CUDA) && SCI_HAS_CUDA)
