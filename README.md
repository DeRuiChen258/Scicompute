# SciComputeInfra - AI Scientific Computing Infrastructure

## 中文版本

**高性能计算运行时 + CUDA Kernel 库 + Tensor 运行时 + 调度系统 + 基准测试体系**

---

### 项目背景

SciComputeInfra 不是从零开始的项目，而是整合了多个已有 AI/RL/HPC 项目技术资产的统一升级版本。

### 旧项目映射关系

| 源项目 | 源路径 | 贡献模块 | 复用方式 |
|--------|--------|---------|----------|
| **linux_cpp/Pool** | `Memory_Pool.cpp` | 无锁内存池、Slab 分配器、TLS 缓存 | ✅ 已集成到 `include/memory/memory_pool.hpp` |
| **linux_cpp/Pool** | `Thread_Pool.cpp` | 生产级线程池 | ✅ 已集成到 `include/scheduler/thread_pool.hpp` |
| **linux_cpp/ipc** | `include/ipc/*` | IPC 通信库（队列、共享内存、同步） | ✅ 已集成到 `include/communication/ipc.hpp` |
| **RL_infra/vllm** | `src/kernels/*.cu` | CUDA Kernel (LayerNorm, Attention, Softmax 等) | ✅ 已集成到 `cuda/kernels/` |
| **RL_infra/Turbol** | `turbol/include/*` | 分布式 RL 运行时设计 | 🔄 设计参考，调度架构 |

### 核心特性

- **高性能张量计算**: 支持 FP32/FP16/BF16/INT8 等数据类型
- **统一内存管理**: 内存池、Slab 分配器、TLS 缓存（来自 linux_cpp/Pool）
- **CUDA Kernel 库**: 优化的 GPU 算子（来自 vLLM）
  - LayerNorm / RMSNorm
  - Softmax / LogSoftmax
  - Attention / FlashAttention
  - Elementwise 操作
  - Reduction 操作
- **线程调度**: 生产级线程池，支持任务图执行
- **IPC 通信**: 进程间高速通信，支持共享内存（来自 linux_cpp/ipc）
- **基准测试**: 内置性能测试框架

### 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Applications                               │
│              (LLM / RL Training / Scientific Computing)              │
├─────────────────────────────────────────────────────────────────────┤
│                      High Level Compute API                          │
├─────────────────────────────────────────────────────────────────────┤
│                     AI Scientific Computing Layer                     │
│     Tensor Operations | Linear Algebra | Attention | Normalization   │
├─────────────────────────────────────────────────────────────────────┤
│                        Runtime Layer                                 │
│   Scheduler | Task Graph | Async Runtime | Pipeline Engine           │
├─────────────────────────────────────────────────────────────────────┤
│                        Device Layer                                  │
│              CUDA Backend | CPU Backend | Multi-GPU                 │
├─────────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                             │
│         IPC | Thread Pool | Memory Pool | Profiler                  │
├─────────────────────────────────────────────────────────────────────┤
│                          Hardware                                   │
│                     CUDA GPU | CPU | NUMA                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
SciComputeInfra/
├── CMakeLists.txt              # 主构建配置
│
├── include/
│   ├── core/                   # 核心类型和工具
│   │   ├── types.hpp          # DType, Layout, TensorShape, DeviceType
│   │   ├── status.hpp         # Result/Expected 类型
│   │   ├── macros.hpp         # 编译期宏
│   │   ├── config.hpp         # 配置管理
│   │   ├── timer.hpp          # 计时器
│   │   └── version.hpp        # 版本信息
│   │
│   ├── tensor/                # 张量核心
│   │   ├── tensor.hpp         # Tensor, TensorView, TensorOptions
│   │   └── tensor_shape.hpp   # Shape 工具
│   │
│   ├── memory/                 # 内存管理
│   │   ├── allocator.hpp       # 分配器接口
│   │   ├── memory_pool.hpp    # ⭐ 来自 linux_cpp/Pool
│   │   │                        #   - LockFreeStack
│   │   │                        #   - FixedSizeMemoryPool
│   │   │                        #   - SlabAllocator
│   │   │                        #   - TlsCache
│   │   └── buffer_handle.hpp   # RAII 缓冲区封装
│   │
│   ├── device/                 # 设备抽象
│   │   ├── device.hpp         # Device 基类, CPUDevice
│   │   └── stream.hpp          # CUDA Stream 抽象
│   │
│   ├── scheduler/              # 任务调度
│   │   ├── thread_pool.hpp    # ⭐ 来自 linux_cpp/Pool
│   │   ├── task.hpp            # Task 基类
│   │   └── task_graph.hpp      # DAG 任务图
│   │
│   ├── math/                   # 数学运算
│   │   ├── elementwise.hpp     # add, mul, sub, div 等
│   │   ├── reduction.hpp       # sum, mean, max, min 等
│   │   ├── softmax.hpp         # Softmax 及其变体
│   │   └── normalization.hpp   # LayerNorm, RMSNorm
│   │
│   ├── benchmark/              # 基准测试
│   │   ├── benchmark_case.hpp   # Benchmark 用例
│   │   └── benchmark_runner.hpp  # 运行器
│   │
│   ├── communication/           # 通信层
│   │   └── ipc.hpp             # ⭐ 来自 linux_cpp/ipc
│   │                              #   - SharedMemorySegment
│   │                              #   - MPSCQueue
│   │                              #   - SPSCQueue
│   │                              #   - SpinLock
│   │
│   └── kernel/                  # Kernel 注册表
│       └── kernel_registry.hpp
│
├── src/                        # 实现文件
│   ├── core/
│   ├── tensor/
│   ├── memory/
│   ├── device/
│   ├── scheduler/
│   ├── math/
│   ├── benchmark/
│   └── communication/
│
├── cuda/
│   └── kernels/                # ⭐ 来自 RL_infra/vllm
│       ├── layernorm.cu/.cuh   # LayerNorm/RMSNorm Kernel
│       ├── softmax.cu/.cuh     # Softmax Kernel
│       ├── attention.cu/.cuh   # Attention Kernel
│       ├── elementwise.cu/.cuh # Elementwise Kernel
│       └── reduction.cu/.cuh   # Reduction Kernel
│
├── tests/
│   └── unit/                   # 单元测试
│       ├── test_tensor.cpp
│       ├── test_math.cpp
│       ├── test_memory.cpp
│       └── test_scheduler.cpp
│
├── benchmarks/                 # 基准测试
│   ├── bench_elementwise.cpp
│   ├── bench_reduction.cpp
│   └── bench_softmax.cpp
│
├── examples/                    # 示例程序
│   ├── simple_tensor.cpp        # 基础张量演示
│   └── cuda_kernel_demo.cu     # CUDA Kernel 演示
│
├── scripts/                     # 脚本
│   ├── build.sh
│   └── run_benchmarks.sh
│
├── TASK_PLAN.md                 # 项目任务计划
├── DESIGN.md                   # 设计文档
└── README.md                   # 本文件
```

### 快速开始

```bash
# 基本构建 (仅 CPU)
cd SciComputeInfra
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=OFF
make -j4

# 运行示例
./examples/simple_tensor
```

### 构建选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `SCI_ENABLE_CUDA` | 启用 CUDA 支持 | OFF |
| `SCI_ENABLE_OPENMP` | 启用 OpenMP 并行 | ON |
| `SCI_BUILD_TESTS` | 构建单元测试 | OFF |
| `SCI_BUILD_BENCHMARKS` | 构建基准测试 | OFF |

### 依赖

- C++20 编译器 (GCC 12+, Clang 15+)
- CMake 3.20+
- Threads (pthreads)
- OpenMP 4.5+ (可选)
- CUDA Toolkit 11.0+ (可选，用于 GPU 支持)
- GTest (可选，用于测试)
- Google Benchmark (可选，用于基准测试)

### 与旧项目详细关联说明

#### 1. linux_cpp/Pool → memory_pool.hpp

```cpp
// 来自 linux_cpp/Pool/Memory_Pool.cpp 的核心组件已集成:

// 无锁栈 - 用于高性能内存块管理
class LockFreeStack {
    // Push/Pop 操作
};

// 单尺寸内存池 - 固定大小对象的分配
class FixedSizeMemoryPool {
    // Allocate/Deallocate
};

// Slab 分配器 - 多尺寸内存池
class SlabAllocator {
    // 支持 16, 32, 64, 128, 256, 512, 1024, 4096 字节
};

// TLS 缓存 - 减少跨核 CAS 竞争
class TlsCache {
    // 批量预取和释放
};
```

#### 2. linux_cpp/Pool → thread_pool.hpp

```cpp
// 来自 linux_cpp/Pool/Thread_Pool.cpp:

class ThreadPool {
    // 支持的配置:
    // - min_threads / max_threads
    // - idle_timeout_ms
    
    // 核心功能:
    // - Enqueue() 返回 std::future
    // - WaitAll() 批量等待
    // - Pause/Resume 控制
    // - Resize 动态调整
};
```

#### 3. linux_cpp/ipc → communication/ipc.hpp

```cpp
// 来自 linux_cpp/ipc/include/ipc/*:

// 共享内存段
class SharedMemorySegment {
    bool create(const std::string& name, size_t size);
    bool open(const std::string& name);
    void close();
};

// MPSC 队列 (单生产者多消费者)
template<typename T>
class MPSCQueue {};

// SPSC 队列 (单生产者单消费者)
template<typename T>
class SPSCQueue {};

// 自旋锁
class SpinLock {};
```

#### 4. RL_infra/vllm → cuda/kernels/

```cpp
// 来自 RL_infra/vllm/src/kernels/*.cu 的 CUDA Kernel:

// LayerNorm - 融合 kernel，warp 级归约
launch_layer_norm(out, in, weight, bias, rows, cols, eps, stream);
launch_rms_norm(out, in, weight, rows, cols, eps, stream);

// Softmax - 数值稳定实现
launch_softmax(out, x, n, dim, stream);

// Attention - FlashAttention 风格
launch_attention(out, q, k, v, ...);

// Elementwise - fused add/mul/sub/div
launch_add(out, a, b, n, stream);
```

### 版本历史

- **0.1.0** (2026-08-17): 初始版本，核心框架完成

---

## English Version

**High-Performance Computing Runtime + CUDA Kernel Library + Tensor Runtime + Scheduling System + Benchmark Suite**

---

### Project Background

SciComputeInfra is not built from scratch, but a unified upgrade integrating technology assets from multiple existing AI/RL/HPC projects.

### Legacy Project Mapping

| Source Project | Source Path | Contributed Module | Integration |
|----------------|-------------|-------------------|-------------|
| **linux_cpp/Pool** | `Memory_Pool.cpp` | Lock-free memory pool, Slab allocator, TLS cache | ✅ Integrated into `include/memory/memory_pool.hpp` |
| **linux_cpp/Pool** | `Thread_Pool.cpp` | Production-grade thread pool | ✅ Integrated into `include/scheduler/thread_pool.hpp` |
| **linux_cpp/ipc** | `include/ipc/*` | IPC library (queues, shared memory, sync) | ✅ Integrated into `include/communication/ipc.hpp` |
| **RL_infra/vllm** | `src/kernels/*.cu` | CUDA Kernels (LayerNorm, Attention, Softmax) | ✅ Integrated into `cuda/kernels/` |
| **RL_infra/Turbol** | `turbol/include/*` | Distributed RL runtime design | 🔄 Design reference, scheduling architecture |

### Core Features

- **High-Performance Tensor Computing**: Supports FP32/FP16/BF16/INT8 data types
- **Unified Memory Management**: Memory pool, Slab allocator, TLS cache (from linux_cpp/Pool)
- **CUDA Kernel Library**: Optimized GPU operators (from vLLM)
  - LayerNorm / RMSNorm
  - Softmax / LogSoftmax
  - Attention / FlashAttention
  - Elementwise operations
  - Reduction operations
- **Thread Scheduling**: Production-grade thread pool with task graph execution
- **IPC Communication**: High-speed inter-process communication with shared memory (from linux_cpp/ipc)
- **Benchmark Framework**: Built-in performance testing infrastructure

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Applications                               │
│              (LLM / RL Training / Scientific Computing)              │
├─────────────────────────────────────────────────────────────────────┤
│                      High Level Compute API                          │
├─────────────────────────────────────────────────────────────────────┤
│                     AI Scientific Computing Layer                     │
│     Tensor Operations | Linear Algebra | Attention | Normalization   │
├─────────────────────────────────────────────────────────────────────┤
│                        Runtime Layer                                 │
│   Scheduler | Task Graph | Async Runtime | Pipeline Engine           │
├─────────────────────────────────────────────────────────────────────┤
│                        Device Layer                                  │
│              CUDA Backend | CPU Backend | Multi-GPU                 │
├─────────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                             │
│         IPC | Thread Pool | Memory Pool | Profiler                  │
├─────────────────────────────────────────────────────────────────────┤
│                          Hardware                                   │
│                     CUDA GPU | CPU | NUMA                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
SciComputeInfra/
├── CMakeLists.txt              # Main build configuration
│
├── include/
│   ├── core/                   # Core types and utilities
│   │   ├── types.hpp          # DType, Layout, TensorShape, DeviceType
│   │   ├── status.hpp         # Result/Expected types
│   │   ├── macros.hpp         # Compile-time macros
│   │   ├── config.hpp         # Configuration management
│   │   ├── timer.hpp          # Timer utilities
│   │   └── version.hpp        # Version information
│   │
│   ├── tensor/                # Tensor core
│   │   ├── tensor.hpp         # Tensor, TensorView, TensorOptions
│   │   └── tensor_shape.hpp   # Shape utilities
│   │
│   ├── memory/                 # Memory management
│   │   ├── allocator.hpp       # Allocator interface
│   │   ├── memory_pool.hpp    # ⭐ From linux_cpp/Pool
│   │   │                        #   - LockFreeStack
│   │   │                        #   - FixedSizeMemoryPool
│   │   │                        #   - SlabAllocator
│   │   │                        #   - TlsCache
│   │   └── buffer_handle.hpp   # RAII buffer wrapper
│   │
│   ├── device/                 # Device abstraction
│   │   ├── device.hpp         # Device base class, CPUDevice
│   │   └── stream.hpp          # CUDA Stream abstraction
│   │
│   ├── scheduler/              # Task scheduling
│   │   ├── thread_pool.hpp    # ⭐ From linux_cpp/Pool
│   │   ├── task.hpp           # Task base class
│   │   └── task_graph.hpp     # DAG task graph
│   │
│   ├── math/                   # Mathematical operations
│   │   ├── elementwise.hpp     # add, mul, sub, div, etc.
│   │   ├── reduction.hpp       # sum, mean, max, min, etc.
│   │   ├── softmax.hpp         # Softmax variants
│   │   └── normalization.hpp   # LayerNorm, RMSNorm
│   │
│   ├── benchmark/              # Benchmarking
│   │   ├── benchmark_case.hpp  # Benchmark case
│   │   └── benchmark_runner.hpp # Runner
│   │
│   ├── communication/          # Communication layer
│   │   └── ipc.hpp            # ⭐ From linux_cpp/ipc
│   │                              #   - SharedMemorySegment
│   │                              #   - MPSCQueue
│   │                              #   - SPSCQueue
│   │                              #   - SpinLock
│   │
│   └── kernel/                 # Kernel registry
│       └── kernel_registry.hpp
│
├── src/                        # Implementation files
│   ├── core/
│   ├── tensor/
│   ├── memory/
│   ├── device/
│   ├── scheduler/
│   ├── math/
│   ├── benchmark/
│   └── communication/
│
├── cuda/
│   └── kernels/                # ⭐ From RL_infra/vllm
│       ├── layernorm.cu/.cuh   # LayerNorm/RMSNorm Kernel
│       ├── softmax.cu/.cuh     # Softmax Kernel
│       ├── attention.cu/.cuh   # Attention Kernel
│       ├── elementwise.cu/.cuh # Elementwise Kernel
│       └── reduction.cu/.cuh   # Reduction Kernel
│
├── tests/
│   └── unit/                   # Unit tests
│       ├── test_tensor.cpp
│       ├── test_math.cpp
│       ├── test_memory.cpp
│       └── test_scheduler.cpp
│
├── benchmarks/                  # Benchmarks
│   ├── bench_elementwise.cpp
│   ├── bench_reduction.cpp
│   └── bench_softmax.cpp
│
├── examples/                    # Example programs
│   ├── simple_tensor.cpp        # Basic tensor demo
│   └── cuda_kernel_demo.cu     # CUDA kernel demo
│
├── scripts/                     # Scripts
│   ├── build.sh
│   └── run_benchmarks.sh
│
├── TASK_PLAN.md                 # Project task plan
├── DESIGN.md                   # Design document
└── README.md                   # This file
```

### Quick Start

```bash
# Basic build (CPU only)
cd SciComputeInfra
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=OFF
make -j4

# Run example
./examples/simple_tensor
```

### Build Options

| Option | Description | Default |
|-------|-------------|---------|
| `SCI_ENABLE_CUDA` | Enable CUDA support | OFF |
| `SCI_ENABLE_OPENMP` | Enable OpenMP parallelism | ON |
| `SCI_BUILD_TESTS` | Build unit tests | OFF |
| `SCI_BUILD_BENCHMARKS` | Build benchmarks | OFF |

### Dependencies

- C++20 compiler (GCC 12+, Clang 15+)
- CMake 3.20+
- Threads (pthreads)
- OpenMP 4.5+ (optional)
- CUDA Toolkit 11.0+ (optional, for GPU support)
- GTest (optional, for testing)
- Google Benchmark (optional, for benchmarking)

### Detailed Legacy Project Integration

#### 1. linux_cpp/Pool → memory_pool.hpp

Core components from `linux_cpp/Pool/Memory_Pool.cpp`:

```cpp
// Lock-free stack for high-performance memory block management
class LockFreeStack {
    void Push(void* ptr) noexcept;
    void* Pop() noexcept;
};

// Fixed-size memory pool
class FixedSizeMemoryPool {
    void* Allocate();
    void Deallocate(void* ptr) noexcept;
};

// Multi-size slab allocator
class SlabAllocator {
    void* Allocate(size_t size) noexcept;
    void Deallocate(void* ptr, size_t size) noexcept;
};

// TLS cache to reduce cross-core CAS contention
class TlsCache {
    void* Get();
    void Put(void* ptr);
};
```

#### 2. linux_cpp/Pool → thread_pool.hpp

From `linux_cpp/Pool/Thread_Pool.cpp`:

```cpp
class ThreadPool {
    // Configuration:
    // - min_threads / max_threads
    // - idle_timeout_ms
    
    // Core features:
    // - Enqueue() returns std::future
    // - WaitAll() for batch waiting
    // - Pause/Resume control
    // - Resize for dynamic adjustment
};
```

#### 3. linux_cpp/ipc → communication/ipc.hpp

From `linux_cpp/ipc/include/ipc/*`:

```cpp
// Shared memory segment
class SharedMemorySegment {
    bool create(const std::string& name, size_t size);
    bool open(const std::string& name);
    void close();
};

// MPSC queue (single producer, multiple consumers)
template<typename T>
class MPSCQueue {};

// SPSC queue (single producer, single consumer)
template<typename T>
class SPSCQueue {};

// Spin lock
class SpinLock {};
```

#### 4. RL_infra/vllm → cuda/kernels/

From `RL_infra/vllm/src/kernels/*.cu`:

```cpp
// LayerNorm - fused kernel with warp-level reduction
launch_layer_norm(out, in, weight, bias, rows, cols, eps, stream);
launch_rms_norm(out, in, weight, rows, cols, eps, stream);

// Softmax - numerically stable implementation
launch_softmax(out, x, n, dim, stream);

// Attention - FlashAttention style
launch_attention(out, q, k, v, ...);

// Elementwise - fused add/mul/sub/div
launch_add(out, a, b, n, stream);
```

### Version History

- **0.1.0** (2026-08-17): Initial version, core framework complete

---

**Project Location**: `本仓库`
