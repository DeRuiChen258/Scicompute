# SciComputeInfra - AI Scientific Computing Infrastructure

## 1. Project Overview

**Project Name:** SciComputeInfra  
**Type:** High-Performance Computing Runtime + CUDA Kernel Library + Tensor Runtime  
**Core Purpose:** Build a sustainable, extensible foundation for AI/RL/Numerical Computing workloads

### 1.1 Key Objectives

- High-performance tensor computation and numerical computing
- Unified execution framework: CPU/CUDA/SIMD/Multi-threaded/Async scheduling
- Kernel library with reference + optimized implementations
- Comprehensive benchmark and testing infrastructure
- Extensible architecture for future AI Infra, RL, scientific computing extensions

### 1.2 Technical Stack

- **C++20**: Primary implementation language
- **CUDA C++**: GPU kernels and CUDA runtime integration
- **CMake**: Build system
- **Python**: Benchmark harness, testing scripts, bindings
- **GoogleTest/Google Benchmark**: Testing and profiling

### 1.3 Technology Assets Reuse

| Old Project     | New System Component                    |
| --------------- | --------------------------------------- |
| linux_cpp/Pool  | Memory Pool → Tensor Allocator          |
| linux_cpp/ipc   | IPC → Communication Layer               |
| RL_infra/Turbol | Scheduler, Task Graph, Async Runtime    |
| RL_infra/vllm   | CUDA Kernels, Attention, Memory Manager |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│               High-Level Ops / Graph Layer                  │
├─────────────────────────────────────────────────────────────┤
│                   Tensor Runtime Layer                      │
├─────────────────────────────────────────────────────────────┤
│              Scheduler / Memory / Device Layer              │
├─────────────────────────────────────────────────────────────┤
│                     Backend Layer                           │
│     ┌─────────────────┐  ┌─────────────────────────────┐   │
│     │   CPU Backend   │  │       CUDA Backend          │   │
│     │  (SIMD/OpenMP)  │  │   (cuBLAS/cuRAND/NVTX)      │   │
│     └─────────────────┘  └─────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Hardware                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Module Responsibilities

### 3.1 Core Module (include/core/, src/core/)

- Type definitions, error codes, status types
- Logger, configuration, timing utilities
- Foundation for all other modules

### 3.2 Tensor Module (include/tensor/, src/tensor/)

- Tensor: Main tensor object with shape, dtype, device, layout
- TensorShape: Shape/size management
- TensorView: Non-owning tensor reference
- TensorOptions: Construction options (dtype, device, layout)
- TensorDescriptor: Memory layout description

### 3.3 Memory Module (include/memory/, src/memory/)

- HostAllocator: CPU memory allocation
- DeviceAllocator: GPU memory allocation
- PinnedAllocator: Pinned memory for DMA
- MemoryPool: Pool-based allocator (TLS cache, slab)
- BufferHandle: RAII wrapper for allocated buffers

### 3.4 Device Module (include/device/, src/device/)

- Device: Abstract device (CPU/CUDA)
- DeviceType: Enum for device types
- DeviceContext: Device state management
- Stream: Execution stream abstraction
- Event: Synchronization events
- DeviceInfo: Device capability queries

### 3.5 Scheduler Module (include/scheduler/, src/scheduler/)

- Task: Base task representation
- TaskGraph: DAG of tasks with dependencies
- Scheduler: Task scheduling and execution
- ThreadPool: CPU thread pool (from linux_cpp/Pool)
- Worker: Worker thread abstraction

### 3.6 Kernel Module (include/kernel/, src/kernel/, cuda/)

- KernelRegistry: Kernel registration and dispatch
- KernelLauncher: Unified kernel launch interface
- KernelContext: Execution context for kernels
- CUDA kernels: elementwise, reduction, softmax, layernorm, attention

### 3.7 Math Module (include/math/, src/math/)

- Elementwise: add, mul, sub, div
- Reduction: sum, max, min, mean
- Softmax: standard, log-softmax
- Normalization: layer norm, RMS norm
- Sampling: topk, argmax, sampling
- Distance: L2, cosine similarity

### 3.8 Graph Module (include/graph/, src/graph/)

- Graph: Computation graph representation
- Node: Graph node with op and inputs
- GraphExecutor: Graph execution engine
- GraphPass: Optimization passes (fusion)

### 3.9 Benchmark Module (include/benchmark/, src/benchmark/)

- BenchmarkRunner: Main benchmark orchestrator
- BenchmarkCase: Individual benchmark case
- BenchmarkReport: Result aggregation and export
- BenchmarkProfiler: Profiling integration

### 3.10 Utils Module (include/utils/, src/utils/)

- Logging (spdlog integration)
- Configuration (nlohmann/json)
- Timing utilities
- String/file utilities

---

## 4. Directory Structure

```
SciComputeInfra/
├── CMakeLists.txt
├── cmake/
│   ├── FindCUDA.cmake
│   ├── CompilerOptions.cmake
│   └── GenerateExportHeader.cmake
├── include/
│   ├── core/
│   │   ├── status.hpp
│   │   ├── error.hpp
│   │   ├── types.hpp
│   │   ├── config.hpp
│   │   └── macros.hpp
│   ├── tensor/
│   │   ├── tensor.hpp
│   │   ├── tensor_shape.hpp
│   │   ├── tensor_view.hpp
│   │   ├── tensor_options.hpp
│   │   └── tensor_descriptor.hpp
│   ├── memory/
│   │   ├── allocator.hpp
│   │   ├── memory_pool.hpp
│   │   ├── buffer_handle.hpp
│   │   └── device_allocator.hpp
│   ├── device/
│   │   ├── device.hpp
│   │   ├── device_type.hpp
│   │   ├── stream.hpp
│   │   ├── event.hpp
│   │   └── device_info.hpp
│   ├── scheduler/
│   │   ├── task.hpp
│   │   ├── task_graph.hpp
│   │   ├── scheduler.hpp
│   │   └── thread_pool.hpp
│   ├── kernel/
│   │   ├── kernel_registry.hpp
│   │   ├── kernel_launcher.hpp
│   │   └── kernel_traits.hpp
│   ├── math/
│   │   ├── elementwise.hpp
│   │   ├── reduction.hpp
│   │   ├── softmax.hpp
│   │   ├── normalization.hpp
│   │   └── sampling.hpp
│   ├── graph/
│   │   ├── graph.hpp
│   │   ├── node.hpp
│   │   └── executor.hpp
│   ├── benchmark/
│   │   ├── benchmark_runner.hpp
│   │   ├── benchmark_case.hpp
│   │   └── benchmark_report.hpp
│   └── utils/
│       ├── logger.hpp
│       ├── config.hpp
│       ├── timer.hpp
│       └── string_utils.hpp
├── src/
│   ├── core/
│   ├── tensor/
│   ├── memory/
│   ├── device/
│   ├── scheduler/
│   ├── kernel/
│   ├── math/
│   ├── graph/
│   ├── benchmark/
│   └── utils/
├── cuda/
│   ├── kernels/
│   │   ├── elementwise.cuh
│   │   ├── reduction.cuh
│   │   ├── softmax.cuh
│   │   ├── layernorm.cuh
│   │   └── attention.cuh
│   ├── launch/
│   │   └── kernel_launcher.cuh
│   └── helpers/
│       └── math_utils.cuh
├── tests/
│   ├── unit/
│   │   ├── test_tensor.cpp
│   │   ├── test_memory.cpp
│   │   ├── test_math.cpp
│   │   └── test_device.cpp
│   ├── integration/
│   │   └── test_scheduler.cpp
│   └── benchmark/
│       └── bench_tensor_ops.cpp
├── python/
│   ├── bindings/
│   │   └── pybind_tensor.cpp
│   ├── benchmarks/
│   │   └── benchmark_runner.py
│   └── examples/
│       └── simple_ops.py
├── benchmarks/
│   ├── micro/
│   │   ├── bench_elementwise.cpp
│   │   ├── bench_reduction.cpp
│   │   └── bench_softmax.cpp
│   └── macro/
│       └── bench_pipeline.cpp
├── examples/
│   ├── simple_tensor.cpp
│   └── cuda_kernel_demo.cu
├── docs/
│   ├── architecture.md
│   ├── api_reference.md
│   └── kernel_design.md
├── scripts/
│   ├── build.sh
│   └── run_benchmarks.sh
└── README.md
```

---

## 5. Core Data Structures

### 5.1 Status / Result Type

```cpp
enum class StatusCode { OK, ERROR, UNIMPLEMENTED, ... };
struct Status { StatusCode code; std::string message; };
template<typename T> using Result = Expected<T, Status>;
```

### 5.2 Tensor

```cpp
struct Tensor {
    TensorShape shape;
    DType dtype;
    Device device;
    Layout layout;
    BufferHandle buffer;
    // methods: reshape, view, clone, to_device
};
```

### 5.3 Kernel Registry Pattern

```cpp
template<typename Output, typename... Inputs>
class KernelRegistry {
public:
    using KernelFn = std::function<Result<Output>(Inputs...)>;
    void Register(const std::string& name, KernelFn fn);
    Result<Output> Execute(const std::string& name, Inputs... inputs);
};
```

---

## 6. CUDA Kernel Design

### 6.1 Required Kernel Versions

Each operator must provide:

1. Reference: Clear, readable implementation
2. Optimized: Regular optimization for common cases
3. Specialized: FP16/BF16/INT8 paths, small/large size variants

### 6.2 Target Operators (Priority Order)

1. add / mul / sub / div (elementwise)
2. reduction sum / max / mean
3. softmax / log-softmax
4. layer norm / RMS norm
5. topk / argmax / sampling
6. matmul (simplified)
7. attention primitive
8. rotary embedding

### 6.3 Performance Standards

- Memory coalescing
- Warp-level primitives
- Shared memory optimization
- Occupancy tuning
- NVTX annotations

---

## 7. Benchmark Strategy

### 7.1 Benchmark Types

- Micro: Single operator/kernel
- Macro: Module/pipeline level
- Comparative: vs baseline/reference
- Stress: Extreme inputs
- Regression: Version comparisons

### 7.2 Metrics

- Latency (mean, p50, p95, p99)
- Throughput (ops/sec, GB/sec)
- Memory usage
- GPU occupancy
- Bandwidth utilization

### 7.3 Output Formats

- JSON (machine-readable)
- CSV (spreadsheet analysis)
- Markdown (human-readable)

---

## 8. Testing Strategy

### 8.1 Unit Tests

- Normal path coverage
- Boundary values
- Error paths
- Random inputs
- Dtype coverage

### 8.2 Integration Tests

- Tensor + Memory + Device combinations
- Scheduler execution
- CPU/CUDA consistency
- Async execution correctness

### 8.3 Performance Tests

- Baseline establishment
- Regression detection
- Optimization validation

---

## 9. Phase 1: Minimum Viable Product (MVP)

### Components to Implement

1. Core module (status, error, types, macros)
2. Device module (CPU device implementation)
3. Memory module (host allocator, buffer handle)
4. Tensor module (basic tensor structure)
5. Math module: add, mul operations (CPU)
6. Simple benchmark infrastructure
7. Unit test framework

### First 10 Files to Create

1. include/core/types.hpp - Base type definitions
2. include/core/status.hpp - Status/Result types
3. include/device/device.hpp - Device abstraction
4. include/memory/allocator.hpp - Memory allocation
5. include/tensor/tensor.hpp - Tensor class
6. src/core/*.cpp - Core implementation
7. src/device/*.cpp - Device implementation
8. src/memory/*.cpp - Memory implementation
9. src/tensor/*.cpp - Tensor implementation
10. tests/unit/test_tensor.cpp - Basic tensor test

---

## 10. Risk Analysis

### Technical Risks

| Risk                   | Mitigation                           |
| ---------------------- | ------------------------------------ |
| CUDA kernel complexity | Start with reference, optimize later |
| Memory pool contention | TLS cache, NUMA awareness            |
| API instability        | Version control, deprecation policy  |
| Performance regression | CI benchmarks, baseline tracking     |

### Technical Debt

- Immediate: API design review
- Short-term: Complete error handling
- Long-term: Documentation, examples

---

## 11. Long-term Roadmap

### Phase 2: Basic Operators (add, mul, reduction, softmax)

### Phase 3: CUDA Backend + Advanced Kernels

### Phase 4: Scheduler + Task Graph

### Phase 5: Python Bindings + Documentation

### Phase 6: Fused Kernels + Attention Primitives
