# SciComputeInfra - AI Scientific Computing Infrastructure

高性能计算运行时 + CUDA Kernel库 + Tensor运行时 + 调度系统 + 基准测试体系

## 项目背景

本项目是从已有的 AI/RL/HPC 项目统一升级而来，整合了以下技术资产：

| 源项目 | 贡献模块 |
|--------|---------|
| linux_cpp/Pool | Memory Pool (无锁Slab分配器)、Thread Pool |
| linux_cpp/ipc | IPC通信库 (共享内存、队列、同步) |
| RL_infra/vllm | CUDA Kernels (LayerNorm, Attention, Softmax等) |
| RL_infra/Turbol | 任务调度和异步执行设计 |

## 核心特性

- **高性能张量计算**: 支持 FP32/FP16/BF16 等数据类型
- **CUDA Kernel库**: 优化的 GPU 算子 (LayerNorm, Attention, Softmax等)
- **统一内存管理**: 内存池、Slab分配器、TLS缓存
- **线程调度**: 生产级线程池，支持任务图执行
- **IPC通信**: 进程间高速通信，支持共享内存
- **基准测试**: 内置性能测试框架

## 快速开始

```bash
# 克隆并构建
cd SciComputeInfra
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=OFF
make -j4

# 运行示例
./examples/simple_tensor
```

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    Applications                      │
├─────────────────────────────────────────────────────┤
│              High Level Compute API                  │
├─────────────────────────────────────────────────────┤
│         AI Scientific Computing Layer                │
│  (Tensor Operations, Linear Algebra, Attention)     │
├─────────────────────────────────────────────────────┤
│                   Runtime Layer                      │
│  (Scheduler, Task Graph, Async Runtime, Pipeline)   │
├─────────────────────────────────────────────────────┤
│                  Device Layer                        │
│      (CUDA Backend, CPU Backend, Multi-GPU)         │
├─────────────────────────────────────────────────────┤
│              Infrastructure Layer                    │
│   (IPC, Thread Pool, Memory Pool, Profiler)         │
└─────────────────────────────────────────────────────┘
```

## 模块说明

### Core (`include/core/`)
- `types.hpp`: 类型定义 (DType, Layout, TensorShape, DeviceType)
- `status.hpp`: Result/Expected类型用于错误处理
- `macros.hpp`: 编译期宏
- `config.hpp`: 配置管理

### Tensor (`include/tensor/`)
- `tensor.hpp`: 核心Tensor类，支持视图、reshape、transpose等操作
- `tensor_shape.hpp`: Shape工具函数

### Memory (`include/memory/`)
- `allocator.hpp`: 内存分配器接口
- `memory_pool.hpp`: 来自linux_cpp/Pool的高性能内存池
- `buffer_handle.hpp`: RAII缓冲区封装

### Device (`include/device/`)
- `device.hpp`: 设备抽象基类和CPUDevice实现
- `stream.hpp`: CUDA Stream抽象

### Scheduler (`include/scheduler/`)
- `thread_pool.hpp`: 来自linux_cpp/Pool的生产级线程池
- `task.hpp`: 任务抽象
- `task_graph.hpp`: DAG任务图

### Math (`include/math/`)
- `elementwise.hpp`: 元素级运算 (add, mul, sub, div等)
- `reduction.hpp`: 归约运算 (sum, mean, max, min等)
- `softmax.hpp`: Softmax及其变体
- `normalization.hpp`: LayerNorm, RMSNorm, BatchNorm

### Communication (`include/communication/`)
- `ipc.hpp`: IPC通信接口，来自linux_cpp/ipc

### CUDA Kernels (`cuda/kernels/`)
- 来自vLLM的优化CUDA kernel实现
- LayerNorm, Softmax, Attention, Elementwise等

## 构建选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `SCI_ENABLE_CUDA` | 启用CUDA支持 | OFF |
| `SCI_ENABLE_OPENMP` | 启用OpenMP并行 | ON |
| `SCI_BUILD_TESTS` | 构建单元测试 | OFF |
| `SCI_BUILD_BENCHMARKS` | 构建基准测试 | OFF |

## 依赖

- C++20 编译器 (GCC 12+, Clang 15+)
- CMake 3.20+
- Threads (pthreads)
- OpenMP 4.5+ (可选)
- CUDA Toolkit 11.0+ (可选)
- GTest (可选，用于测试)
- Google Benchmark (可选，用于基准测试)

## 许可证

本项目采用 MIT 许可证。

## 版本

当前版本: **0.1.0**

---

项目地址: `本仓库`
