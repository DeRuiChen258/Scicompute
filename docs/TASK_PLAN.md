# SciComputeInfra 项目任务计划

## 项目概述

- **项目名称**: SciComputeInfra (AI Scientific Computing Infrastructure)
- **目标**: 高性能计算运行时 + CUDA Kernel库 + Tensor运行时 + 调度系统 + 基准测试体系
- **工作区**: 本仓库（SciComputeInfra）
- **版本**: 0.1.0

---

## 已有项目映射关系

| 旧项目路径                                   | 旧项目功能                               | 新项目映射                     | 复用方式    |
| --------------------------------------- | ----------------------------------- | ------------------------- | ------- |
| linux_cpp/Pool/Memory_Pool.cpp          | 无锁内存池、Slab分配器、TLS缓存                 | memory/memory_pool.hpp    | ✅ 已集成   |
| linux_cpp/Pool/Thread_Pool.cpp          | 生产级线程池                              | scheduler/thread_pool.hpp | ✅ 已集成   |
| linux_cpp/ipc/include/ipc/*             | IPC通信库（队列、共享内存、同步）                  | communication/ipc.hpp     | ✅ 已集成   |
| RL_infra/Turbol/turbol/include/turbol/* | 分布式RL运行时                            | scheduler/async_engine    | 🔄 设计参考 |
| RL_infra/vllm/src/kernels/*.cu          | CUDA kernel (layernorm, attention等) | cuda/kernels/*.cu         | ✅ 已集成   |

---

## ✅ 阶段一：核心框架 MVP (已完成)

### 1.1 Core模块 ✅

- [x] include/core/types.hpp - 类型定义 (DType, Layout, TensorShape, DeviceType)
- [x] include/core/status.hpp - Status/Result类型
- [x] include/core/macros.hpp - 编译期宏
- [x] include/core/config.hpp - 配置管理
- [x] include/core/timer.hpp - 计时器
- [x] include/core/version.hpp - 版本信息
- [x] include/core/platform.hpp - 平台抽象

### 1.2 Device模块 ✅

- [x] include/device/device.hpp - 设备抽象基类 + CPUDevice
- [x] include/device/stream.hpp - Stream/Event抽象
- [x] include/device/device_info.hpp - 设备信息查询
- [x] src/device/device.cpp - 设备实现
- [x] src/device/stream.cpp - Stream实现

### 1.3 Memory模块 ✅

- [x] include/memory/allocator.hpp - 分配器接口
- [x] include/memory/memory_pool.hpp - 内存池（来自linux_cpp/Pool）
- [x] include/memory/buffer_handle.hpp - RAII缓冲区封装
- [x] src/memory/allocator.cpp - Host/Pinned分配器实现
- [x] src/memory/buffer_handle.cpp - BufferHandle实现
- [x] src/memory/memory_pool.cpp - 内存池实现

### 1.4 Tensor模块 ✅

- [x] include/tensor/tensor.hpp - Tensor/TensorView/TensorOptions
- [x] include/tensor/tensor_shape.hpp - 扩展shape工具
- [x] src/tensor/tensor.cpp - Tensor实现

### 1.5 Scheduler模块 ✅

- [x] include/scheduler/thread_pool.hpp - 线程池（来自linux_cpp/Pool）
- [x] include/scheduler/task.hpp - Task基类
- [x] include/scheduler/task_graph.hpp - DAG任务图
- [x] src/scheduler/thread_pool.cpp - 线程池实现
- [x] src/scheduler/task.cpp - Task实现
- [x] src/scheduler/task_graph.cpp - 任务图实现

### 1.6 Math模块 ✅

- [x] include/math/elementwise.hpp - 元素级运算声明
- [x] src/math/elementwise.cpp - 元素级运算实现
- [x] include/math/reduction.hpp - 归约运算声明
- [x] src/math/reduction.cpp - 归约运算实现
- [x] include/math/softmax.hpp - Softmax声明
- [x] src/math/softmax.cpp - Softmax实现
- [x] include/math/normalization.hpp - 归一化声明
- [x] src/math/normalization.cpp - 归一化实现

### 1.7 Benchmark模块 ✅

- [x] include/benchmark/benchmark_case.hpp - Benchmark用例
- [x] include/benchmark/benchmark_runner.hpp - Benchmark运行器
- [x] src/benchmark/benchmark_runner.cpp - 运行器实现

### 1.8 Communication模块 ✅ (来自linux_cpp/ipc)

- [x] include/communication/ipc.hpp - IPC通信接口
- [x] src/communication/ipc.cpp - SharedMemory实现

### 1.9 构建系统 ✅

- [x] 完善 src/CMakeLists.txt
- [x] 创建主库 sci_library
- [x] 配置依赖 (Threads, OpenMP)
- [x] 添加测试/基准测试开关

---

## ✅ 阶段二：CUDA支持 (已完成)

### 2.1 CUDA Kernel头文件 ✅

- [x] cuda/kernels/elementwise.cuh - 元素级CUDA kernel
- [x] cuda/kernels/reduction.cuh - 归约CUDA kernel
- [x] cuda/kernels/softmax.cuh - Softmax CUDA kernel
- [x] cuda/kernels/layernorm.cuh - LayerNorm CUDA kernel (来自vLLM)
- [x] cuda/kernels/attention.cuh - Attention CUDA kernel

### 2.2 CUDA Kernel实现 ✅

- [x] cuda/kernels/elementwise.cu - 元素级kernel实现
- [x] cuda/kernels/reduction.cu - 归约kernel实现
- [x] cuda/kernels/softmax.cu - Softmax kernel实现
- [x] cuda/kernels/layernorm.cu - LayerNorm kernel实现

### 2.3 CUDA Backend ✅

- [x] include/device/cuda_device.hpp + src/device/cuda_device.cpp - CUDA设备实现 (Device接口, 无GPU环境优雅降级)
- [x] include/device/cuda_allocator.hpp - CUDA设备内存分配器 (sci::Allocator 接口)
- [x] DeviceManager::ScanDevices 自动注册CUDA设备
- [x] Stream 异步拷贝/填充接入 CUDA 流 (copy_async / memset_async)

---

## ✅ 阶段三：测试系统 (已完成)

### 3.1 单元测试 ✅

- [x] tests/unit/test_tensor.cpp - Tensor单元测试
- [x] tests/unit/test_memory.cpp - Memory单元测试
- [x] tests/unit/test_math.cpp - Math单元测试
- [x] tests/unit/test_scheduler.cpp - Scheduler单元测试

### 3.2 集成测试 ✅

- [x] tests/integration/test_scheduler.cpp - 调度器集成测试 (线程池+任务图+Tensor流水线, 依赖顺序, 执行器复用)
- [x] tests/integration/test_tensor_ops.cpp - Tensor运算集成测试 (元素级链式运算, Softmax/LayerNorm数值性质, 归约, View/拷贝, 跨设备迁移)
- [x] 修复 GraphExecutor 重复入队竞态: 任务入队后、执行前可能被多次调度 (TSan 验证通过)
- [x] 修复 math::var / math::norm 空形状输出导致的空指针解引用

---

## ✅ 阶段四：Benchmark系统 (已完成)

### 4.1 Micro Benchmarks ✅

- [x] benchmarks/micro/bench_elementwise.cpp - 元素级运算benchmark
- [x] benchmarks/micro/bench_reduction.cpp - 归约benchmark
- [x] benchmarks/micro/bench_softmax.cpp - Softmax benchmark
- [x] benchmarks/micro/bench_layernorm.cpp - LayerNorm benchmark

### 4.2 Integration Benchmarks ✅

- [x] benchmarks/integration/bench_transformer.cpp - Transformer层核心流水线benchmark (LN→FFN→Residual→Softmax→Reduction)
- [x] benchmarks/integration/bench_attention.cpp - Attention benchmark (QK^T→Softmax→PV, 参考GEMM)
- [x] benchmarks/integration/ref_gemm.hpp - 共享参考GEMM/Softmax实现

---

## 项目目录结构

```
SciComputeInfra/
├── CMakeLists.txt
├── include/
│   ├── core/           ✅ 类型、状态、宏、配置、计时器
│   ├── tensor/         ✅ Tensor/TensorView
│   ├── memory/         ✅ 分配器、内存池、BufferHandle
│   ├── device/         ✅ Device抽象、Stream、CPUDevice
│   ├── scheduler/      ✅ 线程池、Task、TaskGraph
│   ├── math/           ✅ 元素级、归约、Softmax、归一化
│   ├── benchmark/      ✅ Benchmark框架
│   ├── communication/  ✅ IPC通信 (来自linux_cpp/ipc)
│   └── utils/          ✅ 日志等工具
├── src/
│   ├── core/
│   ├── tensor/
│   ├── memory/
│   ├── device/
│   ├── scheduler/
│   ├── math/
│   ├── benchmark/
│   └── communication/
├── cuda/
│   └── kernels/        ✅ CUDA Kernel (来自vLLM)
├── tests/
│   └── unit/           ✅ 单元测试
├── benchmarks/
├── examples/           ✅ simple_tensor等示例
├── docs/               ✅ DESIGN.md（架构设计）、TASK_PLAN.md（本文件）
├── LICENSE             ✅ MIT
└── README.md
```

---

## 构建说明

```bash
# 基本构建 (仅CPU)
cd SciComputeInfra
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=OFF
make -j4

# 启用CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=ON

# 启用测试
cmake .. -DCMAKE_BUILD_TYPE=Release -DSCI_BUILD_TESTS=ON

# 运行示例
./examples/simple_tensor
```

---

## 技术栈

- **语言**: C++20, CUDA C++
- **构建**: CMake 3.20+
- **依赖**: Threads, OpenMP, (可选: CUDA, GTest, Google Benchmark)
- **命名空间**: sci::

---

## 版本历史

- **0.1.0** (2026-08-17): 初始版本，核心框架完成
- **0.1.1** (2026-08-18): CUDA设备后端 + 集成测试 + Benchmark体系, 修复调度器竞态与归约空指针

---

## ✅ 验证记录 (2026-08-18，开发机无 GPU 驱动)

- **CUDA ON 构建** (Release, tests + benchmarks): 通过, `ctest` 55/55 通过
- **CPU-only 构建** (Release, tests + benchmarks): 通过, `ctest` 55/55 通过
- **ThreadSanitizer**: `test_integration_scheduler` 4/4 通过, 无竞态报告
- **Benchmarks**: 6 个 benchmark (elementwise / reduction / softmax / layernorm / transformer / attention) 全部运行验证通过
- **运行环境说明**: 当前机器无 NVIDIA 驱动, CUDA 后端编译可用但运行时自动降级 (`CudaDevice::available()` 返回 false, 跨设备迁移返回明确错误)

## ✅ 验证记录 (2026-09-17，RTX 5070 Laptop + CUDA 13.2)

- **CUDA ON 构建** (Release, tests + benchmarks): 通过, `ctest` **56/56** 通过
- **CPU-only 构建** (Release, tests): 通过, `ctest` **56/56** 通过
- **基准**: `benchmark_test` + 6 个 Google Benchmark 全部运行通过（数据见 README §6）
- **本轮修复**（在有 GPU 的机器上暴露）:
  1. `DeviceManager::get_device()` 首次查询非 CPU 设备时自动 `ScanDevices()`，
     否则 `Tensor::to(kCUDA,0)` 在 GPU 机器上返回 "Invalid device type"
  2. `Tensor::to()`/`copy_from()` 增加方向感知的跨设备拷贝（Host→Device 用目标设备、
     Device→Host 用源设备、设备间经主机暂存），修复 CUDA→CPU 回拷时用主机 memcpy 读显存导致的段错误
  3. 新增 `Device::copy_within()`（CUDA 后端为 `cudaMemcpyDeviceToDevice`），
     修复同设备克隆在显存上仍走主机 memcpy 的问题
  4. `CrossDeviceTransfer` 集成测试改为「拷回主机后校验」，并增加显存内克隆断言
  5. CMake：CUDA 路径不再硬编码 `/usr/local/cuda-13.2`，改用 `FindCUDAToolkit`；
     架构列表改为可覆盖的 `SCI_CUDA_ARCHITECTURES`
