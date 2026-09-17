# SciComputeInfra

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![C++](https://img.shields.io/badge/C%2B%2B-20-00599C.svg)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-optional-76B900.svg)](cuda/kernels)
[![CMake](https://img.shields.io/badge/cmake-3.20%2B-064F8C.svg)](CMakeLists.txt)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)](#7-测试)

**面向 RL 与科学计算的高性能 C++ 基础设施库：无锁 IPC + 内存池 + 设备抽象 + 张量运行时 + 任务调度 + 数学算子 + CUDA Kernel + 基准体系。**

SciComputeInfra 把一套「训练/推理系统」反复要用到的底层能力做成可组合的独立模块：需要单机低延迟数据面时用无锁队列与内存池，
需要张量与算子时用统一的 `Tensor` + `math::*`，需要并行时用线程池与 DAG 任务图，需要 GPU 时打开 `SCI_ENABLE_CUDA`
并复用同一套接口——CUDA 不可用时自动降级而不是崩溃。

---

## 目录

- [1. 特性一览](#1-特性一览)
- [2. 快速开始](#2-快速开始)
  - [2.1 环境要求](#21-环境要求)
  - [2.2 CPU-only 构建](#22-cpu-only-构建)
  - [2.3 CUDA 构建](#23-cuda-构建)
  - [2.4 测试](#24-测试)
  - [2.5 基准测试](#25-基准测试)
  - [2.6 示例程序](#26-示例程序)
- [3. 架构](#3-架构)
- [4. 核心组件与 API](#4-核心组件与-api)
- [5. 构建选项](#5-构建选项)
- [6. 性能实测](#6-性能实测)
- [7. 测试](#7-测试)
- [8. 目录结构](#8-目录结构)
- [9. 项目状态与路线图](#9-项目状态与路线图)
- [10. 已知限制](#10-已知限制)
- [11. 贡献指南](#11-贡献指南)
- [12. 许可证](#12-许可证)

---

## 1. 特性一览

| 模块 | 能力 | 关键指标（本机实测，见 [§6](#6-性能实测)） |
| --- | --- | --- |
| **无锁队列** | SPSC 环形缓冲、MPSC（带 ABA 防护）、批量 Push/Pop、按需扩容 | SPSC ~3.0×10⁸ ops/s（3.3 ns/op）、MPSC ~1.5×10⁸ ops/s（6.6 ns/op） |
| **内存管理** | 固定大小内存池、Slab 分配器、TLS 缓存、RAII `BufferHandle`、Host/Pinned/CUDA 分配器 | 内存池 ~4.6×10⁷ ops/s（22 ns/op），对比 `new/delete` ~4.3×10⁷ ops/s |
| **Tensor 运行时** | `Tensor`/`TensorView`、Shape/DType/Device、`Empty/Zeros/Ones/Rand`、跨设备迁移、`Result<T>` 错误处理 | 1024 元素逐元素加 0.114 µs（100 GiB/s 等效带宽） |
| **设备抽象** | 统一 `Device` 接口、`DeviceManager` 自动探测与热插拔回调、CUDA 不可用时优雅降级 | CUDA:0 可用时 `Tensor::to(kCUDA,0)` 直接可用（无需手动 ScanDevices） |
| **任务调度** | 线程池（动态扩缩、future 返回）、`TaskGraph` DAG + `GraphExecutor` | 集成测试覆盖依赖顺序、执行器复用与重复入队竞态 |
| **数学算子** | 元素级、归约（sum/mean/max/var/std/norm）、Softmax/LogSoftmax、LayerNorm，且提供 `math::ref::*` 参考实现 | Softmax 32×512：44.7 µs/批（3.67×10⁸ 元素/s）；LayerNorm 32×512：17.9 µs（6.8 GiB/s） |
| **CUDA Kernel** | elementwise / reduction / softmax / layernorm / attention（含 FlashAttention、RoPE、masked attention 模板） | 见 `cuda/kernels/*.cu[h]`，编译进 `sci_bridges` |
| **Benchmark 体系** | Google Benchmark 微基准 4 个 + 集成基准 2 个，另有自研吞吐基准 | 见 [§6](#6-性能实测) |

设计取向：

1. **CPU 路径永远是可用基线**：不开 CUDA、甚至没有 GPU 驱动也能完整构建、测试、跑基准。
2. **错误显式化**：可能失败的路径返回 `Result<T>`（`Status` 带具体错误码，如 `kCudaNotAvailable`、`kPoolDoubleFree`），不靠异常或静默失败。
3. **模块可独立使用**：`sci_memory`、`sci_tensor`、`sci_scheduler` 等静态库彼此解耦，`sci_compute` 只是汇总用的 INTERFACE 目标。
4. **性能主张必须有数字**：每个热路径都有可复现基准（命令与硬件写在 [§6](#6-性能实测)）。

## 2. 快速开始

### 2.1 环境要求

| 场景 | 要求 |
| --- | --- |
| 基础（CPU-only） | CMake ≥ 3.20、支持 C++20 的 GCC/Clang、pthread |
| 并行 | OpenMP（默认开启，`-DSCI_ENABLE_OPENMP=OFF` 可关） |
| GPU | CUDA Toolkit（本仓库路径无关，用 `FindCUDAToolkit` 定位，可用 `-DCUDAToolkit_ROOT=` 指定）+ 计算能力 ≥ 7.5 的 GPU |
| 测试 | GoogleTest（缺省时自动跳过测试目标） |
| 基准 | Google Benchmark（缺省时自动跳过基准目标，可用 `-DCMAKE_PREFIX_PATH=<prefix>` 指定安装位置） |

### 2.2 CPU-only 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DSCI_ENABLE_CUDA=OFF -DSCI_BUILD_TESTS=ON
cmake --build build -j"$(nproc)"

ctest --test-dir build --output-on-failure     # 56 个用例
./build/examples/simple_tensor                 # 张量/LayerNorm 演示
```

### 2.3 CUDA 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DSCI_ENABLE_CUDA=ON \
      -DSCI_BUILD_TESTS=ON -DSCI_BUILD_BENCHMARKS=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

默认编译 `sm_75;89;90;120`（覆盖 T4/RTX 20-50 系列与 A100/H100），可覆盖：

```bash
cmake -S . -B build -DSCI_CUDA_ARCHITECTURES="89;120"
```

也可以用仓库自带脚本：

```bash
./scripts/build.sh --release            # CUDA + tests + benchmarks
./scripts/build.sh --release --no-cuda  # 纯 CPU
./scripts/run_benchmarks.sh
```

### 2.4 测试

```bash
ctest --test-dir build --output-on-failure          # 全部
ctest --test-dir build -R MPSCQueue                 # 按套件过滤
./build/tests/integration/test_integration_tensor_ops --gtest_filter='*CrossDevice*'
```

测试分两层：`tests/unit`（41 个用例）与 `tests/integration`（15 个用例），覆盖张量语义、内存/分配器生命周期、
数学算子数值性质、MPSC 队列、线程池与 DAG 调度、跨设备迁移与 CUDA 优雅降级。详见 [§7](#7-测试)。

### 2.5 基准测试

```bash
# 自研吞吐基准（队列 / 内存池）
./build/examples/benchmark_test

# Google Benchmark 微基准与集成基准
./build/benchmarks/bench_elementwise
./build/benchmarks/bench_reduction
./build/benchmarks/bench_softmax
./build/benchmarks/bench_layernorm
./build/benchmarks/bench_transformer
./build/benchmarks/bench_attention

# 一次跑完全部基准
cmake --build build --target run_benchmarks
```

### 2.6 示例程序

| 程序 | 说明 |
| --- | --- |
| `examples/simple_tensor` | 张量创建、逐元素运算、LayerNorm 数值演示 |
| `examples/bridge_demo_gpu` | 桥接层（内存池 / 线程池 / IPC）+ GPU 内存使用演示（需 CUDA） |
| `examples/bridge_integration_test` | 桥接层端到端自检（27 项断言） |
| `examples/stress_test` | 多线程压力测试（MPSC、内存池并发、SPSC 吞吐） |
| `examples/benchmark_test` | 队列与内存池吞吐/延迟对比 |

## 3. 架构

```text
                      ┌──────────────────────────────────────────────┐
   应用 / 上层框架     │  训练循环 / 推理服务 / 科学计算 Pipeline      │
                      └──────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
┌────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│  scheduler     │           │  tensor + math   │           │  bridges         │
│  线程池 / DAG   │           │  Tensor / 算子    │           │  队列 / 池 / IPC  │
└────────────────┘           └──────────────────┘           └──────────────────┘
        │                              │                              │
        └──────────────┬───────────────┴──────────────┬───────────────┘
                       ▼                              ▼
              ┌──────────────────┐           ┌──────────────────┐
              │  memory          │           │  device          │
              │  池 / Slab / RAII │           │  CPU / CUDA / 流  │
              └──────────────────┘           └──────────────────┘
                       │                              │
                       └──────────────┬───────────────┘
                                      ▼
                       ┌──────────────────────────────┐
                       │  core：类型 / 状态 / 配置 / 计时 │
                       └──────────────────────────────┘
                                      │
                                      ▼
         ┌─────────────────────────────────────────────────────────┐
         │  cuda/kernels：elementwise / reduction / softmax /       │
         │  layernorm / attention（编译进 sci_bridges，CPU 可降级）  │
         └─────────────────────────────────────────────────────────┘
```

模块职责：

| 模块 | 头文件 | 实现要点 |
| --- | --- | --- |
| **core** | `include/core/*.hpp` | `DType`/`TensorShape`/`DeviceType`、`Status`+`Result<T>`、宏、配置、计时器、版本、平台抽象（INTERFACE 目标 `sci_core`） |
| **tensor** | `include/tensor/tensor.hpp` | `Tensor`/`TensorView`、Shape 校验、跨设备 `to()`、`clone()`、`Result` 错误传播 |
| **memory** | `include/memory/*.hpp` | `Allocator` 接口、`HostAllocator`、固定大小池、Slab、`BufferHandle`（RAII + 生命周期回调） |
| **device** | `include/device/*.hpp` | `Device`/`Stream`/`Event` 抽象、`DeviceManager` 自动探测与热插拔回调、`CudaDevice`/`CudaDeviceAllocator` |
| **scheduler** | `include/scheduler/*.hpp` | `ThreadPool`（动态扩缩、`enqueue` 返回 future）、`Task`/`TaskGraph`/`GraphExecutor` |
| **math** | `include/math/*.hpp` | 元素级、归约、Softmax/LogSoftmax/温度/hardmax、LayerNorm/RMSNorm，附 `math::ref::*` 参考实现 |
| **benchmark** | `include/benchmark/*.hpp` | `BenchmarkCase`/`BenchmarkRunner`，统一采集吞吐与延迟 |
| **bridges** | `include/bridges/*.hpp` | 无锁 SPSC/MPSC 队列、共享内存段与自旋锁、内存池/线程池桥接、`CudaKernelBridge` |
| **gpu** | `include/gpu/*.hpp` | `CudaAllocator`、`GpuTensor`、`GpuScheduler`、显存统计（`sci_gpu`，仅在 CUDA 打开时构建） |
| **cuda/kernels** | `cuda/kernels/*.cuh` | elementwise / reduction / softmax / layernorm / attention（含 FlashAttention、RoPE、masked）kernel 模板与 launch 封装 |

## 4. 核心组件与 API

所有类型位于 `sci::` 命名空间（队列等在 `sci::bridges::`、算子等在 `sci::math::`）。

### 4.1 无锁队列（SPSC / MPSC）

```cpp
#include "bridges/impl/ipc_queue_impl.hpp"
using namespace sci::bridges;

MPSCQueue<int> queue;
queue.Init(/*capacity=*/10000);

for (int i = 0; i < 1000; ++i) queue.Push(i);      // 多生产者安全
int v = 0;
while (queue.Pop(v)) { /* 消费 */ }                 // 单消费者

int items[100];
size_t n = queue.PopBatch(items, 100);              // 批量弹出

SPSCQueue<int> spsc;                                // 单生产者单消费者，最低延迟
spsc.Init(1024);
```

### 4.2 内存池与 RAII 缓冲区

```cpp
#include "bridges/pool_bridge.hpp"
#include "memory/buffer_handle.hpp"
using namespace sci;

bridges::FixedSizePoolBridge pool(/*block_size=*/64, /*blocks_per_chunk=*/1000);
void* p = pool.allocate(64);        // 失败返回 nullptr，不抛异常
pool.deallocate(p);

// 设备内存的 RAII 封装：析构自动归还
BufferHandle handle(DeviceManager::Instance().get_device(DeviceType::kCPU, 0).get(), 1024);
if (handle) { /* handle.data() 可用 */ }
```

### 4.3 张量与跨设备迁移

```cpp
#include "tensor/tensor.hpp"
using namespace sci;

auto cpu = DeviceManager::Instance().get_device(DeviceType::kCPU, 0);
Tensor x = Tensor::Ones({8}, DType::kFloat32, *cpu);

Tensor y = x.clone();                              // 同设备拷贝（CUDA 时走 DeviceToDevice）

auto on_gpu = x.to(DeviceType::kCUDA, 0);          // Result<Tensor>
if (on_gpu.ok()) {
    auto back = on_gpu->to(DeviceType::kCPU, 0);   // 显存 → 主机回拷
    if (back.ok()) { /* back->data_ptr<float>() 可安全在主机侧读取 */ }
} else {
    // 无 GPU/驱动时：on_gpu.error() == Status::CudaNotAvailable 等
}
```

### 4.4 设备抽象与优雅降级

```cpp
#include "device/device.hpp"
#include "device/cuda_device.hpp"
using namespace sci;

DeviceManager& dm = DeviceManager::Instance();     // get_device() 首次查询 GPU 时自动探测
auto gpu = dm.get_device(DeviceType::kCUDA, 0);    // 无 GPU 时为 nullptr

if (CudaDevice::available(0)) {
    // 真实设备路径
} else {
    // 优雅降级：CudaAllocator::allocate 返回 nullptr，Tensor::to(kCUDA) 返回明确错误
}
```

### 4.5 线程池与 DAG 调度

```cpp
#include "scheduler/thread_pool.hpp"
#include "scheduler/task_graph.hpp"
using namespace sci::scheduler;

ThreadPool pool(8);
auto f = pool.enqueue([](int a, int b) { return a + b; }, 1, 2);
int sum = f.get();
pool.wait_all();

// DAG：add → softmax，依赖顺序由执行器保证
auto pool_ptr = std::make_shared<ThreadPool>(4);
GraphExecutor executor(pool_ptr);
TaskGraph graph;

auto t_add = graph.add_task(std::make_shared<FunctionalTask>(
    []() { /* 计算 */ return Task::Status::kCompleted; }, "add"));
auto t_softmax = graph.add_task(std::make_shared<FunctionalTask>(
    []() { /* 计算 */ return Task::Status::kCompleted; }, "softmax"));
graph.add_edge(t_add, t_softmax);

Status st = executor.execute(graph);      // 拓扑调度，依赖先完成
```

### 4.6 数学算子

```cpp
#include "math/softmax.hpp"
#include "math/reduction.hpp"
using namespace sci;

Tensor x = Tensor::Randn({32, 512}, DType::kFloat32, *cpu);

auto y  = math::softmax(x, /*axis=*/-1, /*stream=*/nullptr);   // Result<Tensor>
auto s  = math::sum(x);                                       // 归约到标量
auto mx = math::max(x);                                       // 全局最大值
auto [vals, idx] = *math::max_with_indices(x, /*axis=*/1);    // 按轴最大值 + 索引
```

每个算子都提供 `math::ref::*` 形式的参考实现（如 `ref::softmax_f32`），便于数值对齐与基准对比。

### 4.7 CUDA Kernel

```cpp
#include "cuda/kernels/softmax.cuh"
using namespace sci::cuda::kernels;

// launch_* 模板封装了 grid/block 配置；stream 为 0 表示默认流
launch_softmax(d_out, d_in, /*rows=*/32, /*cols=*/512, /*stream=*/0);
```

`cuda/kernels/` 提供 elementwise、reduction、softmax、layernorm 与 attention（含 FlashAttention / masked / RoPE）模板；
kernel 源文件当前编译进 `sci_bridges`，无 CUDA 环境时整体跳过。

## 5. 构建选项

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `SCI_ENABLE_CUDA` | 启用 CUDA 设备后端与 kernel | `ON` |
| `SCI_ENABLE_OPENMP` | 启用 OpenMP 并行 | `ON` |
| `SCI_BUILD_TESTS` | 构建 GoogleTest 用例 | `OFF` |
| `SCI_BUILD_BENCHMARKS` | 构建 Google Benchmark 基准 | `OFF` |
| `SCI_CUDA_ARCHITECTURES` | 目标计算能力列表 | `75;89;90;120` |
| `CUDAToolkit_ROOT` | CUDA 安装位置（非默认路径时指定） | 自动探测 |
| `CMAKE_BUILD_TYPE` | 建议 `Release` | — |

## 6. 性能实测

**测试环境**：Intel 32 线程（5.46 GHz 睿频）+ NVIDIA GeForce RTX 5070 Laptop GPU（8 GB，驱动 615.71.09）+ CUDA 13.2 + GCC 15.2，
Release 构建。基准为微秒级测量，笔记本功耗/频率波动会带来 ±10% 抖动，数字用于量级参考而非硬件上限。

### 6.1 队列与内存池

```bash
./build/examples/benchmark_test        # 100000 次操作
```

| 组件 | 吞吐 | 延迟 |
| --- | --- | --- |
| SPSC 队列 | 3.0×10⁸ ops/s | 3.3 ns/op |
| MPSC 队列 | 1.5×10⁸ ops/s | 6.6 ns/op |
| 固定大小内存池 | 4.6×10⁷ ops/s | 22 ns/op |
| `new` / `delete`（对照） | 4.3×10⁷ ops/s | 23 ns/op |

### 6.2 算子与流水线（Google Benchmark）

```bash
./build/benchmarks/bench_elementwise --benchmark_min_time=0.05s
./build/benchmarks/bench_reduction  --benchmark_min_time=0.05s
./build/benchmarks/bench_softmax    --benchmark_min_time=0.05s
./build/benchmarks/bench_layernorm  --benchmark_min_time=0.05s
./build/benchmarks/bench_transformer --benchmark_min_time=0.1s
./build/benchmarks/bench_attention   --benchmark_min_time=0.1s
```

| 基准 | 规模 | 时间 | 吞吐 |
| --- | --- | --- | --- |
| `BM_TensorAdd` | 1024 元素 | 0.114 µs | 8.98×10⁹ 元素/s（100 GiB/s 等效） |
| `BM_TensorAdd` | 2048 元素 | 0.173 µs | 1.19×10¹⁰ 元素/s（133 GiB/s） |
| `BM_TensorSum` | 1024 元素 | 0.404 µs | 2.54×10⁹ 元素/s |
| `BM_Softmax` | 32×512 | 44.7 µs | 3.67×10⁸ 元素/s |
| `BM_LayerNorm` | 32×512 | 17.9 µs | 9.14×10⁸ 元素/s（6.8 GiB/s） |
| `BM_TransformerLayerCore` | 64×256 | 101 µs | 1.62×10⁸ 元素/s |
| `BM_Attention` | batch 4 / 16×32 | 17.8 µs | 3.68×10⁹ 元素/s |

> 说明：以上为当前 CPU 实现路径的实测值（`math::*` 尚未把 `launch_*` CUDA kernel 接到统一调度里），
> 因此把它们当作**基线**：后续 GPU 路径接入后应附「同规模、同硬件」的对照数据。

## 7. 测试

```bash
ctest --test-dir build --output-on-failure
# 100% tests passed out of 56
```

| 层 | 用例数 | 套件 | 覆盖内容 |
| --- | --- | --- | --- |
| 单元 | 41 | `TensorTest`(9)、`MemoryTest`(10)、`MathTest`(8)、`SchedulerTest`(9)、`MPSCQueueTest`(5) | 张量创建/视图/拷贝、分配器与池生命周期、算子数值性质、线程池行为、MPSC 并发正确性 |
| 集成 | 15 | `IntegrationTensorOpsTest`(11)、`IntegrationSchedulerTest`(4) | 元素级链式运算、Softmax/LayerNorm 数值性质、归约、跨设备迁移与回拷、CUDA 分配器降级、线程池+任务图流水线与依赖顺序 |

两种配置均已验证：**CPU-only（`SCI_ENABLE_CUDA=OFF`）56/56 通过**；**CUDA 构建（真实 GPU 存在）56/56 通过**。
此外 `docs/TASK_PLAN.md` 记录了使用 ThreadSanitizer 验证调度器竞态修复的过程。

## 8. 目录结构

```text
SciComputeInfra/
├── README.md / LICENSE
├── CMakeLists.txt                  # 顶层构建（选项、CUDA 定位、子目录）
├── docs/
│   ├── DESIGN.md                   # 架构与模块设计说明
│   └── TASK_PLAN.md                # 开发计划与阶段验收记录
├── include/                        # 公共头文件（core/tensor/memory/device/scheduler/
│                                   #  math/benchmark/bridges/gpu，另有 kernel/utils 头文件）
├── src/                            # 与 include 对应的实现（按模块生成独立静态库）
├── cuda/kernels/                   # CUDA kernel（elementwise/reduction/softmax/layernorm/attention）
├── benchmarks/                     # micro/ 4 个 + integration/ 2 个 Google Benchmark
├── examples/                       # 示例与自检程序（见 §2.6）
├── tests/unit、tests/integration/   # GoogleTest 用例
├── scripts/build.sh                # 一键构建脚本
└── scripts/run_benchmarks.sh       # 一键基准脚本
```

构建产出的库目标：`sci_compute`（汇总 INTERFACE）、`sci_core`、`sci_tensor`、`sci_memory`、`sci_device`、
`sci_scheduler`、`sci_math`、`sci_benchmark`、`sci_bridges`，以及 CUDA 开启时的 `sci_gpu`。

## 9. 项目状态与路线图

### 已完成

- [x] core / tensor / memory / device / scheduler / math / benchmark 模块与单元测试
- [x] 无锁 SPSC / MPSC 队列、共享内存段、自旋锁（bridges）
- [x] 内存池、Slab、TLS 缓存与 RAII `BufferHandle`
- [x] `DeviceManager` 自动探测 + 热插拔回调；CUDA 设备/分配器/流接入，无 GPU 时优雅降级
- [x] 线程池 + DAG 任务图（含重复入队竞态修复）
- [x] 数学算子（元素级/归约/Softmax/LayerNorm）+ 参考实现
- [x] CUDA kernel 模板库（elementwise/reduction/softmax/layernorm/attention）
- [x] 测试体系（56 用例）与基准体系（4 微基准 + 2 集成基准 + 自研吞吐基准）

### 规划中

- [ ] 把 `launch_*` CUDA kernel 接入 `math::*` 的统一调度（当前算子为 CPU 路径）并补充 GPU 侧对照基准
- [ ] Python 绑定与基准脚本（`python/` 目录已预留）
- [ ] `include/kernel/kernel_registry.hpp`、`include/utils/logger.hpp` 接入构建（当前为未使用的头文件）
- [ ] 计算图与自动微分运行时（`include/graph/`、`src/graph/` 目前为空目录，仅预留）
- [ ] 收敛 `cuda/CMakeLists.txt` 与 `src/bridges` 中重复的 kernel 编译路径（当前 kernel 编进 `sci_bridges`）
- [ ] CI（GitHub Actions：CPU 构建 + CUDA 容器构建 + ctest）

## 10. 已知限制

1. **算子目前是 CPU 路径**：`math::*` 尚未调用 CUDA kernel，GPU 加速需要按路线图接入。
2. **设备间（不同 GPU id）拷贝经主机暂存**：单卡机器上无法验证，正确性依赖 `copy_to_host` + `copy_to_device` 两步实现。
3. **`cuda/CMakeLists.txt` 定义的 `sci_cuda` 未接入顶层构建**，kernel 实际由 `src/bridges` 编译，存在两条重复路径。
4. **`examples/bridge_demo.cpp` 与 `examples/cuda_kernel_demo.cu` 未接入构建**（保留为参考示例）。
5. **无 CI**：测试与基准需要在本地手动执行，见 [§7](#7-测试)。
6. **基准数据来自单台笔记本**：用于量级参考，不代表服务器平台表现。

## 11. 贡献指南

```bash
# 提交前至少跑通 CPU-only 构建与测试（最快反馈回路）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCI_ENABLE_CUDA=OFF -DSCI_BUILD_TESTS=ON
cmake --build build -j"$(nproc)" && ctest --test-dir build --output-on-failure
```

- 代码风格：C++20，4 空格缩进，`sci::` 命名空间，接口用 `Result<T>`/`Status` 表达失败。
- 新增功能必须带测试（单元或集成）；性能改动必须给出 `benchmarks/` 的前后对比。
- 提交信息用动词开头（`feat:` / `fix:` / `docs:` / `test:` / `perf:`），必要时在正文说明动机与验证方式。

## 12. 许可证

本项目基于 **MIT License** 发布，见 [LICENSE](LICENSE)。
