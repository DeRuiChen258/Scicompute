# SciComputeInfra

高性能科学计算基础设施库 - RL训练加速专用

## 特性

### 核心组件

- **IPC通信桥接**: 无锁队列 (SPSC/MPSC)，共享内存
- **内存管理**: 固定大小池、Slab分配器、RAII封装
- **设备抽象**: CPU/GPU统一接口，热插拔支持
- **数学运算**: LayerNorm, Softmax, 元素级操作 (vLLM CUDA kernels)

### 技术亮点

1. **无锁数据结构**
   
   - SPSC队列: ~315M ops/s (环形缓冲区)
   - MPSC队列: ~36M ops/s (Michael-Scott + ABA防护)
   - 自动按需扩容

2. **内存管理**
   
   - 固定大小内存池: ~82M ops/s
   - RAII内存块封装
   - 批量操作接口

3. **设备抽象**
   
   - 统一Device接口
   - 设备热插拔回调
   - CUDA自动降级CPU

## 编译

```bash
mkdir build && cd build
cmake .. -DSCI_ENABLE_CUDA=ON
make -j4
```

## 测试

```bash
# 集成测试
./examples/bridge_integration_test

# 压力测试
./examples/stress_test

# 性能基准
./examples/benchmark_test
```

## API示例

### MPSC队列

```cpp
#include "bridges/ipc_bridge.hpp"

sci::bridges::MPSCQueue<int> queue;
queue.Init(10000);

// 单线程推送
for (int i = 0; i < 1000; ++i) {
    queue.Push(i);
}

// 单线程弹出
int value;
while (queue.Pop(value)) {
    // 处理value
}

// 批量操作
int items[100];
size_t count = queue.PopBatch(items, 100);
```

### 内存池

```cpp
#include "bridges/pool_bridge.hpp"

sci::bridges::FixedSizePoolBridge pool(64, 1000);

// 分配
void* ptr = pool.allocate(64);
if (ptr) {
    // 使用ptr
    pool.deallocate(ptr);
}
```

### 设备管理

```cpp
#include "device/device.hpp"

auto* cpu = sci::DeviceManager::GetInstance()->GetCPUDevice();
auto* gpu = sci::DeviceManager::GetInstance()->GetGPUDevice(0);
```

## 架构

```
include/
├── bridges/          # IPC桥接层
│   ├── impl/         # 无锁队列实现
│   ├── ipc_bridge.hpp
│   └── pool_bridge.hpp
├── core/             # 核心类型和工具
│   ├── status.hpp    # Result<T> 错误处理
│   └── memory_resource.hpp  # RAII封装
├── device/           # 设备抽象
└── math/             # 数学运算
```

## 性能基准

| 组件         | 吞吐量        | 延迟         |
| ---------- | ---------- | ---------- |
| SPSC队列     | 315M ops/s | 3.2 ns/op  |
| MPSC队列     | 36M ops/s  | 27.5 ns/op |
| 内存池        | 82M ops/s  | 12.2 ns/op |
| new/delete | 41M ops/s  | 24.2 ns/op |

## 依赖

- C++17 或更高
- CMake 3.20+
- CUDA 11.0+ (可选)
- OpenMP (可选)

## 许可证

MIT
