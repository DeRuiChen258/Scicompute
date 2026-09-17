#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "tensor/tensor.hpp"
#include "device/device.hpp"
#include "device/cuda_device.hpp"
#include "device/cuda_allocator.hpp"
#include "memory/allocator.hpp"
#include "memory/buffer_handle.hpp"
#include "math/elementwise.hpp"
#include "math/reduction.hpp"
#include "math/softmax.hpp"
#include "math/normalization.hpp"

using namespace sci;

namespace {

std::shared_ptr<Device> CpuDevice() {
    return GetCPUDevice();
}

} // namespace

// ============================================================================
// 元素级运算流水线: add -> mul -> sub -> div
// ============================================================================
TEST(IntegrationTensorOpsTest, ElementwiseChain) {
    auto device = CpuDevice();

    Tensor a = Tensor::Full({4, 4}, DType::kFloat32, *device, 2.0f);
    Tensor b = Tensor::Full({4, 4}, DType::kFloat32, *device, 3.0f);
    Tensor c = Tensor::Full({4, 4}, DType::kFloat32, *device, 4.0f);

    auto r1 = math::add(a, b);
    ASSERT_TRUE(r1.ok());
    auto r2 = math::mul(*r1, c);
    ASSERT_TRUE(r2.ok());
    auto r3 = math::sub(*r2, a);
    ASSERT_TRUE(r3.ok());
    auto r4 = math::div(*r3, b);
    ASSERT_TRUE(r4.ok());

    // ((2+3)*4 - 2) / 3 == 6
    const float* data = r4->data_ptr<float>();
    for (size_t i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(data[i], 6.0f);
    }
}

// ============================================================================
// Softmax / LogSoftmax 数值性质
// ============================================================================
TEST(IntegrationTensorOpsTest, SoftmaxNormalizesRows) {
    auto device = CpuDevice();

    Tensor x({3, 5}, DType::kFloat32, *device);
    float* xd = x.data_ptr<float>();
    for (int i = 0; i < 15; ++i) {
        xd[i] = static_cast<float>(i % 5) * 0.5f;
    }

    auto r = math::softmax(x, -1);
    ASSERT_TRUE(r.ok());
    const float* sd = r->data_ptr<float>();
    for (int row = 0; row < 3; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < 5; ++col) {
            sum += sd[row * 5 + col];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
        EXPECT_GT(sd[row * 5 + 4], sd[row * 5 + 0]);
    }

    auto lr = math::log_softmax(x, -1);
    ASSERT_TRUE(lr.ok());
    const float* ld = lr->data_ptr<float>();
    for (int row = 0; row < 3; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < 5; ++col) {
            sum += std::exp(ld[row * 5 + col]);
        }
        EXPECT_NEAR(sum, 1.0f, 1e-4f);
    }
}

// ============================================================================
// LayerNorm: 每行输出应近似零均值、单位方差
// ============================================================================
TEST(IntegrationTensorOpsTest, LayerNormProducesUnitVariance) {
    auto device = CpuDevice();

    Tensor x({4, 8}, DType::kFloat32, *device);
    float* xd = x.data_ptr<float>();
    for (int i = 0; i < 32; ++i) {
        xd[i] = static_cast<float>(i % 8) * 0.7f;
    }

    Tensor w = Tensor::Ones({8}, DType::kFloat32, *device);
    Tensor b = Tensor::Zeros({8}, DType::kFloat32, *device);

    auto r = math::layer_norm(x, w, b, 1e-5f);
    ASSERT_TRUE(r.ok());
    const float* yd = r->data_ptr<float>();
    for (int row = 0; row < 4; ++row) {
        float mean = 0.0f;
        for (int c = 0; c < 8; ++c) {
            mean += yd[row * 8 + c];
        }
        mean /= 8.0f;

        float var = 0.0f;
        for (int c = 0; c < 8; ++c) {
            float diff = yd[row * 8 + c] - mean;
            var += diff * diff;
        }
        var /= 8.0f;

        EXPECT_NEAR(mean, 0.0f, 1e-5f);
        EXPECT_NEAR(var, 1.0f, 1e-4f);
    }
}

// ============================================================================
// 归约运算 (sum/mean/max/min/argmax/var/norm)
// ============================================================================
TEST(IntegrationTensorOpsTest, Reductions) {
    auto device = CpuDevice();

    Tensor a({10}, DType::kFloat32, *device);
    float* ad = a.data_ptr<float>();
    for (int i = 0; i < 10; ++i) {
        ad[i] = static_cast<float>(i + 1);
    }

    auto s = math::sum(a);
    ASSERT_TRUE(s.ok());
    EXPECT_FLOAT_EQ(*s->data_ptr<float>(), 55.0f);

    auto m = math::mean(a);
    ASSERT_TRUE(m.ok());
    EXPECT_FLOAT_EQ(*m->data_ptr<float>(), 5.5f);

    auto mx = math::max(a);
    ASSERT_TRUE(mx.ok());
    EXPECT_FLOAT_EQ(*mx->data_ptr<float>(), 10.0f);

    auto mn = math::min(a);
    ASSERT_TRUE(mn.ok());
    EXPECT_FLOAT_EQ(*mn->data_ptr<float>(), 1.0f);

    auto am = math::argmax(a, -1);
    ASSERT_TRUE(am.ok());
    EXPECT_EQ(*am->data_ptr<int64_t>(), 9);

    auto v = math::var(a);
    ASSERT_TRUE(v.ok());
    EXPECT_NEAR(*v->data_ptr<float>(), 8.25f, 1e-4f);

    auto n = math::norm(a, 2.0f);
    ASSERT_TRUE(n.ok());
    EXPECT_NEAR(*n->data_ptr<float>(), std::sqrt(385.0f), 1e-3f);
}

// ============================================================================
// 未实现/非法路径返回错误而非崩溃
// ============================================================================
TEST(IntegrationTensorOpsTest, NotImplementedPathsReturnErrors) {
    auto device = CpuDevice();
    Tensor a({2, 3}, DType::kFloat32, *device);

    auto along = math::sum(a, 0);
    EXPECT_FALSE(along.ok());
    EXPECT_EQ(along.error().code(), StatusCode::kNotImplemented);

    auto sm_axis = math::softmax(a, 0);
    EXPECT_FALSE(sm_axis.ok());
}

// ============================================================================
// Tensor View 操作 (slice / transpose / reshape) 与拷贝
// ============================================================================
TEST(IntegrationTensorOpsTest, TensorViewsAndCopy) {
    auto device = CpuDevice();

    Tensor x({4, 4}, DType::kFloat32, *device);
    float* xd = x.data_ptr<float>();
    for (int i = 0; i < 16; ++i) {
        xd[i] = static_cast<float>(i);
    }

    auto sliced = x.slice(0, 1, 3);
    EXPECT_EQ(sliced.num_elements(), 8);
    const float* sd = static_cast<const float*>(sliced.data());
    EXPECT_FLOAT_EQ(sd[0], 4.0f);
    EXPECT_FLOAT_EQ(sd[7], 11.0f);

    auto transposed = x.transpose(0, 1);
    EXPECT_EQ(transposed.num_elements(), 16);
    const float* td = static_cast<const float*>(transposed.data());
    // 视图按交换后的 strides 索引: 逻辑 (0,1) == x[1][0] == 4, 位于原始偏移 4
    EXPECT_FLOAT_EQ(td[4], 4.0f);
    EXPECT_FLOAT_EQ(td[1], 1.0f); // 逻辑 (1,0) == x[0][1] == 1

    auto reshaped = x.reshape({2, 8});
    EXPECT_EQ(reshaped.num_elements(), 16);
    EXPECT_FLOAT_EQ(static_cast<const float*>(reshaped.data())[9], 9.0f);

    auto clone = x.clone();
    EXPECT_EQ(clone.shape(), x.shape());
    EXPECT_EQ(clone.dtype(), x.dtype());
    const float* cd = clone.data_ptr<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(cd[i], xd[i]);
    }

    auto moved = x.to(*device);
    EXPECT_EQ(moved.device_type(), DeviceType::kCPU);
    const float* md = moved.data_ptr<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(md[i], xd[i]);
    }
}

// ============================================================================
// 跨设备迁移: 同设备类型成功; 无 CUDA 时返回明确错误
// ============================================================================
TEST(IntegrationTensorOpsTest, CrossDeviceTransfer) {
    auto device = CpuDevice();
    Tensor x = Tensor::Ones({8}, DType::kFloat32, *device);

    auto same = x.to(DeviceType::kCPU, 0);
    ASSERT_TRUE(same.ok());
    EXPECT_EQ(same->device_type(), DeviceType::kCPU);
    for (size_t i = 0; i < 8; ++i) {
        EXPECT_FLOAT_EQ(same->data_ptr<float>()[i], 1.0f);
    }

    auto cuda = x.to(DeviceType::kCUDA, 0);
    if (CudaDevice::available(0)) {
        ASSERT_TRUE(cuda.ok());
        EXPECT_EQ(cuda->device_type(), DeviceType::kCUDA);

        // 显存指针不能由 host 直接解引用：拷回 CPU 后校验数值（同时验证回程拷贝）
        auto round_trip = cuda->to(DeviceType::kCPU, 0);
        ASSERT_TRUE(round_trip.ok());
        for (size_t i = 0; i < 8; ++i) {
            EXPECT_FLOAT_EQ(round_trip->data_ptr<float>()[i], 1.0f);
        }

        // 同设备（显存内）克隆也必须走设备内拷贝，而不是主机 memcpy
        auto cloned = cuda->clone();
        auto cloned_back = cloned.to(DeviceType::kCPU, 0);
        ASSERT_TRUE(cloned_back.ok());
        for (size_t i = 0; i < 8; ++i) {
            EXPECT_FLOAT_EQ(cloned_back->data_ptr<float>()[i], 1.0f);
        }
    } else {
        ASSERT_FALSE(cuda.ok());
    }
}

// ============================================================================
// DeviceManager + CUDA 后端注册 (无 GPU 环境优雅降级)
// ============================================================================
TEST(IntegrationTensorOpsTest, DeviceManagerCudaRegistration) {
    DeviceManager& manager = DeviceManager::Instance();
    manager.ScanDevices();

    if (CudaDevice::available(0)) {
        auto cuda = manager.get_device(DeviceType::kCUDA, 0);
        ASSERT_NE(cuda, nullptr);
        EXPECT_EQ(cuda->type(), DeviceType::kCUDA);
        EXPECT_EQ(cuda->id(), 0);
        EXPECT_TRUE(cuda->is_available());
    } else {
        EXPECT_EQ(manager.get_device(DeviceType::kCUDA, 0), nullptr);
        EXPECT_FALSE(CudaDevice::available(0));
    }
}

// ============================================================================
// CudaDeviceAllocator (sci::Allocator 接口) 优雅降级
// ============================================================================
TEST(IntegrationTensorOpsTest, CudaDeviceAllocatorGracefulDegradation) {
    auto& alloc = CudaDeviceAllocator::Instance(0);

    if (CudaDevice::available(0)) {
        void* p = alloc.allocate(1024);
        if (p != nullptr) {
            EXPECT_EQ(alloc.allocated_bytes(), 1024u);
            alloc.deallocate(p);
            EXPECT_EQ(alloc.allocated_bytes(), 0u);
        }
    } else {
        EXPECT_EQ(alloc.allocate(1024), nullptr);
        EXPECT_EQ(alloc.allocated_bytes(), 0u);
    }
}

// ============================================================================
// 分配器生命周期与统计
// ============================================================================
TEST(IntegrationTensorOpsTest, HostAllocatorLifecycle) {
    auto& host = HostAllocator::Instance();

    void* p1 = host.allocate(4096);
    void* p2 = host.allocate(8192);
    ASSERT_NE(p1, nullptr);
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(host.allocated_bytes(), 4096u + 8192u);

    host.deallocate(p1);
    host.deallocate(p2);
    EXPECT_EQ(host.allocated_bytes(), 0u);
    EXPECT_GE(host.stats().num_allocations, 2u);
    EXPECT_GE(host.stats().num_deallocations, 2u);
}

// ============================================================================
// BufferHandle RAII 封装
// ============================================================================
TEST(IntegrationTensorOpsTest, BufferHandleRaisesDeviceMemory) {
    auto device = CpuDevice();

    BufferHandle handle(device.get(), 1024);
    ASSERT_TRUE(handle);
    EXPECT_EQ(handle.bytes(), 1024u);
    EXPECT_EQ(handle.device_type(), DeviceType::kCPU);

    handle.reset();
    EXPECT_FALSE(handle);
}
