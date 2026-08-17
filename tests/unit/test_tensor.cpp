#include <gtest/gtest.h>
#include "tensor/tensor.hpp"
#include "device/device.hpp"

using namespace sci;

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = GetCPUDevice();
    }
    
    std::shared_ptr<Device> device_;
};

TEST_F(TensorTest, EmptyTensor) {
    Tensor t;
    EXPECT_FALSE(t);
    EXPECT_EQ(t.num_elements(), 0);
}

TEST_F(TensorTest, CreateTensor) {
    Tensor t({2, 3}, DType::kFloat32, *device_);
    EXPECT_TRUE(t);
    EXPECT_EQ(t.ndims(), 2);
    EXPECT_EQ(t.dim(0), 2);
    EXPECT_EQ(t.dim(1), 3);
    EXPECT_EQ(t.num_elements(), 6);
    EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST_F(TensorTest, ZerosTensor) {
    Tensor t = Tensor::Zeros({10, 10}, DType::kFloat32, *device_);
    const float* data = t.data_ptr<float>();
    
    for (size_t i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.0f);
    }
}

TEST_F(TensorTest, OnesTensor) {
    Tensor t = Tensor::Ones({5, 5}, DType::kFloat32, *device_);
    const float* data = t.data_ptr<float>();
    
    for (size_t i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 1.0f);
    }
}

TEST_F(TensorTest, FillTensor) {
    Tensor t({3, 3}, DType::kFloat32, *device_);
    float val = 42.5f;
    t.fill(&val);
    
    const float* data = t.data_ptr<float>();
    for (size_t i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 42.5f);
    }
}

TEST_F(TensorTest, CloneTensor) {
    Tensor t({4, 4}, DType::kFloat32, *device_);
    float val = 3.14f;
    t.fill(&val);
    
    Tensor t2 = t.clone();
    EXPECT_EQ(t2.shape(), t.shape());
    EXPECT_EQ(t2.dtype(), t.dtype());
    
    const float* data1 = t.data_ptr<float>();
    const float* data2 = t2.data_ptr<float>();
    for (size_t i = 0; i < t.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

TEST_F(TensorTest, CopyFrom) {
    Tensor t1({2, 2}, DType::kFloat32, *device_);
    float val = 1.0f;
    t1.fill(&val);
    
    Tensor t2({2, 2}, DType::kFloat32, *device_);
    t2.copy_from(t1);
    
    const float* data1 = t1.data_ptr<float>();
    const float* data2 = t2.data_ptr<float>();
    for (size_t i = 0; i < t1.num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data1[i], data2[i]);
    }
}

TEST_F(TensorTest, Reshape) {
    Tensor t({2, 6}, DType::kFloat32, *device_);
    EXPECT_EQ(t.num_elements(), 12);
    
    auto view = t.reshape({3, 4});
    EXPECT_EQ(view.ndims(), 2);
    EXPECT_EQ(view.dim(0), 3);
    EXPECT_EQ(view.dim(1), 4);
}

TEST_F(TensorTest, TensorView) {
    Tensor t({3, 4}, DType::kFloat32, *device_);
    float val = 5.0f;
    t.fill(&val);
    
    TensorView view = t.view();
    EXPECT_EQ(view.ndims(), 2);
    EXPECT_EQ(view.num_elements(), 12);
}
