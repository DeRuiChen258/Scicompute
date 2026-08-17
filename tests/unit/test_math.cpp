#include <gtest/gtest.h>
#include "math/elementwise.hpp"
#include "math/reduction.hpp"
#include "math/softmax.hpp"
#include "math/normalization.hpp"
#include "tensor/tensor.hpp"
#include "device/device.hpp"

using namespace sci;

class MathTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = GetCPUDevice();
    }
    
    std::shared_ptr<Device> device_;
};

TEST_F(MathTest, Add) {
    Tensor a = Tensor::Ones({100}, DType::kFloat32, *device_);
    Tensor b = Tensor::Full({100}, DType::kFloat32, *device_, static_cast<float>(2.0));
    
    auto result = math::add(a, b);
    ASSERT_TRUE(result.ok());
    
    const float* data = result->data_ptr<float>();
    for (size_t i = 0; i < result->num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 3.0f);
    }
}

TEST_F(MathTest, Mul) {
    Tensor a = Tensor::Full({100}, DType::kFloat32, *device_, static_cast<float>(2.0));
    Tensor b = Tensor::Full({100}, DType::kFloat32, *device_, static_cast<float>(3.0));
    
    auto result = math::mul(a, b);
    ASSERT_TRUE(result.ok());
    
    const float* data = result->data_ptr<float>();
    for (size_t i = 0; i < result->num_elements(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 6.0f);
    }
}

TEST_F(MathTest, Sum) {
    Tensor a({10, 10}, DType::kFloat32, *device_);
    float val = 1.0f;
    a.fill(&val);
    
    auto result = math::sum(a);
    ASSERT_TRUE(result.ok());
    
    const float* data = result->data_ptr<float>();
    EXPECT_FLOAT_EQ(*data, 100.0f);
}

TEST_F(MathTest, Mean) {
    Tensor a({10, 10}, DType::kFloat32, *device_);
    float val = 5.0f;
    a.fill(&val);
    
    auto result = math::mean(a);
    ASSERT_TRUE(result.ok());
    
    const float* data = result->data_ptr<float>();
    EXPECT_FLOAT_EQ(*data, 5.0f);
}

TEST_F(MathTest, Max) {
    Tensor a({5}, DType::kFloat32, *device_);
    float* data = a.data_ptr<float>();
    data[0] = 1.0f;
    data[1] = 5.0f;
    data[2] = 3.0f;
    data[3] = 2.0f;
    data[4] = 4.0f;
    
    auto result = math::max(a);
    ASSERT_TRUE(result.ok());
    
    const float* max_val = result->data_ptr<float>();
    EXPECT_FLOAT_EQ(*max_val, 5.0f);
}

TEST_F(MathTest, Softmax) {
    Tensor a({3}, DType::kFloat32, *device_);
    float* data = a.data_ptr<float>();
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = 2.0f;
    
    auto result = math::softmax(a);
    ASSERT_TRUE(result.ok());
    
    // Check that softmax sums to 1
    const float* out = result->data_ptr<float>();
    float sum = out[0] + out[1] + out[2];
    EXPECT_NEAR(sum, 1.0f, 1e-6f);
    
    // Check ordering
    EXPECT_TRUE(out[2] >= out[1]);
    EXPECT_TRUE(out[1] >= out[0]);
}

TEST_F(MathTest, LayerNorm) {
    Tensor x({2, 4}, DType::kFloat32, *device_);
    Tensor weight({4}, DType::kFloat32, *device_);
    Tensor bias({4}, DType::kFloat32, *device_);
    
    float w_val = 1.0f, b_val = 0.0f;
    weight.fill(&w_val);
    bias.fill(&b_val);
    
    // Fill with some values
    float* x_data = x.data_ptr<float>();
    for (int i = 0; i < 8; ++i) x_data[i] = static_cast<float>(i);
    
    auto result = math::layer_norm(x, weight, bias, 1e-5f);
    ASSERT_TRUE(result.ok());
    
    // Check output shape
    EXPECT_EQ(result->shape().ndims(), 2);
    EXPECT_EQ(result->dim(0), 2);
    EXPECT_EQ(result->dim(1), 4);
}

TEST_F(MathTest, RMSNorm) {
    Tensor x({2, 4}, DType::kFloat32, *device_);
    Tensor weight({4}, DType::kFloat32, *device_);
    
    float w_val = 1.0f;
    weight.fill(&w_val);
    
    float* x_data = x.data_ptr<float>();
    for (int i = 0; i < 8; ++i) x_data[i] = static_cast<float>(i);
    
    auto result = math::rms_norm(x, weight, 1e-5f);
    ASSERT_TRUE(result.ok());
    
    EXPECT_EQ(result->shape().ndims(), 2);
    EXPECT_EQ(result->dim(0), 2);
    EXPECT_EQ(result->dim(1), 4);
}
