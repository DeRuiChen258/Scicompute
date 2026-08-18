# ============================================================================
# Integration Tests
# ============================================================================

# 调度器集成测试 (线程池 + 任务图 + Tensor 运算流水线)
add_executable(test_integration_scheduler test_scheduler.cpp)
target_link_libraries(test_integration_scheduler PRIVATE
    sci_compute
    GTest::gtest_main
)

# Tensor 运算集成测试 (Tensor + Memory + Device + Math 组合)
add_executable(test_integration_tensor_ops test_tensor_ops.cpp)
target_link_libraries(test_integration_tensor_ops PRIVATE
    sci_compute
    GTest::gtest_main
)

# 注册测试
include(GoogleTest)
gtest_discover_tests(test_integration_scheduler)
gtest_discover_tests(test_integration_tensor_ops)
