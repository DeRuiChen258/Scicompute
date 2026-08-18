# ============================================================================
# Unit Tests
# ============================================================================

# Unit tests
add_executable(test_tensor test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE
    sci_compute
    GTest::gtest_main
)

add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE
    sci_compute
    GTest::gtest_main
)

add_executable(test_memory test_memory.cpp)
target_link_libraries(test_memory PRIVATE
    sci_compute
    sci_bridges
    GTest::gtest_main
)

add_executable(test_mpsc_queue test_mpsc_queue.cpp)
target_link_libraries(test_mpsc_queue PRIVATE
    sci_compute
    sci_bridges
    GTest::gtest_main
)

add_executable(test_scheduler test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE
    sci_compute
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_tensor)
gtest_discover_tests(test_math)
gtest_discover_tests(test_memory)
gtest_discover_tests(test_mpsc_queue)
gtest_discover_tests(test_scheduler)
