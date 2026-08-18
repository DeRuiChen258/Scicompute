cmake_minimum_required(VERSION 3.20)

# GPU Bridge Demo
if(SCI_ENABLE_CUDA)
    add_executable(bridge_demo_gpu
        bridge_demo_gpu.cpp
        ${CMAKE_SOURCE_DIR}/src/bridges/pool_bridge.cpp
        ${CMAKE_SOURCE_DIR}/src/bridges/thread_pool_bridge.cpp
        ${CMAKE_SOURCE_DIR}/src/bridges/ipc_bridge.cpp
    )
    target_include_directories(bridge_demo_gpu PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/bridges
    )
    target_link_libraries(bridge_demo_gpu PRIVATE
        sci_compute
        sci_gpu
        Threads::Threads
    )
    target_compile_definitions(bridge_demo_gpu PRIVATE
        SCI_USE_CUDA=1
    )
endif()

# Simple Tensor Demo
add_executable(simple_tensor simple_tensor.cpp)
target_link_libraries(simple_tensor PRIVATE sci_compute)

# Bridge Integration Test
add_executable(bridge_integration_test bridge_integration_test.cpp)
target_link_libraries(bridge_integration_test PRIVATE
    sci_compute
    sci_bridges
    Threads::Threads
)
target_include_directories(bridge_integration_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bridges
)
if(SCI_ENABLE_CUDA)
    target_link_libraries(bridge_integration_test PRIVATE sci_gpu)
    target_compile_definitions(bridge_integration_test PRIVATE SCI_USE_CUDA=1)
endif()

# Stress Test
add_executable(stress_test stress_test.cpp)
target_link_libraries(stress_test PRIVATE
    sci_compute
    sci_bridges
    Threads::Threads
)
target_include_directories(stress_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bridges
)

# Benchmark Test
add_executable(benchmark_test benchmark_test.cpp)
target_link_libraries(benchmark_test PRIVATE
    sci_compute
    sci_bridges
    sci_benchmark
    Threads::Threads
)
target_include_directories(benchmark_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/bridges
)
