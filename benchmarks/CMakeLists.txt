# ============================================================================
# Benchmarks
# ============================================================================

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, benchmarks will not be built")
    return()
endif()

add_executable(bench_elementwise bench_elementwise.cpp)
target_link_libraries(bench_elementwise PRIVATE
    sci_compute
    benchmark::benchmark
)

add_executable(bench_reduction bench_reduction.cpp)
target_link_libraries(bench_reduction PRIVATE
    sci_compute
    benchmark::benchmark
)

add_executable(bench_softmax bench_softmax.cpp)
target_link_libraries(bench_softmax PRIVATE
    sci_compute
    benchmark::benchmark
)

# Register benchmarks
add_custom_target(run_benchmarks
    COMMAND bench_elementwise
    COMMAND bench_reduction
    COMMAND bench_softmax
    COMMENT "Running all benchmarks"
)
