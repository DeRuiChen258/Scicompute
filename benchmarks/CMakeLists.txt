# ============================================================================
# Benchmarks
# ============================================================================

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found, benchmarks will not be built")
    return()
endif()

# Auto-generated benchmarkTargets-*.cmake may reference a stale install prefix
# because the install tree was relocated. Derive the actual prefix from
# benchmark_DIR and override IMPORTED_LOCATION with the static library we built.
get_filename_component(_sci_benchmark_prefix "${benchmark_DIR}" DIRECTORY)
get_filename_component(_sci_benchmark_prefix "${_sci_benchmark_prefix}" DIRECTORY)
get_filename_component(_sci_benchmark_prefix "${_sci_benchmark_prefix}" DIRECTORY)
# Find the static .a explicitly. The .so symlink in this install tree points
# back to libbenchmark.a, which makes CMake import the wrong file type and the
# linker to drop symbols.
# The install layout can be /usr/lib or /usr/lib/x86_64-linux-gnu depending
# on whether the install was relocated; check both.
foreach(_sci_lib_dir
        "${_sci_benchmark_prefix}"
        "${_sci_benchmark_prefix}/lib"
        "${_sci_benchmark_prefix}/lib/x86_64-linux-gnu")
    if(EXISTS "${_sci_lib_dir}/libbenchmark.a")
        file(REAL_PATH "${_sci_lib_dir}/libbenchmark.a" _sci_benchmark_lib)
        break()
    endif()
endforeach()
if(_sci_benchmark_lib AND TARGET benchmark::benchmark)
    set_target_properties(benchmark::benchmark PROPERTIES
        IMPORTED_LOCATION "${_sci_benchmark_lib}"
    )
    message(STATUS "benchmark::benchmark -> ${_sci_benchmark_lib} (relocated)")
endif()

add_executable(bench_elementwise micro/bench_elementwise.cpp)
target_link_libraries(bench_elementwise PRIVATE sci_compute benchmark::benchmark)

add_executable(bench_reduction micro/bench_reduction.cpp)
target_link_libraries(bench_reduction PRIVATE sci_compute benchmark::benchmark)

add_executable(bench_softmax micro/bench_softmax.cpp)
target_link_libraries(bench_softmax PRIVATE sci_compute benchmark::benchmark)

# Micro benchmarks
add_executable(bench_layernorm micro/bench_layernorm.cpp)
target_link_libraries(bench_layernorm PRIVATE sci_compute benchmark::benchmark)

# Integration benchmarks
add_executable(bench_transformer integration/bench_transformer.cpp)
target_link_libraries(bench_transformer PRIVATE sci_compute benchmark::benchmark)

add_executable(bench_attention integration/bench_attention.cpp)
target_link_libraries(bench_attention PRIVATE sci_compute benchmark::benchmark)

# Register benchmarks
add_custom_target(run_benchmarks
    COMMAND bench_elementwise
    COMMAND bench_reduction
    COMMAND bench_softmax
    COMMAND bench_layernorm
    COMMAND bench_transformer
    COMMAND bench_attention
    COMMENT "Running all benchmarks"
)
