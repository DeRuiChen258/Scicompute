add_library(sci_benchmark STATIC
    benchmark_runner.cpp
)
target_include_directories(sci_benchmark PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_benchmark PRIVATE
    sci_tensor
    sci_math
)
