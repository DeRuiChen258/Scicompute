add_subdirectory(core)
add_subdirectory(device)
add_subdirectory(memory)
add_subdirectory(tensor)
add_subdirectory(scheduler)
add_subdirectory(math)
add_subdirectory(benchmark)

add_library(sci_compute INTERFACE)
target_include_directories(sci_compute INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_compute INTERFACE
    Threads::Threads
    sci_core
    sci_device
    sci_memory
    sci_tensor
    sci_scheduler
    sci_math
    sci_benchmark
)

if(SCI_ENABLE_OPENMP)
    target_link_libraries(sci_compute INTERFACE OpenMP::OpenMP_CXX)
endif()
