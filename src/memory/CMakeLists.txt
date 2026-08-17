add_library(sci_memory STATIC
    allocator.cpp
    buffer_handle.cpp
    memory_pool.cpp
)
target_include_directories(sci_memory PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_memory PRIVATE
    sci_device
)
