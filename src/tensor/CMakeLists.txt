add_library(sci_tensor STATIC
    tensor.cpp
)
target_include_directories(sci_tensor PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_tensor PRIVATE
    sci_memory
    sci_device
)
