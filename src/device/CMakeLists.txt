add_library(sci_device STATIC
    device.cpp
    stream.cpp
)
target_include_directories(sci_device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_device PRIVATE
    Threads::Threads
)
