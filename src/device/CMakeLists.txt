add_library(sci_device STATIC
    device.cpp
    stream.cpp
    cuda_device.cpp
)
target_include_directories(sci_device PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_device PRIVATE
    Threads::Threads
)

if(SCI_ENABLE_CUDA)
    target_include_directories(sci_device PRIVATE
        ${CUDA_INCLUDE_DIRS}
    )
    target_compile_definitions(sci_device PUBLIC
        SCI_USE_CUDA=1
        SCI_ENABLE_CUDA=1
    )
    target_link_libraries(sci_device PRIVATE
        CUDA::cudart
    )
endif()
