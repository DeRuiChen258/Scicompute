add_library(sci_math STATIC
    elementwise.cpp
    reduction.cpp
    softmax.cpp
    normalization.cpp
)
target_include_directories(sci_math PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_math PRIVATE
    sci_tensor
    sci_device
)
