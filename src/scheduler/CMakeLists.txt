add_library(sci_scheduler STATIC
    thread_pool.cpp
    task.cpp
    task_graph.cpp
)
target_include_directories(sci_scheduler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(sci_scheduler PRIVATE
    Threads::Threads
)
