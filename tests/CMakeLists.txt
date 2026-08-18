# ============================================================================
# Tests
# ============================================================================

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "GTest not found, tests will not be built")
    return()
endif()

add_subdirectory(unit)
add_subdirectory(integration)
