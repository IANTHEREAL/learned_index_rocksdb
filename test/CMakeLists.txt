cmake_minimum_required(VERSION 3.16)

# Find Google Test
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "Google Test not found. Downloading and building from source...")
    
    # Download Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        release-1.12.1
    )
    
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    FetchContent_MakeAvailable(googletest)
endif()

# Test sources
set(TEST_SOURCES
    test_learned_index_block.cpp
    test_sst_learned_index_manager.cpp
    test_main.cpp
)

# Create test executable
add_executable(learned_index_tests ${TEST_SOURCES})

# Link libraries
target_link_libraries(learned_index_tests
    learned_index
    gtest
    gtest_main
    Threads::Threads
)

# Link CRC32C if available
if(CRC32C_LIB)
    target_link_libraries(learned_index_tests ${CRC32C_LIB})
endif()

# Include directories
target_include_directories(learned_index_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Add tests to CTest
include(GoogleTest)
gtest_discover_tests(learned_index_tests)