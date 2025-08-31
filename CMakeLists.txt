cmake_minimum_required(VERSION 3.10)
project(learned_index_rocksdb)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Check for crc32c library
pkg_check_modules(CRC32C REQUIRED libcrc32c)

# Include directories
include_directories(include)
include_directories(${CRC32C_INCLUDE_DIRS})

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Source files
set(SOURCES
    src/learned_index/learned_index_block.cpp
    src/learned_index/ml_model.cpp
    src/learned_index/sst_learned_index_manager.cpp
)

# Create library
add_library(learned_index_rocksdb ${SOURCES})

# Link libraries
target_link_libraries(learned_index_rocksdb ${CRC32C_LIBRARIES})

# Add test executable if Google Test is available
find_package(GTest QUIET)
if(GTEST_FOUND)
    enable_testing()
    
    set(TEST_SOURCES
        tests/learned_index/test_learned_index_block.cpp
        tests/learned_index/test_ml_model.cpp
        tests/learned_index/test_sst_learned_index_manager.cpp
    )
    
    add_executable(learned_index_tests ${TEST_SOURCES})
    target_link_libraries(learned_index_tests learned_index_rocksdb ${GTEST_LIBRARIES} pthread)
    
    # Register tests
    add_test(NAME learned_index_unit_tests COMMAND learned_index_tests)
endif()

# Example executable
add_executable(learned_index_example examples/basic_usage.cpp)
target_link_libraries(learned_index_example learned_index_rocksdb)

# Installation
install(TARGETS learned_index_rocksdb
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY include/learned_index
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")