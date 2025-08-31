cmake_minimum_required(VERSION 3.12)
project(learned_index_rocksdb)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Create learned_index library
add_library(learned_index 
    src/learned_index/learned_index_block.cpp
    src/learned_index/sst_learned_index_manager.cpp
)

target_include_directories(learned_index PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link with crc32c for checksum calculation
# Note: In a real RocksDB integration, this would use RocksDB's existing CRC32 implementation
target_compile_definitions(learned_index PRIVATE 
    -DCRC32C_STATIC_DEFINE
)

# For testing - create a simple mock of crc32c if not available
if(NOT TARGET crc32c::crc32c)
    add_library(crc32c INTERFACE)
    target_include_directories(crc32c INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/mock)
endif()

target_link_libraries(learned_index PRIVATE crc32c)

# Enable testing
enable_testing()

# Add unit tests
add_executable(learned_index_test
    test/learned_index/learned_index_block_test.cpp
    test/learned_index/sst_learned_index_manager_test.cpp
    test/learned_index/test_main.cpp
)

target_link_libraries(learned_index_test learned_index)

target_include_directories(learned_index_test PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/test
)

add_test(NAME learned_index_unit_tests COMMAND learned_index_test)

# Compiler flags
target_compile_options(learned_index PRIVATE -Wall -Wextra -O2)
target_compile_options(learned_index_test PRIVATE -Wall -Wextra -g)