cmake_minimum_required(VERSION 3.16)
project(learned_index_rocksdb 
    VERSION 1.0.0 
    DESCRIPTION "Learned Index integration for RocksDB LSM trees"
    LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set compiler flags for optimization and debugging
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall")

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find required dependencies
find_package(Threads REQUIRED)

# Find crc32c library (for checksum validation)
find_library(CRC32C_LIB crc32c)
if(CRC32C_LIB)
    message(STATUS "Found crc32c library: ${CRC32C_LIB}")
    add_definitions(-DHAVE_CRC32C_LIB)
else()
    message(STATUS "crc32c library not found. Using built-in CRC32 fallback.")
    set(CRC32C_LIB "")
endif()

# Option to build tests
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" OFF)

# Create learned_index library
set(LEARNED_INDEX_SOURCES
    src/learned_index/learned_index_block.cpp
    src/learned_index/sst_learned_index_manager.cpp
)

set(LEARNED_INDEX_HEADERS
    include/learned_index/learned_index_block.h
    include/learned_index/sst_learned_index_manager.h
)

# Create static library
add_library(learned_index STATIC ${LEARNED_INDEX_SOURCES} ${LEARNED_INDEX_HEADERS})
target_include_directories(learned_index PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries conditionally
target_link_libraries(learned_index PRIVATE Threads::Threads)
if(CRC32C_LIB)
    target_link_libraries(learned_index PRIVATE ${CRC32C_LIB})
endif()

# Set target properties
set_target_properties(learned_index PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "${LEARNED_INDEX_HEADERS}"
)

# Add examples if enabled
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Add tests if enabled
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

# Add benchmarks if enabled
if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

# Installation
install(TARGETS learned_index
    EXPORT learned_index_targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/learned_index
)

# Install headers
install(DIRECTORY include/learned_index
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Create config files for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    learned_index_config_version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/learned_index_config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/learned_index_config.cmake"
    INSTALL_DESTINATION lib/cmake/learned_index
)

install(EXPORT learned_index_targets
    FILE learned_index_targets.cmake
    NAMESPACE learned_index::
    DESTINATION lib/cmake/learned_index
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/learned_index_config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/learned_index_config_version.cmake"
    DESTINATION lib/cmake/learned_index
)

# Print configuration summary
message(STATUS "")
message(STATUS "Learned Index RocksDB Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Build tests: ${BUILD_TESTS}")
message(STATUS "  Build examples: ${BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "  CRC32C library: ${CRC32C_LIB}")
message(STATUS "")