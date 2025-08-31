cmake_minimum_required(VERSION 3.16)

# Example programs
set(EXAMPLE_SOURCES
    basic_usage.cpp
)

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    
    target_link_libraries(${EXAMPLE_NAME}
        learned_index
        ${CRC32C_LIB}
        Threads::Threads
    )
    
    target_include_directories(${EXAMPLE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
    )
endforeach()