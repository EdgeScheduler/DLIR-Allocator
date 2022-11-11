cmake_minimum_required(VERSION 3.0.0)

set(PROJECT_NAME "DLIR_ALLOCATOR")
project(${PROJECT_NAME})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set binary path
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    SET(EXECUTABLE_OUTPUT_PATH "${DLIR_ALLOCATOR_SOURCE_DIR}/bin/debug")
elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    SET(EXECUTABLE_OUTPUT_PATH "${DLIR_ALLOCATOR_SOURCE_DIR}/bin/release")
endif()

include_directories("include")

SET(DLIR_Allocator_BIN "DLIR-Allocator")
set(OYST_MODE_BIN "OYST-Allocator")
set(BNST_MODE_BIN "BNST-Allocator")
set(FIFO_MODE_BIN "FIFO-Allocator")
set(PARALLER_MODE_BIN "PARALLER-Allocator")

add_subdirectory("sources/Common" "Libs/Common")
add_subdirectory("sources/GPUAllocator" "Libs/Allocator")
add_subdirectory("sources/Random" "Libs/Random")
add_subdirectory("sources/Tensor" "Libs/Tensor")
add_subdirectory("entrance/test" "Libs/test")

aux_source_directory("entrance/Allocator" DLIR_SRCS)

link_libraries(
    stdc++fs
    onnxruntime
    pthread
)

add_executable(${DLIR_Allocator_BIN} ${DLIR_SRCS})
target_link_libraries(${DLIR_Allocator_BIN} test allocator tensor common random)