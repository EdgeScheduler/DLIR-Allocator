cmake_minimum_required(VERSION 3.0.0)

# set(COMPILE_MODE "DLIR_MODE")     # set target version

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

include_directories("include" "testlib" "libs")

SET(DLIR_Allocator_BIN "DLIR-Allocator")
set(OYST_MODE_BIN "OYST-Allocator")
set(BNST_MODE_BIN "BNST-Allocator")
set(FIFO_MODE_BIN "FIFO-Allocator")
set(PARALLER_MODE_BIN "PARALLER-Allocator")

if(COMPILE_MODE AND (COMPILE_MODE STREQUAL "OYST_MODE"))
    add_definitions(-DOYST_MODE)
    set(target ${OYST_MODE_BIN})
elseif(COMPILE_MODE AND (COMPILE_MODE STREQUAL "BNST_MODE"))
    add_definitions(-DBNST_MODE)
    set(target ${BNST_MODE_BIN})
elseif(COMPILE_MODE AND (COMPILE_MODE STREQUAL "FIFO_MODE"))
    add_definitions(-DFIFO_MODE)
    set(target ${FIFO_MODE_BIN})
elseif(COMPILE_MODE AND (COMPILE_MODE STREQUAL "PARALLER_MODE"))
    add_definitions(-DPARALLER_MODE)
    set(target ${PARALLER_MODE_BIN})
elseif(COMPILE_MODE AND (COMPILE_MODE STREQUAL "DLIR_MODE"))
    add_definitions(-DDLIR_MODE)
    set(target ${DLIR_Allocator_BIN})
else()
    set(target ${DLIR_Allocator_BIN})
endif()


add_subdirectory("sources/Common" "Libs/Common")
add_subdirectory("sources/GPUAllocator" "Libs/Allocator")
add_subdirectory("sources/Random" "Libs/Random")
add_subdirectory("sources/Tensor" "Libs/Tensor")
add_subdirectory("testlib" "Libs/TestLib")

aux_source_directory("entrance/Allocator" DLIR_SRCS)

add_executable(${target} ${DLIR_SRCS})
target_link_libraries(${target} test allocator tensor common random stdc++fs onnxruntime pthread)


