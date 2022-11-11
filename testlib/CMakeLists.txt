include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. TEST_SRCS)

add_library(test ${TEST_SRCS})