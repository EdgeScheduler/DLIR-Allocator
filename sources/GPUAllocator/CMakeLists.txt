include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. ALLOCATOR_SRCS)

add_library(allocator ${ALLOCATOR_SRCS})