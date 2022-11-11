include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. TENSOR_SRCS)

add_library(tensor ${TENSOR_SRCS})