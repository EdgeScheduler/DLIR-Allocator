include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. RANDOM_SRCS)

add_library(random ${RANDOM_SRCS})