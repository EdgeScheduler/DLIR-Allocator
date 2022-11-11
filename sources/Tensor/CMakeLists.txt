include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. TENSOR_SRCS)

add_library(tensor ${TENSOR_SRCS})
target_link_libraries(tensor stdc++fs onnxruntime pthread)