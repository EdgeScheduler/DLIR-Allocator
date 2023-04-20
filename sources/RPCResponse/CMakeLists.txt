include_directories("${DLIR_ALLOCATOR_SOURCE_DIR}/include")
aux_source_directory(. RPCRESONSE_SRCS)

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
add_library(rpcresponse ${RPCRESONSE_SRCS})
target_link_libraries(rpcresponse
  protolib
  gRPC::grpc++_reflection
  grpc++
  protobuf::libprotobuf)