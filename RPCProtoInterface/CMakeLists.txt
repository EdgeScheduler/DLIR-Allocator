include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../include" ".")

find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

# Proto file
set(PROTO_NAME rpcinterface)
get_filename_component(rpc_interface_proto "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.proto" ABSOLUTE)
get_filename_component(rpc_interface_proto_path "${rpc_interface_proto}" PATH)

set(rpc_interface_proto_srcs "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.pb.cc")
set(rpc_interface_proto_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.pb.h")
set(rpc_interface_grpc_srcs "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.grpc.pb.cc")
set(rpc_interface_grpc_hdrs "${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.grpc.pb.h")

message("${CMAKE_CURRENT_SOURCE_DIR}/")
add_custom_command(
  OUTPUT "${rpc_interface_proto_srcs}" "${rpc_interface_proto_hdrs}" "${rpc_interface_grpc_srcs}" "${rpc_interface_grpc_hdrs}"
  COMMAND protoc
  ARGS --grpc_out "${CMAKE_CURRENT_SOURCE_DIR}/"
    --cpp_out "${CMAKE_CURRENT_SOURCE_DIR}/"
    -I "${CMAKE_CURRENT_SOURCE_DIR}/"
    --plugin=protoc-gen-grpc=`$(which grpc_cpp_plugin)`
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROTO_NAME}.proto
  DEPENDS "${rpc_interface_proto}")

add_library(protolib 
  ${rpc_interface_proto_srcs}
  ${rpc_interface_proto_hdrs}
  ${rpc_interface_grpc_srcs}
  ${rpc_interface_grpc_hdrs}
)
target_link_libraries(protolib
  gRPC::grpc++_reflection
  grpc++
  protobuf::libprotobuf)