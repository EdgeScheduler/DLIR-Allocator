#ifndef __GRPCINTERFACE_H__
#define __GRPCINTERFACE_H__

#include <iostream>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include <atomic>
#include <mutex>

// grpc 头文件
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <GPUAllocator/ExecutorManager.h>
#include "ThreadSafe/SafeQueue.hpp"
#include "GPUAllocator/Task.h"
#include "Tensor/ModelTensorsInfo.h"
#include "RPCProtoInterface/rpcinterface.pb.h"
#include "RPCProtoInterface/rpcinterface.grpc.pb.h"

class GRPCInterface : public RPCInterface::DLIRService::Service
{
public:
    GRPCInterface(ExecutorManager *executorManager,std::map<std::string, ModelInfo>* modelInfos,int port);

    /// @brief deal with gRPC request for `DoInference`
    /// @param context 
    /// @param request request info give by client
    /// @param response response to reply to client
    /// @return gRPC status
    virtual grpc::Status DoInference(::grpc::ServerContext *context, const RPCInterface::RequestInference *request, RPCInterface::ReplyInference *response) override;

    /// @brief deal with gRPC request for `DoInference`
    /// @param context 
    /// @param request request info give by client
    /// @param response response to reply to client
    /// @return gRPC status
    virtual ::grpc::Status GetIOShape(::grpc::ServerContext* context, const ::RPCInterface::RequestIOShape* request, ::RPCInterface::ReplyIOShape* response) override;

    /// @brief deal with gRPC request for `GetService`
    /// @param context 
    /// @param request request info give by client
    /// @param response response to reply to client
    /// @return gRPC status 
    virtual grpc::Status GetService(::grpc::ServerContext *context, const RPCInterface::RequestInfo *request, RPCInterface::ReplyInfo *response) override;

public:
    static std::map<std::string, std::shared_ptr<SafeQueue<std::shared_ptr<Task>>>> waitTasks;

private:
    ExecutorManager *executorManager;
    std::map<std::string, ModelInfo>* modelInfos;
    std::vector<std::string> modelNames;
    static std::atomic<int> tag;
    static std::mutex tagMutex;
    static std::mutex taskMapMutex;
    int serverPort;
};

#endif // __GRPCINTERFACE_H__