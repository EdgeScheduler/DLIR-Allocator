#include <string>
#include <map>
#include <iostream>
#include <vector>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <GPUAllocator/ExecutorManager.h>
// grpc 头文件
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include "Tensor/TensorValue.hpp"
#include "Tensor/ValueInfo.h"
#include "RPCResponse/GRPCInterface.h"
#include "RPCResponse/IPs.h"
#include "ThreadSafe/SafeQueue.hpp"
#include "GPUAllocator/Task.h"

GRPCInterface::GRPCInterface(ExecutorManager *executorManager, std::map<std::string, ModelInfo> *modelInfos, int port) : RPCInterface::DLIRService::Service(), executorManager(executorManager), serverPort(port), modelInfos(modelInfos)
{
    for (auto executorInfo : executorManager->GetExecutorInformation())
    {
        this->modelNames.push_back(executorInfo.first);
    }
}

grpc::Status GRPCInterface::DoInference(grpc::ServerContext *context, const RPCInterface::RequestInference *request, RPCInterface::ReplyInference *response)
{
    try
    {
        auto iter = this->modelInfos->find(request->modelname());
        if (iter == this->modelInfos->end())
        {
            return grpc::Status(grpc::StatusCode::UNKNOWN, "The model you invoked has not been deployed.");
        }
        const TensorsInfo &tensorsInfo = iter->second.GetInput();

        std::unique_lock<std::mutex> lock(GRPCInterface::tagMutex);
        int current_tag = GRPCInterface::tag;
        GRPCInterface::tag = GRPCInterface::tag + 1; // The data may overflow, but the range is large enough to ensure that the tag is unique.
        lock.unlock();
        std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValueObject>>> inputs = std::make_shared<std::map<std::string, std::shared_ptr<TensorValueObject>>>();

        nlohmann::json input_data = nlohmann::json::parse(request->data());
        // for(auto iter: input_data.items())
        // {
        //     std::cout<<iter.key()<<std::endl;
        // }

        for (const ValueInfo info : tensorsInfo.GetAllTensors())
        {
            if (!input_data.contains(info.GetName()))
            {
                return grpc::Status(grpc::StatusCode::UNKNOWN, "input datas broken, loss key: \"" + info.GetName() + "\"");
            }

            bool success = false;
            std::shared_ptr<TensorValueObject> ptr;
            switch (info.GetType())
            {
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                ptr = std::make_shared<TensorValue<int8_t>>(info, input_data[info.GetName()].get<std::vector<int8_t>>(), success);
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                ptr = std::make_shared<TensorValue<int16_t>>(info, input_data[info.GetName()].get<std::vector<int16_t>>(), success);
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                ptr = std::make_shared<TensorValue<int32_t>>(info, input_data[info.GetName()].get<std::vector<int32_t>>(), success);
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                ptr = std::make_shared<TensorValue<int64_t>>(info, input_data[info.GetName()].get<std::vector<int64_t>>(), success);
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                ptr = std::make_shared<TensorValue<float>>(info, input_data[info.GetName()].get<std::vector<float>>(), success);
                break;
            default:
                ptr = std::make_shared<TensorValue<float>>(info, input_data[info.GetName()].get<std::vector<float>>(), success);
            }

            if (!success)
            {
                return grpc::Status(grpc::StatusCode::UNKNOWN, "input datas broken, loss data");
            }
            inputs->insert(std::make_pair(info.GetName(), ptr));
        }

        // wait for request
        std::unique_lock<std::mutex> mapAddLock(GRPCInterface::taskMapMutex);
        // GRPCInterface::waitTasks.insert(std::pair<std::string, SafeQueue<std::shared_ptr<Task>>>(std::to_string(current_tag),SafeQueue<std::shared_ptr<Task>>()));
        // this->waitTasks->insert(std::pair<std::string, SafeQueue<std::shared_ptr<Task>>>(std::to_string(current_tag),SafeQueue<std::shared_ptr<Task>>(1)));
        GRPCInterface::waitTasks.insert(std::make_pair(std::to_string(current_tag), std::make_shared<SafeQueue<std::shared_ptr<Task>>>(1)));
        mapAddLock.unlock();

        this->executorManager->AddTask(request->modelname(), inputs, std::to_string(current_tag));

        auto task = GRPCInterface::waitTasks[std::to_string(current_tag)]->Pop();
        std::unique_lock<std::mutex> mapRemoveLock(GRPCInterface::taskMapMutex);
        GRPCInterface::waitTasks.erase(std::to_string(current_tag));
        mapRemoveLock.unlock();

        // // deal with task, unfinished
        // task->PrintOutputs();

        nlohmann::json result;
        for (auto output : task->GetOutputs())
        {
            switch (output->GetValueInfo().GetType())
            {
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<int8_t>>(output)->GetData();
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<int16_t>>(output)->GetData();
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<int32_t>>(output)->GetData();
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<int64_t>>(output)->GetData();
                break;
            case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<float>>(output)->GetData();
                break;
            default:
                result[output->GetValueInfo().GetName()] = std::dynamic_pointer_cast<TensorValue<float>>(output)->GetData();
            }
        }

        response->set_info("ok");
        response->set_status(0);
        response->set_result(result.dump());

        return ::grpc::Status::OK;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error happened while doing inference: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::UNKNOWN, "some error happened while deal with your request.");
    }
}

grpc::Status GRPCInterface::GetIOShape(::grpc::ServerContext *context, const ::RPCInterface::RequestIOShape *request, ::RPCInterface::ReplyIOShape *response)
{
    try
    {
        auto iter = this->modelInfos->find(request->modelname());
        if (iter == this->modelInfos->end())
        {
            return grpc::Status(grpc::StatusCode::UNKNOWN, "The model you invoked has not been deployed.");
        }
        response->set_inputs(iter->second.GetInput().ToJson().dump());
        response->set_outputs(iter->second.GetOutput().ToJson().dump());
        return ::grpc::Status::OK;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error happened while get io shapes: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::UNKNOWN, "some error happened while deal with your request.");
    }
    
}

// 请求推理
grpc::Status GRPCInterface::GetService(grpc::ServerContext *context, const RPCInterface::RequestInfo *request, RPCInterface::ReplyInfo *response)
{
    try
    {
        response->set_ip(GetLocalIPs());
        response->set_port(this->serverPort);
        for (auto name : this->modelNames)
        {
            response->add_modelnames(name);
        }
        return grpc::Status::OK;
    }
    catch (const std::exception &e)
    {
        std::cerr << "error happened while get services describe: " << e.what() << '\n';
        return grpc::Status(grpc::StatusCode::UNKNOWN, "some error happened while deal with your request.");
    }
}

std::map<std::string, std::shared_ptr<SafeQueue<std::shared_ptr<Task>>>> GRPCInterface::waitTasks;
std::atomic<int> GRPCInterface::tag = 0;
std::mutex GRPCInterface::tagMutex;
std::mutex GRPCInterface::taskMapMutex;