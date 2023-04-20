#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <string>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "Tensor/TensorValue.hpp"
#include "Common/JsonSerializer.h"
#include "Tensor/ModelTensorsInfo.h"
#include "Common/PathManager.h"
#include "Random/UniformRandom.h"
#include "Random/PossionRandom.h"
#include "RPCResponse/RPCServer.h"
#include "GPUAllocator/ModeCheck.h"
#include "GPUAllocator/ExecutorManager.h"
#include "cmdline.h"
#include "testlib/TaskGenerate.h"

// using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

static auto models = {"vgg19", "resnet50", "googlenet", "gpt128v2", "yolov2"};

// ./bin/release/DLIR-Allocator && ./bin/release/BNST-Allocator && ./bin/release/FIFO-Allocator && ./bin/release/PARALLER-Allocator && ./bin/release/OYST-Allocator
int main(int argc, char *argv[])
{
    std::string mode_name;
    if (ModelCheck(mode_name))
    {
        std::cout << "Run mode:" << mode_name << std::endl;
    }
    else
    {
        std::cout << "error mode, please compile your binary with correct args." << std::endl;
        return 0;
    }

    cmdline::parser parser;
    parser.set_program_name("Allocator-" + mode_name);
    parser.add<int>("port", 'p', "Port to expose gRPC service [>200]", false, 85001);

    // parse args
    parser.parse_check(argc, argv);
    int port = parser.get<int>("port") > 200 ? parser.get<int>("port") : 85001;

    std::vector<std::string> model_names;
    std::vector<int> lambdas;
    for (auto model_name : models)
    {
        model_names.push_back(model_name);
        lambdas.push_back(0);
    }

    ExecutorManager executorManager;
    std::map<std::string, ModelInfo> modelInfos;
    for (auto model_name : models)
    {
        nlohmann::json json = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath(model_name));
        executorManager.RunExecutor(model_name);
        modelInfos.insert(std::pair<std::string, ModelInfo>(model_name, ModelInfo(json)));
    }

    // executorManager.RunExecutorReTest();

    std::thread reqestGenerateThread(RunRPCServer, &executorManager,&modelInfos,port);
    ReplyGather(&executorManager, -1, lambdas, model_names);

    executorManager.Close();

    reqestGenerateThread.join();
    executorManager.Join();

    std::cout << "process terminate." << std::endl;
    return 0;
}