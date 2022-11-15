#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "Tensor/TensorValue.hpp"
#include "Common/JsonSerializer.h"
#include "Tensor/ModelInputCreator.h"
#include "Common/PathManager.h"
#include "Random/UniformRandom.h"
#include "Random/PossionRandom.h"
#include "TaskGenerate.h"
#include "GPUAllocator/ModeCheck.h"
#include "GPUAllocator/ExecutorManager.h"

using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

// ./bin/release/DLIR-Allocator && ./bin/release/BNST-Allocator && ./bin/release/FIFO-Allocator && ./bin/release/PARALLER-Allocator && ./bin/release/OYST-Allocator
int main(int argc, char *argv[])
{
    std::string mode_name;
    if(ModelCheck(mode_name))
    {
        std::cout<<"Run mode:"<<mode_name<<std::endl;
    }
    else
    {
        std::cout<<"error mode, please compile your binary with correct args."<<std::endl;
        return 0;
    }
    int dataCount = 1000;
    float lambda = 200;
    if (argc >= 2)
    {
        dataCount = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        lambda = atoi(argv[2]);
    }
    std::cout<<"cout="<<dataCount<<", λ="<<lambda<<"ms"<<std::endl;

    ExecutorManager executorManager;
    std::vector<std::pair<std::string, ModelInputCreator>> inputCreators;

    for (auto model_name : {"vgg19", "resnet50", "googlenet", "squeezenetv1"})
    {
        nlohmann::json json = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath(model_name));
        ModelInfo info(json);
        ModelInputCreator creator(info.GetInput());
        executorManager.RunExecutor(model_name);
        inputCreators.push_back(std::pair<std::string, ModelInputCreator>(model_name, creator));
    }

    std::thread reqestGenerateThread(ReqestGenerate, &executorManager, &inputCreators, dataCount, lambda);
    ReplyGather(&executorManager, dataCount);

    executorManager.Close();
    
    reqestGenerateThread.join();
    executorManager.Join();

    std::cout<<"end ok."<<std::endl;
    return 0;
}