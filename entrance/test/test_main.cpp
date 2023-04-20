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
#include "Tensor/ModelInputCreator.h"
#include "Common/PathManager.h"
#include "Random/UniformRandom.h"
#include "Random/PossionRandom.h"
#include "TaskGenerate.h"
#include "GPUAllocator/ModeCheck.h"
#include "GPUAllocator/ExecutorManager.h"
#include "cmdline.h"

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

    // ready to parse args
    cmdline::parser parser;
    parser.set_program_name("Allocator-" + mode_name);
    parser.add<int>("count", 'c', "how many requests total to send [>0]", false, 1000);
    parser.add<int>("lambda", 'l', "default λ(ms) [>0]", false, 200);
    for (auto &model_name : models)
    {
        parser.add<int>(model_name, '\0', std::string("λ(ms) for ") + model_name + std::string("[>=0], and you can set default value by --lambda"), false);
    }

    // parse args
    parser.parse_check(argc, argv);

    int dataCount = parser.get<int>("count") > 0 ? parser.get<int>("count") : 1000;
    float lambda = parser.get<int>("lambda") > 0 ? parser.get<int>("lambda") : 200;
    std::cout << "cout=" << dataCount << ", default λ=" << lambda << "ms" << std::endl;

    std::vector<int> lambdas;
    std::vector<std::string> model_names;

    for (auto model_name : models)
    {
        int tmp = lambda;
        if (parser.exist(model_name))
        {
            tmp = parser.get<int>(model_name) >= 0 ? parser.get<int>(model_name) : lambda;
        }
        std::cout << "λ-" << model_name << "=" << tmp << "ms" << std::endl;

        model_names.push_back(model_name);
        lambdas.push_back(tmp);
    }

    // check if data exist
    if (CheckReady(dataCount, lambdas, model_names))
    {
        std::cout << "data already exist, skip run." << std::endl;
        return 0;
    }

    ExecutorManager executorManager;
    std::vector<std::pair<std::string, ModelInputCreator>> inputCreators;
    for (auto model_name : models)
    {
        nlohmann::json json = JsonSerializer::LoadJson(OnnxPathManager::GetModelParamsSavePath(model_name));
        ModelInfo info(json);
        ModelInputCreator creator(info.GetInput());
        executorManager.RunExecutor(model_name);
        inputCreators.push_back(std::pair<std::string, ModelInputCreator>(model_name, creator));
    }

   //  executorManager.RunExecutorReTest();

    std::thread reqestGenerateThread(ReqestGenerate, &executorManager, &inputCreators, dataCount, lambdas);
    ReplyGather(&executorManager, dataCount, lambdas, model_names);

    executorManager.Close();

    reqestGenerateThread.join();
    executorManager.Join();

    std::cout << "end ok." << std::endl;
    return 0;
}