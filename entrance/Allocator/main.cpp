#include <iostream>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "../../include/GPUAllocator/ExecutorManager.h"
#include "../../include/Tensor/TensorValue.hpp"
#include "../../include/Common/JsonSerializer.h"
#include "../../include/Tensor/ModelInputCreator.h"
#include "../../include/Common/PathManager.h"
#include "../../include/Random/UniformRandom.h"
#include "../../include/Random/PossionRandom.h"

using DatasType = std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>>;

// g++ -DALLOW_GPU_PARALLEL for parallel
void ReqestGenerate(ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, float lambda = 30);
void ReplyGather(ExecutorManager *executorManager, int count);

int main(int argc, char *argv[])
{
    int dataCount = 1000;
    float lambda = 60;
    if (argc >= 2)
    {
        dataCount = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        lambda = atoi(argv[2]);
    }

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

void AddRequestInThread(std::mutex *mutex, std::condition_variable *condition, int index, ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, float lambda, int *current_count)
{
    static std::atomic<int> flag=0;
    PossionRandom possionRandom(index * 100+100);
    UniformRandom uniformRandom(index * 10+10);
    std::pair<std::string, ModelInputCreator> *creator = &(*inputCreators)[index];
    std::vector<std::pair<std::string, DatasType>> datas(20);
    for (int i = 0; i < 20; i++)
    {
        datas[i] = std::pair<std::string, DatasType>(creator->first, creator->second.CreateInput());
    }

    flag++;

    while(flag<inputCreators->size())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::cout<<"start to send request for "<<creator->first<<"."<<std::endl;

    int i = 0;
    int tmp=0;
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds((int)possionRandom.Random(lambda)));
        std::unique_lock<std::mutex> lock(*mutex);
        if (*current_count >= count)
        {
            // to be end.
            break;
        }
        tmp = *current_count;
        (*current_count)++;
        lock.unlock();
        condition->notify_all();
            executorManager->AddTask(datas[i%20].first, datas[i%20].second, std::to_string(tmp+1));
        i++;
    }
    std::cout<<"end send "<<creator->first<<", total: "<<i<<std::endl;
}

void ReqestGenerate(ExecutorManager *executorManager, std::vector<std::pair<std::string, ModelInputCreator>> *inputCreators, int count, float lambda)
{
    std::cout << "wait system to init..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));

    std::cout << "run request generate with possion, and start to prepare random input." << std::endl;
    
    std::condition_variable condition;
    std::vector<std::shared_ptr<std::thread>> threads;
    int current_count=0;
    std::mutex mutex;
    for(int i=0;i<inputCreators->size();i++)
    {
        threads.push_back(std::make_shared<std::thread>(AddRequestInThread,&mutex,&condition,i,executorManager,inputCreators,count,lambda,&current_count));
    }

    for(auto thread: threads)
    {
        thread->join();
    }

    std::cout << "end send request." << std::endl;
}

void ReplyGather(ExecutorManager *executorManager, int count)
{
    std::cout << "run reply gather." << std::endl;
#ifdef ALLOW_GPU_PARALLEL
    auto saveFold = RootPathManager::GetRunRootFold() / "data" / "raw";
#else
    std::filesystem::path saveFold = RootPathManager::GetRunRootFold() / "data" / "allocator";
#endif

    std::filesystem::remove_all(saveFold);
    std::filesystem::create_directories(saveFold);
    auto &applyQueue = executorManager->GetApplyQueue();
    for (int i = 0; i < count; i++)
    {
        auto task = applyQueue.Pop();
        JsonSerializer::StoreJson(task->GetDescribe(), saveFold / (std::to_string(i) + ".json"));
    }
    std::cout << "all apply for " << count << " task received." << std::endl;
}
