#include "GPUAllocator/ExecutorManager.h"
#include "Common/Drivers.h"
#include <iostream>

ExecutorManager::ExecutorManager() : environment(ORT_LOGGING_LEVEL_WARNING, "test"), executorCount(0), tokenManager(), taskRegistration(&tokenManager, &dealTask)
{
    sessionOption.SetIntraOpNumThreads(1);
    sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    sessionOption.AppendExecutionProvider_CUDA(Drivers::GPU_CUDA::GPU0);

    this->tokenDespenseThread = std::make_shared<std::thread>(&TaskRegistration::TokenDispense, &taskRegistration);
}

ExecutorManager::~ExecutorManager()
{
    Ort::OrtRelease(sessionOption.release());
    Ort::OrtRelease(environment.release());
}

void ExecutorManager::Close()
{
    this->taskRegistration.CloseRegistration();

    for(auto& iter: this->executorMap)
    {
        iter.second->executor->CloseExecutor();
    }
}

void ExecutorManager::RunExecutor(std::string model_name)
{
    this->executorCount++; // it means whether threads create successfully or not, give a token_id.
    std::shared_ptr<ExecutorDescribe> executorDescribe = std::make_shared<ExecutorDescribe>();
    executorDescribe->executor = std::make_shared<ModelExecutor>(model_name, &sessionOption, &environment, executorCount, &tokenManager, &gpuMutex, &dealTask);
    executorDescribe->executorID = executorCount;
    executorDescribe->modelName = model_name;
    executorDescribe->threadHandle = std::make_shared<std::thread>(&ModelExecutor::RunCycle, executorDescribe->executor);
    executorDescribe->resultGatherThread=std::make_shared<std::thread>(&ExecutorManager::GatherTask,this,&(executorDescribe->executor->GetResultQueue()));
    this->executorMap.insert(std::pair<std::string, std::shared_ptr<ExecutorDescribe>>(model_name, executorDescribe));
}

void ExecutorManager::GatherTask(SafeQueue<std::shared_ptr<Task>>* taskQueue)
{
    try
    {
        while(true)
        {
            this->applyQueue.Emplace(taskQueue->Pop());
        }
    }
    catch (DILException ex)
    {
        if (ex == DILException::SYSTEM_CLOSE)
        {
            return;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

void ExecutorManager::AddTask(std::string model_name, std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> datas, std::string tag)
{
    try
    {
        auto iter = this->executorMap.find(model_name);
        if (iter == this->executorMap.end())
        {
            std::cout << "no such model-executor run here." << std::endl;
            return;
        }

        // add lock to ensure task in executor and 
        // std::unique_lock<std::mutex> lock(taskMutex);
        this->taskRegistration.RegisteTask(model_name, iter->second->executor->GetExecuteTime(), iter->second->executor->GetTokenID(), iter->second->executor->GetChildModelCount(), iter->second->executor->GetModelExecuteTime(),iter->second->executor->GetTaskQueue().Size());
        iter->second->executor->AddTask(datas, tag);
        // lock.unlock();
    }
    catch (DILException ex)
    {
        if (ex == DILException::SYSTEM_CLOSE)
        {
            return;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

std::map<std::string, std::shared_ptr<ExecutorDescribe>> &ExecutorManager::GetExecutorInformation()
{
    return this->executorMap;
}

std::vector<std::shared_ptr<std::thread>> ExecutorManager::GetAllThreads()
{
    std::vector<std::shared_ptr<std::thread>> result;
    for (auto iter = executorMap.begin(); iter != executorMap.end(); iter++)
    {
        result.push_back(iter->second->threadHandle);
        result.push_back(iter->second->resultGatherThread);
    }
    result.push_back(this->tokenDespenseThread);
    return result;
}

void ExecutorManager::Join()
{
    std::vector<std::shared_ptr<std::thread>> threads = this->GetAllThreads();
    for (auto &thread : threads)
    {
        thread->join();
    }
}

bool ExecutorManager::Grant(int token, bool block)
{
    try
    {
        bool flag = this->tokenManager.Grant(token, block);
        this->dealTask.notify_all();
        return flag;
    }
    catch (DILException ex)
    {
        if (ex == DILException::SYSTEM_CLOSE)
        {
            return true;
        }
        else
        {
            std::cout << ex << std::endl;
            return false;
        }
    }
}

SafeQueue<std::shared_ptr<Task>>& ExecutorManager::GetApplyQueue()
{
    return this->applyQueue;
}
