#ifndef __EXECUTORMANAGER_H__
#define __EXECUTORMANAGER_H__

#include <onnxruntime_cxx_api.h>
#include <mutex>
#include <string>
#include <thread>
#include <memory>
#include <map>
#include <vector>
#include <condition_variable>
#include "TokenManager.h"
#include "ModelExecutor.h"
#include "TaskRegistration.h"
#include "ThreadSafe/SafeQueue.hpp"

struct ExecutorDescribe
{
    std::shared_ptr<ModelExecutor> executor;
    int executorID;
    std::string modelName;
    std::shared_ptr<std::thread> threadHandle;
    std::shared_ptr<std::thread> resultGatherThread;
};

/// @brief 
class ExecutorManager
{
public:
    ExecutorManager();
    ~ExecutorManager();

    /// @brief get all executor threads
    /// @return
    std::vector<std::shared_ptr<std::thread>> GetAllThreads();

    /// @brief start an executor
    /// @param model_name
    void RunExecutor(std::string model_name);

    /// @brief add task to executor
    /// @param model_name executor/model name
    /// @param datas model inputs data
    void AddTask(std::string model_name, std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> datas, std::string tag = "");

    /// @brief get executor describes
    /// @return
    std::map<std::string, std::shared_ptr<ExecutorDescribe>> &GetExecutorInformation();

    /// @brief give token to xx
    /// @param token ID, 0 means free
    /// @param block if token is still there, block or not.
    /// @return
    bool Grant(int token, bool block = true);

    /// @brief join all thread to current-thread
    void Join();

    /// @brief gather tasks from queue
    void GatherTask(SafeQueue<std::shared_ptr<Task>>* taskQueue);

    /// @brief returns a read/write reference Queue to the apply of all the request that have been finished.
    SafeQueue<std::shared_ptr<Task>>& GetApplyQueue();

    /// @brief close executor-manager safely
    void Close();

private:
    Ort::SessionOptions sessionOption;
    Ort::Env environment;
    std::condition_variable dealTask; // must define before taskRegistration
    TokenManager tokenManager;        // must define before taskRegistration
    int executorCount;
    std::mutex gpuMutex;
    std::mutex taskMutex;

    TaskRegistration taskRegistration; // must define after tokenManager and dealTask

    std::shared_ptr<std::thread> tokenDespenseThread;
    std::map<std::string, std::shared_ptr<ExecutorDescribe>> executorMap;

    SafeQueue<std::shared_ptr<Task>> applyQueue;
};

#endif // __EXECUTORMANAGER_H__