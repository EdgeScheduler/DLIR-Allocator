#ifndef __MODELEXECUTOR_H__
#define __MODELEXECUTOR_H__

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <mutex>
#include <memory>
#include <condition_variable>
#include <map>
#include "../ThreadSafe/SafeQueue.hpp"
#include "../Tensor/ModelTensorsInfo.h"
#include "TokenManager.h"
#include "Task.h"

class ModelExecutor
{
public:
    ModelExecutor(std::string model_name, Ort::SessionOptions *session_opt, Ort::Env *env, int token_id, TokenManager *token_manager, std::mutex *gpu_mutex, std::condition_variable *deal_task);

    /// @brief add new task to executor by share_ptr
    /// @param datas model inputs
    /// @param tag add some information to task
    void AddTask(std::shared_ptr<std::map<std::string, std::shared_ptr<TensorValue<float>>>> datas, std::string tag = "");

    /// @brief record current task to the end.
    void ToNext();

    /// @brief Load task args, if not exist, it will come to block.
    void LoadTask();

    /// @brief Inference for once task.
    /// @return recv end signal or not.
    bool RunOnce();

    /// @brief run all model automatically.
    void RunCycle();

    /// @brief get token id of this executor.
    /// @return
    int GetTokenID();

    /// @brief get how long the model limit-timecost.
    /// @return
    float &GetModelExecuteTime();

    /// @brief get how many pieces this model is split into.
    /// @return
    int GetChildModelCount();

    SafeQueue<std::shared_ptr<Task>> &GetResultQueue();
    SafeQueue<std::shared_ptr<Task>> &GetTaskQueue();
    std::shared_ptr<std::vector<float>> GetExecuteTime();

private:
    int modelCount = 0;
    Ort::Env *onnxruntimeEnv;
    std::shared_ptr<Ort::Session> rawSession;
    Ort::SessionOptions *sessionOption;
    std::vector<Ort::Session> sessions;
    std::vector<ModelInfo> modelInfos;
    std::vector<std::vector<const char *>> inputLabels;
    std::vector<std::vector<const char *>> outputLabels;
    std::shared_ptr<ModelInfo> rawModelInfo;

    SafeQueue<std::shared_ptr<Task>> task_queue;
    SafeQueue<std::shared_ptr<Task>> finish_queue;

    int todo;
    std::string modelName;

    int tokenID;
    TokenManager *tokenManager;

    std::mutex *gpuMutex;
    std::condition_variable *dealTask;
    // runtime args
private:
    std::shared_ptr<Task> current_task;

    // record how long will models run cost.
    std::shared_ptr<std::vector<float>> executeTime;
    float modelExecuteTime;
};

#endif // __MODELEXECUTOR_H__