#ifndef __TASKREGISTRATION_H__
#define __TASKREGISTRATION_H__

#include <list>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "TaskDigest.h"
#include "TokenManager.h"

class TaskRegistration
{
public:
    TaskRegistration(TokenManager* tokenManager,std::condition_variable* dealTask);

    /// @brief register a new task to registration.
    /// @param executeTime
    /// @param requiredToken
    /// @param requiredTokenCount
    /// @param modelExecuteTime
    /// @param taskCount
    void RegisteTask(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime, const int& taskCount);

    /// @brief dispense tokens to model-executors, need to run with sync.
    void TokenDispense();
    void CloseRegistration();

private:
    /// @brief the end of the tasks is next task to be deal.
    std::list<TaskDigest> tasks;
    float queueLength; // how long the queue last.(ms)
    TokenManager *tokenManager;
    float reduceTime;

    std::mutex mutex;
    /// @brief add restrictions on consumer-threads
    std::condition_variable m_notEmpty;
    std::condition_variable* dealTask;

    bool closeRegistration;

    std::atomic<TaskDigest*> currentTask;
};

#endif // __TASKREGISTRATION_H__