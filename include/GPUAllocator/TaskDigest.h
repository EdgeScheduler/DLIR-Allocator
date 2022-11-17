#ifndef __TASKDIGEST_H__
#define __TASKDIGEST_H__

#include <ctime>
#include <vector>
#include <string>
#include <memory>

class TaskDigest
{
public:
    /// @brief
    /// @param executeTime each child-model execute-time(ms).
    /// @param requiredToken
    /// @param requiredTokenCount how many child-models there is
    /// @param modelExecuteTime
    /// @param penaltyValue
    TaskDigest(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime, const int &taskCount, std::vector<const int *> &otherTaskCount, float penaltyValue = -1.0);

public:
    /// @brief calculate SLO-time(ms)
    /// @return
    float GetSLO();

    // /// @brief evaluate the score to the current wait-time, better choice face to higher score.
    // /// @param waitTime how long the task still need to in-queue. (ms)
    // /// @return score<=1.0. If score>1, it means this task had been finished before.
    // float Evaluate(float waitTime);

    /// @brief evaluate the score to the current wait-time, better choice face to lower score.
    /// @param waitTime how long the task still need to in-queue. (ms)
    /// @return score
    float Evaluate(float waitTime);

    float LeftRunTime();

    /// @brief Get new Token
    /// @param reduceTime used to return how long this task reduce
    /// @param enableSegmentation if false, will try to disable model-segment. set value to true if fail.
    /// @return 0 if there is no need.
    int GetToken(float &reduceTime, bool &enableSegmentation);

    /// @brief get sequence info.
    /// @return
    std::string GetInfo(int offset = 0);

    /// @brief calculate whether it is suggested to run segmentation
    /// @return true if ok, else false
    bool SuggestRunSegmentation();

public:
    int requiredToken;
    int requiredTokenCount;
    int childsCount;
    std::vector<const int *> &otherTaskCount;
    const int &taskCount;

private:
    std::string name;
    clock_t startTime;
    std::shared_ptr<std::vector<float>> executeTime;

    float leftRuntime;
    float &limitRuntime;
    float penaltyValue;
    float childsRuntime;
};

#endif // __TASKDIGEST_H__