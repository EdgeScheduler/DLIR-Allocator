#include "GPUAllocator/TaskDigest.h"

TaskDigest::TaskDigest(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime, const int &taskCount, std::vector<const int *> &otherTaskCount, float penaltyValue) : executeTime(executeTime), requiredToken(requiredToken), requiredTokenCount(requiredTokenCount), limitRuntime(modelExecuteTime), leftRuntime(0.0F), name(name), childsRuntime(0.0F), childsCount(requiredTokenCount), taskCount(taskCount), otherTaskCount(otherTaskCount)
{
    this->startTime = clock();
    this->penaltyValue = penaltyValue;

    for (auto cost : *executeTime)
    {
        this->leftRuntime += cost;
    }

    this->childsRuntime = this->leftRuntime;
}

bool TaskDigest::SuggestRunSegmentation()
{
    for(auto iter: this->otherTaskCount)
    {
        if(*iter<1)
        {
            return true;
        }
    }

    return false;
    //return this->taskCount<2;
}

float TaskDigest::GetSLO()
{
    return this->limitRuntime * 10;
}

// y=a*x^2 + b*x +c, y(limit-time)=1, y(slo)=0
// using l=limitTime; using s=slotime;
// y= (1/(s-l)^2) * (-x^2 + 2*l*x + s^2-2*l*s)
float TaskDigest::Evaluate(float waitTime)
{
    if (this->requiredTokenCount <= 0)
    {
        return 1.0F;
    }

    waitTime += (clock() - startTime) / CLOCKS_PER_SEC * 1000.0F + leftRuntime;

    // float slo = GetSLO();
    // float value = (-waitTime * waitTime + 2 * limitRuntime * waitTime + slo * slo - 2 * limitRuntime * slo) / (slo - limitRuntime) / (slo - limitRuntime);

    // if (waitTime > slo)
    // {
    //     value = 1.0 / (value + penaltyValue);
    // }

    return waitTime/limitRuntime;
}

int TaskDigest::GetToken(float &reduceTime, bool &enableSegmentation)
{
    if (this->requiredTokenCount < 1)
    {
        reduceTime = 0.0F;
        return 0;
    }
    else
    {
        if (enableSegmentation || this->requiredTokenCount < this->childsCount)
        {
            enableSegmentation = true;
            reduceTime = (*executeTime)[requiredTokenCount - 1];
            this->leftRuntime -= reduceTime;
            this->requiredTokenCount -= 1;
            if (this->requiredTokenCount < 1)
            {
                this->leftRuntime = 0.0F;
            }
        }
        else
        {
            // disable segment
            reduceTime = this->childsRuntime;
            this->leftRuntime = 0;
            this->requiredTokenCount = 0;
        }

        return this->requiredToken;
    }
}

float TaskDigest::LeftRunTime()
{
    return this->leftRuntime;
}

std::string TaskDigest::GetInfo(int offset)
{
    return name + "-" + std::to_string(requiredTokenCount + offset);
}
