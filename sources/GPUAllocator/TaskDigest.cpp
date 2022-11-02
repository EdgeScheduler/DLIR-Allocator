#include "../../include/GPUAllocator/TaskDigest.h"

TaskDigest::TaskDigest(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime, float penaltyValue) : executeTime(executeTime), requiredToken(requiredToken), requiredTokenCount(requiredTokenCount), limitRuntime(modelExecuteTime), leftRuntime(0.0F), name(name)
{
    this->startTime = clock();
    this->penaltyValue = penaltyValue;

    for(auto cost: *executeTime)
    {
        this->leftRuntime+=cost;
    }
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
        return 2.0F;
    }

    waitTime += (clock() - startTime) / CLOCKS_PER_SEC * 1000.0F + leftRuntime;

    float slo = GetSLO();
    float value = (-waitTime * waitTime + 2 * limitRuntime * waitTime + slo * slo - 2 * limitRuntime * slo) / (slo - limitRuntime) / (slo - limitRuntime);

    if (waitTime > slo)
    {
        value = 1.0/(value+penaltyValue);
    }

    return value;
}

int TaskDigest::GetToken(float& reduceTime)
{
    if (this->requiredTokenCount < 1)
    {
        reduceTime=0.0F;
        return -1;
    }
    else
    {
        reduceTime=(*executeTime)[requiredTokenCount-1];
        this->leftRuntime-=reduceTime;
        this->requiredTokenCount -= 1;
        if(this->requiredTokenCount<1)
        {
            this->leftRuntime=0.0F;
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
    return name+"-"+std::to_string(requiredTokenCount+offset);
}
