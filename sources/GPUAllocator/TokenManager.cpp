#include "../../include/GPUAllocator/TokenManager.h"

TokenManager::TokenManager() : flag(0.0F)
{
}

void TokenManager::Release()
{
    // std::unique_lock<std::mutex> lock(mutex);
    this->flag = 0.0F;
    // lock.unlock();
    needNewToken.notify_all();
}

bool TokenManager::Grant(float token, bool enableSegmentation)
{
#ifndef ALLOW_GPU_PARALLEL
    std::unique_lock<std::mutex> lock(mutex);
    needNewToken.wait(lock, [this]() -> bool
                           { return this->flag < 1.0F; });
    if(enableSegmentation)
    {
        this->flag = token;
    }
    else
    {
        this->flag = token+0.5F;
    }
    
    lock.unlock();
    return true;
#else
    return true;
#endif
}

TokenManager::operator int()
{
    return this->flag;
}

float TokenManager::GetFlag()
{
    return this->flag;
}

std::condition_variable& TokenManager::NeedNewToken()
{
    return this->needNewToken;
}

void TokenManager::WaitFree()
{
    std::unique_lock<std::mutex> lock(mutex);
    needNewToken.wait(lock, [this]() -> bool
                           { return this->flag < 1.0F; });
    lock.unlock();
    // needNewToken.notify_all();
}
