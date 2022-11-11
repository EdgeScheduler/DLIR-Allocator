#include "../../include/GPUAllocator/TokenManager.h"
#include "../../include/Common/DILException.h"
#include <iostream>
TokenManager::TokenManager() : flag(-1), runningLock(nullptr),closeTokenManager(false)
{
}

void TokenManager::Release()
{
    this->flag = -1;

#ifndef ALLOW_GPU_PARALLEL
    if (runningLock)
    {
        runningLock->unlock();
        runningLock = nullptr;
    }

    needNewToken.notify_all();
#endif
}

void TokenManager::Expire()
{
    this->flag = 0;
}

bool TokenManager::Grant(int token, bool enableSegmentation)
{
    if(token<1)
    {
        std::cout<<"warning: send token "<<token<<std::endl;
        return true;
    }

#ifndef ALLOW_GPU_PARALLEL
    std::unique_lock<std::mutex> lock(mutex);
    needNewToken.wait(lock, [this]() -> bool
                      { return this->flag < 0|| closeTokenManager; });

    if(closeTokenManager)
    {
        lock.unlock();
        throw DILException::SYSTEM_CLOSE;
    }

    this->runningLock = std::make_shared<std::unique_lock<std::mutex>>(runningMutex);
    if (enableSegmentation)
    {
        this->flag = token<<1;
    }
    else
    {
        this->flag = (token<<1)+1;
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

int TokenManager::GetFlag()
{
    return this->flag;
}

std::condition_variable &TokenManager::NeedNewToken()
{
    return this->needNewToken;
}

void TokenManager::WaitFree()
{
#ifndef ALLOW_GPU_PARALLEL
    std::unique_lock<std::mutex> lock(runningMutex);
    lock.unlock();
    needNewToken.notify_all();
#endif
}

void TokenManager::CloseTokenManager()
{
    this->closeTokenManager=true;
    this->needNewToken.notify_all();
    if (runningLock)
    {
        runningLock->unlock();
        runningLock = nullptr;
    }
}
