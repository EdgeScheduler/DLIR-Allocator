#ifndef __TOKENMANAGER_H__
#define __TOKENMANAGER_H__

#include <mutex>
#include <vector>
#include <condition_variable>

class TokenManager
{
public:
    TokenManager();

    /// @brief set to free
    void Release();

    /// @brief set token expire
    void Expire();

    /// @brief give token to xx
    /// @param token ID, 0 means free
    /// @param enableSegmentation if false, will run total model.
    /// @return
    bool Grant(int token, bool enableSegmentation = true);
    int GetFlag();

    /// @brief block until flag<=0
    void WaitFree();

    std::condition_variable& NeedNewToken();

    void CloseTokenManager();

    operator int();

private:
    int flag; // 0: free 1~n: token_id
    std::mutex mutex;  
    std::mutex runningMutex;
    std::shared_ptr<std::unique_lock<std::mutex>> runningLock;
    std::condition_variable needNewToken;
    bool closeTokenManager;
};

#endif // __TOKENMANAGER_H__