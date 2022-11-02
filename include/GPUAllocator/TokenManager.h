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

    /// @brief give token to xx
    /// @param token ID, 0 means free
    /// @param enableSegmentation if false, will run total model.
    /// @return
    bool Grant(float token, bool enableSegmentation = true);
    float GetFlag();

    /// @brief block until flag<=0
    void WaitFree();

    std::condition_variable& NeedNewToken();

    operator int();

private:
    float flag; // 0: free 1~n: token_id
    std::mutex mutex;  
    std::condition_variable needNewToken;
};

#endif // __TOKENMANAGER_H__