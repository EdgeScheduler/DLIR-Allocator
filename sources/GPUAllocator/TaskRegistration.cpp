#include "../../include/GPUAllocator/TaskRegistration.h"
#include <iostream>
#include <algorithm>
#include "../../include/Common/DILException.h"

TaskRegistration::TaskRegistration(TokenManager *tokenManager, std::condition_variable *dealTask) : queueLength(0.0F), tokenManager(tokenManager), dealTask(dealTask), currentTask(nullptr), closeRegistration(false)
{
}

void TaskRegistration::RegisteTask(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime)
{
    TaskDigest task(name, executeTime, requiredToken, requiredTokenCount, modelExecuteTime);
    std::unique_lock<std::mutex> lock(mutex);

#ifdef ALLOW_GPU_PARALLEL
    tasks.push_front(task);
    this->queueLength += task.LeftRunTime();
#else
    if (tasks.size() < 1)
    {
        tasks.push_front(task);
        goto SCHEDULE;
    }
    else
    {
        float total_wait = queueLength;
        for (auto iter = tasks.begin(); iter != tasks.end(); iter++)
        {
            // it is not possible to insert before same type of task.
            if (iter->requiredToken == requiredToken)
            {
                tasks.insert(iter, std::move(task));
                goto SCHEDULE;
            }

            float new_task_back = task.Evaluate(total_wait);

            total_wait -= iter->LeftRunTime();
            float new_task_front = task.Evaluate(total_wait);
            float iter_front = iter->Evaluate(total_wait);
            float iter_back = iter->Evaluate(total_wait + task.LeftRunTime());

            // if (new_task_back * iter_front > new_task_front * iter_back)
            if ((new_task_back * iter_front > new_task_front * iter_back && (iter_front > 0 || new_task_front * iter_back > 0)) || (new_task_back * iter_front < 0 && ((new_task_front < 0 && iter_back < 0) || (new_task_front * iter_back < 0 && (std::min(new_task_front, iter_back) > std::min(new_task_back, iter_front))))))
            {
                tasks.insert(iter, std::move(task));
                goto SCHEDULE;
            }
        }
        tasks.insert(tasks.end(), std::move(task));

        // update current_task (queue head)
        this->currentTask = &tasks.back();

        goto SCHEDULE;
    }

SCHEDULE:
#endif // ALLOW_GPU_PARALLEL
    // to release lock;
    this->queueLength = this->queueLength + task.LeftRunTime();
    // std::cout<<"add: "<<task.LeftRunTime()<<std::endl;
    lock.unlock();
    m_notEmpty.notify_all();
    return;
}

void TaskRegistration::TokenDispense()
{
    try
    {
        float reduce_time = 0.0F;
        int next_token = 0;
        // TaskDigest* currentTaskPtr=nullptr;
        while (true)
        {
            // std::unique_lock<std::mutex> lock(mutex);
            // std::string discribe;

            if (this->currentTask == nullptr or this->currentTask->requiredTokenCount < 1)
            {
                // to read valid task
                std::unique_lock<std::mutex> lock(mutex);
                while (true)
                {
                    m_notEmpty.wait(lock, [this]() -> bool
                                    { return tasks.size() > 0 || closeRegistration; });
                    if (closeRegistration)
                    {
                        lock.unlock();
                        throw DILException::SYSTEM_CLOSE;
                    }

                    currentTask = &tasks.back();
                    if (currentTask->requiredTokenCount < 1)
                    {
                        tasks.pop_back();
                        continue;
                    }
                    else
                    {
                        break;
                    }
                }
                lock.unlock();
            }

            // std::cout<<"wait free here."<<std::endl;
            tokenManager->WaitFree();

            // std::cout<<"end wait free here."<<std::endl;
            queueLength = queueLength - reduce_time;

            next_token = this->currentTask->GetToken(reduce_time); // currentTask is allowed to be update by TaskRegistration::RegisteTask
            // std::cout<<"to give token:"<<next_token<<std::endl;

            if (tokenManager)
            {
                // std::cout << next_token << ": " << discribe << std::endl;
                tokenManager->Grant(next_token, true);
                if (tasks.size() < 1)
                {
                    queueLength = 0.0F;
                }
                // std::cout<<"give token done:"<<next_token<<std::endl;

                this->dealTask->notify_all();
            }
        }
    }
    catch (DILException ex)
    {
        if (ex == DILException::SYSTEM_CLOSE)
        {
            std::cout << "end token dispense" << std::endl;
            return;
        }
        else
        {
            std::cout << ex << std::endl;
        }
    }
}

void TaskRegistration::CloseRegistration()
{
    this->closeRegistration=true;
    this->m_notEmpty.notify_all();
}
