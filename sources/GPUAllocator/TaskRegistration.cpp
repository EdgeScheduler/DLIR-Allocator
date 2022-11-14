#include "GPUAllocator/TaskRegistration.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include "Common/DILException.h"

TaskRegistration::TaskRegistration(TokenManager *tokenManager, std::condition_variable *dealTask) : queueLength(0.0F), tokenManager(tokenManager), dealTask(dealTask), currentTask(nullptr), closeRegistration(false), reduceTime(0.0F)
{
}

void TaskRegistration::RegisteTask(std::string name, std::shared_ptr<std::vector<float>> executeTime, int requiredToken, int requiredTokenCount, float &modelExecuteTime, const int &taskCount)
{
    TaskDigest task(name, executeTime, requiredToken, requiredTokenCount, modelExecuteTime, taskCount);
    std::unique_lock<std::mutex> lock(mutex);

    if (reduceTime > 0)
    {
        queueLength -= reduceTime;
        reduceTime = 0.0F;
    }

#ifdef PARALLER_MODE
    tasks.push_front(task);
    this->queueLength += task.LeftRunTime();
#else
    if (tasks.size() < 1)
    {
        queueLength = 0.0F;
        tasks.push_front(task);
        goto SCHEDULE;
    }
    else
    {
        float total_wait = queueLength;
        for (auto iter = tasks.begin(); iter != tasks.end();)
        {
            if (iter->requiredTokenCount < 1)
            {
                iter = tasks.erase(iter);
                continue;
            }

#ifdef FIFO_MODE
            tasks.insert(iter, std::move(task));
            goto SCHEDULE;

#endif // DEBUG

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

            iter++;
        }
        tasks.insert(tasks.end(), std::move(task));

        // update current_task (queue head)
        this->currentTask.store(&tasks.back());

        goto SCHEDULE;
    }

SCHEDULE:
#endif // PARALLER_MODE
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
        int next_token = 0;

        if (reduceTime > 0)
        {
            std::unique_lock<std::mutex> lock(mutex);
            queueLength -= reduceTime;
            reduceTime = 0.0F;
            lock.unlock();
        }

        while (true)
        {
            // std::unique_lock<std::mutex> lock(mutex);
            auto m = this->currentTask.load();
            if (m == nullptr || m->requiredTokenCount < 1)
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

                    m = &tasks.back();
                    if (m->requiredTokenCount < 1)
                    {
                        tasks.pop_back();
                        continue;
                    }
                    else
                    {
                        currentTask.store(m);
                        break;
                    }
                }
                lock.unlock();
            }

            tokenManager->WaitFree();
            m = this->currentTask.load();

            bool enableSegmentation = m->taskCount < 2;

#if (defined(FIFO_MODE) || defined(BNST_MODE))
            enableSegmentation = false;
#endif

#ifdef OYST_MODE
            enableSegmentation = true;
#endif

            next_token = m->GetToken(reduceTime, enableSegmentation); // currentTask is allowed to be update by TaskRegistration::RegisteTask
            // debug[next_token]+=1;

            tokenManager->Grant(next_token, enableSegmentation);

            while (true)
            {
                this->dealTask->notify_all();
                std::this_thread::sleep_for(std::chrono::microseconds(5));
                if (tokenManager->GetFlag() < 1)
                {
                    break;
                }
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
    this->closeRegistration = true;
    this->m_notEmpty.notify_all();
}
