#include <thread>
#include <iostream>
#include <mutex>
#include <memory>
#include <condition_variable>
using namespace std;

std::mutex m, k;
std::shared_ptr<std::unique_lock<std::mutex>> l;
int flag = -1;
bool close = false;
std::condition_variable cond;

typedef void Func();
int count=1000;

void Run0()
{
    while (true)
    {

        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, []() -> bool
                  { return flag == 0 || close; });
        if (close)
        {
            throw -1;
        }

        cout << "0"<< endl;
        flag = -1;
        if(l)
        {
            l->unlock();
        }
    }
}

void Run1()
{
    while (true)
    {

        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, []() -> bool
                  { return flag == 1 || close; });

        if (close)
        {
            throw -1;
        }

        cout << "1" << endl;
        flag = -1;
        if(l)
        {
            l->unlock();
        }
    }
}

void Run2()
{
    while (true)
    {

        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, []() -> bool
                  { return flag == 2 || close; });
        if (close)
        {
            throw -1;
        }

        cout << "2" << endl;
        flag = -1;
        if(l)
        {
            l->unlock();
        }
    }
}

void Run3()
{
    while (true)
    {
        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock, []() -> bool
                  { return flag == 3 || close; });
        if (close)
        {
            throw -1;
        }
        cout << "3" << endl;
        flag = -1;
        if(l)
        {
            l->unlock();
        }
    }
}

void Run(Func func)
{
    try
    {
        func();
    }
    catch (int a)
    {
        std::cout << a << endl;
    }
    catch(...)
    {
        cout<<"???"<<endl;
    }
}

int main()
{
    thread t0(Run,Run0);
    thread t1(Run,Run1);
    thread t2(Run,Run2);
    thread t3(Run,Run3);

    for (int i = 0; i < 100000; i++)
    {
        l = std::make_shared<std::unique_lock<std::mutex>>(k);
        flag = i % 4;
        // cout<<flag<<"("<<i<<")";
        cond.notify_all();
    }
    l = std::make_shared<std::unique_lock<std::mutex>>(k);
    l->unlock();
    close = true;
    cond.notify_all();

    t0.join();
    t1.join();
    t2.join();
    t3.join();
    cout<<"to end."<<endl;
    return 0;
}