#include <thread>
#include <iostream>
#include <mutex>
#include <thread>

#include <memory>
#include <condition_variable>
using namespace std;

std::mutex m;
std::mutex k;
std::unique_lock<std::mutex> l;
volatile int flag = -1;
bool close = false;
std::condition_variable cond;
int count=10000;
int times=0;

void Run()
{
    
    while (true)
    {

        std::unique_lock<std::mutex> lock(m);
        cond.wait(lock,[]()->bool{return flag==0 || close;});
        


        if (close)
        {
            return;
        }
        times++;
        cout <<times<<endl;
    
        
        flag = -1;
        l.unlock();


        if(count==times)
        {
            break;
        }
    }
}

int main()
{
    thread t(Run);

    for (int i = 0; i < count; i++)
    {
        try
        {
            while(l.owns_lock())
            {
                cond.notify_all();
            }

            l=std::move(std::unique_lock<std::mutex>(k));

        }
        catch(...)
        {
            std::cerr << "...................???.................???????????" << '\n';
        }

        flag = 0;
        std::cout<<i+1<<": ";
        // cout<<flag<<"("<<i<<")";
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));

        
        cond.notify_all();
    }

    std::unique_lock<std::mutex> lock(k);
    close = true;
    cond.notify_all();

    t.join();
    return 0;
}