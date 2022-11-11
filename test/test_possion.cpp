#include "Random/PossionRandom.h"

#include <iostream>
using namespace std;

// 测试泊松分布
int main()
{
    PossionRandom rand;

    for(int i=0;i<10;i++)
    {
        // lambda is 10
        cout<<rand.Random(12)<<endl;
    }
    return 0;
}

/*

8
10
9
8
14
11
12
9
19
7

*/