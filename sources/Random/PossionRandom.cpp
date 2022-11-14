#include "Random/PossionRandom.h"

PossionRandom::PossionRandom(unsigned int seed) : engin(seed), uniform_creator(0, 1)
{
}

float PossionRandom::Random(float lambda)
{
    // for (int i = 0; i < this->data.size(); i++)
    // {
    //     this->data[i] = (T)uniform_creator(engin);
    // }

    float p = 0.0F;
    int k = 0;

    while (p < lambda)
    {
        k++;
        float u = uniform_creator(engin);
        p -= log(u);
    }

    return k - 1;
}
