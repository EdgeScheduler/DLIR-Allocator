#ifndef __POSSIONRANDOM_H__
#define __POSSIONRANDOM_H__

#include <random>

class PossionRandom
{
public:
    PossionRandom(unsigned int seed = 0);
    float Random(float lambda);

private:
    std::default_random_engine engin;
    std::uniform_real_distribution<double> uniform_creator;
};

#endif // __POSSIONRANDOM_H__