#ifndef __UNIFORMRANDOM_H__
#define __UNIFORMRANDOM_H__

#include <random>

class UniformRandom
{
public:
    UniformRandom(unsigned int seed = 0);
    float Random();

private:
    std::default_random_engine engin;
    std::uniform_real_distribution<double> uniform_creator;
};

#endif // __UNIFORMRANDOM_H__