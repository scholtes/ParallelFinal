/* Author: Garrett Scholtes
 * Date:   2015-12-02
 *
 * tests.cu - A test bed for the root-finding
 * functions, including benchmarks.
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctime>

#include "rootfinding.h"


// Mock some expensive function that is continuous and sign changing
struct expensive_functor {
    __host__ __device__
    float operator()(float value) const {
        for(int i = 0; i < 10000; i++){}
        return 1 - (value + 0.1f) * (value + 0.1f);
    }
};


// Run the tests here
int main(void) {

    std::clock_t start;
    unsigned int duration;
    expensive_functor f;
    float result;

    start = std::clock();
    for(int i = 0; i < 1000; i++) result = findRootSequential(0, 1, f);
    duration = (std::clock() - start);
    printf("[SEQUENTIAL] Root of   f(x) = 1 - (x+0.1)^2   in [0, 1]: %0.5f\n", result);
    printf("    Duration = %d cycles\n", duration);

    start = std::clock();
    for(int i = 0; i < 1000; i++) result = findRootParallel1(0, 1, f);
    duration = (std::clock() - start);
    printf("[PARALLEL]   Root of   f(x) = 1 - (x+0.1)^2   in [0, 1]: %0.5f\n", result);
    printf("    Duration = %d cycles\n", duration);

    return 0;
}