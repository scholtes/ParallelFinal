/* Author: Garrett Scholtes
 * Date:   2015-12-02
 *
 * tests.cu - A test bed for the root-finding
 * functions, including benchmarks.
 */

#include <stdlib.h>
#include <stdio.h>
#include <ctime>

// Does not actually find roots
// This version reports how many steps the algorithms take
#include "rtfindanalyt.h"


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

    expensive_functor f;
    unsigned int steps;

    steps = findRootSequential(0, 1, f);
    printf("[SEQUENTIAL]: Number of steps to find root = %d\n", steps);

    steps = findRootParallel1(0, 1, f);
    printf("[PARALLEL]: Number of steps to find root   = %d\n", steps);


    return 0;
}