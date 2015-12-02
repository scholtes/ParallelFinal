/* Author: Garrett Scholtes
 * Date:   2015-12-02
 *
 * tests.cu - A test bed for the root-finding
 * functions, including benchmarks.
 */

#include <stdlib.h>
#include <stdio.h>

#include "rootfinding.h"


// Mock some expensive function that is continuous and sign changing
struct expensive_functor {
    __host__ __device__
    float operator()(float value) const {
        return 1 - (value + 0.1f) * (value + 0.1f);
    }
};


// Run the tests here
int main(void) {

    expensive_functor f;

    printf("Root of   f(x) = 1 - (x+0.1)^2   in [0, 1]: %0.5f\n", findRootSequential(0, 1, f));

    return 0;
}