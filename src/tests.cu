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

    return 0;
}