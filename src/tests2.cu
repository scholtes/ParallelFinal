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
    int wait_time;

    __host__ __device__
    float operator()(float value) const {
        for(int i = 0; i < wait_time; i++){}
        return 1 - (value + 0.1f) * (value + 0.1f);
    }
};


// Run the tests here
int main(void) {

    std::clock_t start;
    float duration;
    expensive_functor f;
    float result;

    const int phase_1_trials = 100000;
    const int phase_2_3_trials = 1000;
    const int max_cost = 10000;
    const int cost_step = 250;

    // Phase 1: empirically generate and measures functions with different costs
    printf("function_time\n-------------\n");
    for(int i = 0; i < max_cost; i += cost_step) {
        f.wait_time = i;
        start = std::clock();
        for(int trial = 0; trial < phase_1_trials; trial++) f(1.0f);
        duration = (std::clock() - start) / (float)phase_1_trials;
        printf("%0.3f\n", duration);
    }

    // Phase 2: given a function that costs a certain amount, empirically
    // measure how long it takes to find the root of that function using
    // the sequential version
    printf("\nsequential_time\n---------------\n");
    for(int i = 0; i < max_cost; i += cost_step) {
        f.wait_time = i;
        start = std::clock();
        for(int trial = 0; trial < phase_2_3_trials; trial++) findRootSequential(0, 1, f);
        duration = (std::clock() - start) / float(phase_2_3_trials);
        printf("%0.3f\n", duration);
    }

    // Phase 3: given a function that costs a certain amount, empirically
    // measure how long it takes to find the root of that function using
    // the parallel version (same as phase 2 except parallel, not sequential)
    printf("\nparallel_time\n-------------\n");
    for(int i = 0; i < max_cost; i += cost_step) {
        f.wait_time = i;
        start = std::clock();
        for(int trial = 0; trial < phase_2_3_trials; trial++) findRootParallel1(0, 1, f);
        duration = (std::clock() - start) / float(phase_2_3_trials);
        printf("%0.3f\n", duration);
    }

    return 0;
}