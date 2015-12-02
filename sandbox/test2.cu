/* Author: Garrett Scholtes
 * Date:   2015-12-02
 *
 * tests.cu - A test bed for the root-finding
 * functions, including benchmarks.
 */

#include <stdlib.h>
#include <stdio.h>


// Mock some expensive function that is continuous and sign changing
struct expensive_functor {
    __host__ __device__
    float operator()(float value) const {
        return 1 - (value + 0.1f) * (value + 0.1f);
    }
};

template <typename UnaryFunction> __global__ void evalFunctorKernel(float * result, const UnaryFunction & f) {
    *result = f(0.3);
}

template <typename UnaryFunction> float evaluatesFunctorAt03(const UnaryFunction & f) {
    float *result_h;
    float *result_d;
    float returnVal;
    cudaMalloc(&result_d, sizeof(float));
    result_h = (float *)malloc(sizeof(float));

    evalFunctorKernel<<<1, 1>>>(result_d, f);

    cudaMemcpy(result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);
    returnVal = *result_h;

    free(result_h);
    cudaFree(result_d);

    return returnVal;
}


// Run the tests here
int main(void) {

    expensive_functor someFunctorFunction;

    float result = evaluatesFunctorAt03(someFunctorFunction);//someFunctorFunction(0.3f);

    printf("evaluatesFunctorAt03 = %0.2f\n", result);

    return 0;
}