#ifndef ROOTFINDER_H
#define ROOTFINDER_H

#define BLOCK_WIDTH 1024

// Micro-optimization.  Ugly, but it works.
// Checks if sign(X) == sign(Y)
#define __SIGNS_CMP(X, Y) (((X)<0.0f?-1:((X)==0.0f?0:1))==((Y)<0.0f?-1:((Y)==0.0f?0:1)))


// Finds the root on an interval for some function
// continuous and sign-changing on that interval.
// Parameters:
//     a - lower bound of interval
//     b - upper bound of interval
//     f - unary function created from a functor
// Returns the location of the root.
// This function uses the bisection method.
template <typename UnaryFunction>
float findRootSequential(float a,
                         float b,
                         UnaryFunction & f)
{
    float old_diff;
    float midpoint;

    float f_a;
    float f_midpoint;

    do {
        old_diff = b - a;
        midpoint = (a + b)/2;

        f_a = f(a);
        f_midpoint = f(midpoint);

        if(__SIGNS_CMP(f_a, f_midpoint)) {
            a = midpoint;
        } else {
            b = midpoint;
        }

    } while(b < old_diff + a);

    return a;
}




/////////////////////////////// KERNEL FUNCTIONS ///////////////////////////////

template <typename UnaryFunction>
__global__ void coarseRootKernel(float *result,
                                 float a,
                                 float b,
                                 UnaryFunction & f)
{
    float old_diff;
    float width;

    float f_a;
    float f_b;

    __shared__ float answer[1];
    *answer = a;

    do {
        old_diff = b - a;
        width = (b - a) / blockDim.x;
        a = (*answer) + width * threadIdx.x;
        b = a + width;

        f_a = f(a);
        f_b = f(b);

        if(!__SIGNS_CMP(f_a, f_b)) {
            *answer = a;
        }

        __syncthreads();
        a = *answer;
        b = a + width;
    } while(b < old_diff + a);

    *result = a;
    return;
}

///////////////////////////// END KERNEL FUNCTIONS /////////////////////////////




// This function uses a parallelized version of the bisection method,
// splitting the interval into n-ary sections and evaluating the intervals
// in parallel.
// VERSION 1: LOCAL MEMORY (coarse-grained -- redundant computations with less memory access time)
template <typename UnaryFunction>
float findRootParallel1(float a,
                        float b,
                        UnaryFunction & f)
{
    float *result_h;
    float *result_d;
    float returnVal;
    cudaMalloc(&result_d, sizeof(float));
    result_h = (float *)malloc(sizeof(float));

    coarseRootKernel<<<1, BLOCK_WIDTH>>>(result_d, a, b, f);

    cudaMemcpy(result_h, result_d, sizeof(float), cudaMemcpyDeviceToHost);
    returnVal = *result_h;

    free(result_h);
    cudaFree(result_d);

    return returnVal;
}





#endif