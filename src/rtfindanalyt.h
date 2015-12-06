// ANALYTICAL version of rootfinding.h
// Reports how many steps the algorithms take, for
// purposes of analysis.

#ifndef ROOTFINDER_H
#define ROOTFINDER_H

#define BLOCK_WIDTH 1024

// Micro-optimization.  Ugly, but it works.
// Checks if sign(X) == sign(Y)
#define __SIGNS_CMP(X, Y) (((X)<0.0f?-1:((X)==0.0f?0:1))==((Y)<0.0f?-1:((Y)==0.0f?0:1)))


// Like in rootfinding.h, but returns how many steps the algorithm takes
template <typename UnaryFunction>
unsigned int findRootSequential(float a,
                         float b,
                         UnaryFunction & f)
{
    float old_diff;
    float midpoint;

    float f_a;
    float f_midpoint;

    unsigned int steps = 0;

    do {
        steps++;

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

    return steps;
}




/////////////////////////////// KERNEL FUNCTIONS ///////////////////////////////

template <typename UnaryFunction>
__global__ void coarseRootKernel(unsigned int *result,
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

    unsigned int steps = 0;

    do {
        steps++;

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

    *result = steps;
    return;
}

///////////////////////////// END KERNEL FUNCTIONS /////////////////////////////




// Parallel bisection, but reports # of steps used in algorithm instead of a result
template <typename UnaryFunction>
unsigned int findRootParallel1(float a,
                        float b,
                        UnaryFunction & f)
{
    unsigned int *result_h;
    unsigned int *result_d;
    unsigned int returnVal;
    cudaMalloc(&result_d, sizeof(unsigned int));
    result_h = (unsigned int *)malloc(sizeof(unsigned int));

    coarseRootKernel<<<1, BLOCK_WIDTH>>>(result_d, a, b, f);

    cudaMemcpy(result_h, result_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    returnVal = *result_h;

    free(result_h);
    cudaFree(result_d);

    return returnVal;
}





#endif