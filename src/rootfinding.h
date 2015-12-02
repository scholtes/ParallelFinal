#ifndef ROOTFINDER_H
#define ROOTFINDER_H

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

#endif