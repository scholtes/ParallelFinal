# ParallelFinal
Final project for Parallel Computing -- root finding by *n*-ary section method (e.g., parallelized bisection method).

##Organization

Herein lies the directory and project strutcture

* `src/` - Contains the bulk of the project code  
  * `rootfinding.h` - Contains the implementations of the root-finding methods  
  * `rtfindanalyt.h` - A version of the root-finding methods that count how many are used  
  * `tests.cu` - A basic demonstration of the methods
  * `tests2.cu` - Benchmarks
  * `tests3.cu` - Use `rtfindanalyt.h` to compute number of steps used for each algorithm  
* `sandbox/` - Useless garbage (playing around with stuff)  
  * `test1.cu` - Thrust example from NVIDIA's site  
  * `test2.cu` - Experimenting with functors in CUDA  

All of `tests.cu`, `tests2.cu`,`tests3.cu`, `test1.cu`, and `test2.cu` can be individually compiled with `nvcc` from within their respective directories.  The executables can be run directly no args.
