# Notes on the performance poisson solver

nFields | shape | hardware | Iterations | Time per it.(ms) | notes
-- | --- | --- | -- | --
1 | 500x500 | 1 CPU | 10 | 0.238957 | base
1 | 500x500 | 1 GPU | 10 | 47.2857 | offload at every iteration, including first step
1 | 500x500 | 1 GPU | 10 | 0.296866 | offload only once per block, including first step
1 | 500x500 | 1 GPU | 10 | 0.0186408 | offload only once per block, excluding first step. The kernel itself runs for $7 \mu s$. Thus kernel issue latency seems to be $11 \mu/s.$. This is measured using an empty kernel.
1 | 1000x1000 | 1 GPU | 10 | 0.02014 | offload only once per block, excluding first step. The kernel itself runs for $7 \mu s$. Thus kernel issue latency seems to be $11 \mu/s.$. This is measured using an empty kernel.


# Compiling

You can compile with

```bash
CC -lroctracer64 -lroctx64 -g -O2 -fopenmp poisson_solver_omp.cpp  -o test
```