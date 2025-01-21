set -e
set -x

hipcc -ferror-limit=1 -c -std=c++17 -x hip -D__HIP_ROCclr__ --rocm-path=${ROCM_PATH} -D__HIP_PLATFORM_AMD__ --offload-arch=gfx90a jacobi_kernel.cpp -o jacobi_kernel.o
CC -ferror-limit=1 -fopenmp -c -D__HIP_PLATFORM_AMD__ poisson_solver_omp.cpp -o poisson_solver_omp.o
CC -lroctracer64 -lroctx64 -g -O2 -fopenmp  poisson_solver_omp.o jacobi_kernel.o -o test
