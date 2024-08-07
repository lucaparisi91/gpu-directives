// Try to compute the matrix product C = A * B
// B : N x M matrix
// A : K x N matrix
// C : K x M matrix
#include <random>
#include <iostream>
#include <mkl_cblas.h>
#include <omp.h>
#include "matrix_cuda_kernels.h"


__host__ void myErrorHandler(cudaError_t ifail, const char * file,
                             int line, int fatal) {
    if (ifail != cudaSuccess) {
        fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
                cudaGetErrorName(ifail), cudaGetErrorString(ifail));
        if (fatal) exit(ifail);
    }

  return;
}

#define CUDA_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }




void set_random_matrix(double * A, size_t N, size_t M,std::mt19937 &rd)
{
    std::uniform_real_distribution<double> dist(0, 1);
    for(size_t i=0;i<N;i++)
        for(size_t j=0;j<M;j++)
        {
            A[i*M + j ]=dist(rd);
        }
}

auto diff_matrix(double * A , double * B,size_t M,size_t N)
{
    double diff=0;
    for(size_t i=0;i<N;i++)
        for(size_t j=0;j<M;j++)
        {
            diff+=std::abs(A[i*M + j ] - B[i*M+j])/(A[i*M + j ]);
        }
    return diff/(N*M);
}

void print_matrix(double * A,size_t N,size_t M)
{
    for(size_t i=0;i< N ;i++)
        {
            for(size_t j=0;j< M ;j++)
                {
                    std::cout << A[i*M + j ] << " ";
                }
            std::cout << std::endl;
        }

}


int main(int argc, char ** argv)
{   

    size_t N=32*1,M=32*1,K=32*1;
    double * A;
    double * B;
    double * C;
    double * A_d;
    double *B_d;
    double *C_d;
    double * C_test;
    const double tol=1e-7;
    int seed = 1039;
    int nTrials=10;

    std::random_device rd;

    A = new double [K * N];
    B = new double [ M * N];
    C = new double [ K *M];
    C_test = new double [ K *M];

    CUDA_ASSERT( cudaMalloc(&A_d, K*N*sizeof(double)) );
    CUDA_ASSERT( cudaMalloc(&B_d, M*N*sizeof(double)) );
    CUDA_ASSERT( cudaMalloc(&C_d, K*M*sizeof(double)) );

    
    std::mt19937 mt( seed );
    set_random_matrix(C,K,M,mt );
    mt = std::mt19937( seed );
    set_random_matrix(C_test,K,M,mt );

    set_random_matrix(A,K,N,mt );
    set_random_matrix(B,N,M,mt );

    CUDA_ASSERT( cudaMemcpy(A_d, A, K*N*sizeof(double),  cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(B_d, B, M*N*sizeof(double),  cudaMemcpyHostToDevice) );
    CUDA_ASSERT( cudaMemcpy(C_d, C, K*M*sizeof(double),  cudaMemcpyHostToDevice) );

    // for(size_t i = 0 ; i< K; i++ )
    // {
    //     for(size_t j=0;j<M;j++)
    //     {
    //         C[i*M + j ]=0;
    //         for(size_t l=0;l<N;l++)
    //         {
    //             C[i*M + j ]+=A[i*N + l]*B[l*M + j];
    //         }
    //     }
    // }


    for(int i=0;i<nTrials;i++)
    {
        cblas_dgemm(  
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            (int)K,
            (int)M,
            (int)N,
            1e+0,
            A,
            (int)N,
            B,
            (int)M,
            1e+0,
            C_test,
            (int)M
        );
    }
    // std::cout << "A" << std::endl;
    // print_matrix(A,K,N);

    // std::cout << "B" << std::endl;
    // print_matrix(B,N,M);
    

    // std::cout << "C" << std::endl;


    // print_matrix(C,K,M);

    // std::cout << "C_ref" << std::endl;
    // print_matrix(C_test,K,M);


    int nThreads = 32;
    dim3 blockSize(AN,AK);

    dim3 gridSize (
        (M + AN - 1 )/AN,
        (K + AK - 1 )/AK,
        1
    );


    if (  ((K % nThreads) != 0) || ((M % nThreads) != 0 ) || ((N % nThreads) != 0) )
    {
        std::cout << "All dimensions should be a multiple of 32." << std::endl;
    }


    auto start = omp_get_wtime();
    for(int i=0;i<nTrials;i++)
    {
        matrix_matrix_shared_kernel2<<< gridSize,blockSize>>>(A_d,B_d,C_d,(int)K,(int)M,(int)N);
    }
    CUDA_ASSERT( cudaDeviceSynchronize() );
    auto end = omp_get_wtime();



    CUDA_ASSERT( cudaPeekAtLastError() );
    CUDA_ASSERT( cudaMemcpy(C, C_d, K*M*sizeof(double),  cudaMemcpyDeviceToHost) );
    
    if(diff_matrix(C,C_test,K,M) > tol )
        {
            std::cout << "Failed! " <<std::endl;
            std::cout << "C " << std::endl;
            print_matrix(C,K,M);
            std::cout << "C ref." << std::endl;
            print_matrix(C_test,K,M);

            exit(1);
        }
    else 
    {
        std::cout << "Success!" << std::endl;
    }

    std::cout << "Time per replica( micros ): " << (end - start)*1e+6/nTrials << std::endl;


}