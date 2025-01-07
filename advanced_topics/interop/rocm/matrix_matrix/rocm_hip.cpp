// Try to compute the matrix product C = A * B
// B : N x M matrix
// A : K x N matrix
// C : K x M matrix
#include <random>
#include <iostream>
#include <omp.h>
#include <rocblas.h>
#include <hip/hip_runtime_api.h>


__host__ void myErrorHandler(hipError_t ifail, const char * file,
                             int line, int fatal) {
    if (ifail != hipSuccess) {
        fprintf(stderr, "Line %d (%s): %s: %s\n", line, file,
                hipGetErrorName(ifail), hipGetErrorString(ifail));
        if (fatal) exit(ifail);
    }

  return;
}

#define HIP_ASSERT(call) { myErrorHandler((call), __FILE__, __LINE__, 1); }




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

    size_t N=4,M=2,K=3;
    double * A;
    double * B;
    double * C;


    double * A_d;
    double *B_d;
    double *C_d;
    double * C_test;
    const double tol=1e-7;
    int seed = 1039;
    int nTrials=1;

    std::random_device rd;

    A = new double [K * N];
    B = new double [ M * N];
    C = new double [ K *M];
    C_test = new double [ K *M];

    HIP_ASSERT( hipMalloc(&A_d, K*N*sizeof(double)) );
    HIP_ASSERT( hipMalloc(&B_d, M*N*sizeof(double)) );
    HIP_ASSERT( hipMalloc(&C_d, K*M*sizeof(double)) );
    
    std::mt19937 mt( seed );

    set_random_matrix(A,K,N,mt );
    set_random_matrix(B,N,M,mt );
    
    std::fill ( C , C+K*M, 0);

    HIP_ASSERT( hipMemcpy(A_d, A, K*N*sizeof(double),  hipMemcpyHostToDevice) );
    HIP_ASSERT( hipMemcpy(B_d, B, M*N*sizeof(double),  hipMemcpyHostToDevice) );
    HIP_ASSERT( hipMemcpy(C_d, C, K*M*sizeof(double),  hipMemcpyHostToDevice) );

    rocblas_operation trans_a =  rocblas_operation_none ;
    rocblas_operation trans_b =  rocblas_operation_none ;


    rocblas_handle handle;
    rocblas_create_handle(&handle);

    double alpha=1;
    double beta=0;

    for(int i=0;i<nTrials;i++)
    {
        rocblas_dgemm(  
            handle,
            trans_b, trans_a,
            (int)M,
            (int)K,
            (int)N,
            &alpha,
            B_d,
            (int)M,
            A_d,
            (int)N,
            &beta,
            C_d,
            (int)M
        );

    }

    HIP_ASSERT( hipPeekAtLastError() );
    HIP_ASSERT( hipMemcpy(C, C_d, K*M*sizeof(double),  hipMemcpyDeviceToHost) );

    std::cout << "A" << std::endl;
    print_matrix(A,K,N);
    std::cout << "B" << std::endl;
    print_matrix(B,N,M);
    std::cout << "C" << std::endl;
    print_matrix(C,K,M);
    
    std::cout << "Success" << std::endl;

}   