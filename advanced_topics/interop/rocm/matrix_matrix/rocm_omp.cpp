// Try to compute the matrix product C = A * B
// B : N x M matrix
// A : K x N matrix
// C : K x M matrix
#include <random>
#include <iostream>
#include <omp.h>
#include <rocblas/rocblas.h> 


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

    const double tol=1e-7;
    int seed = 1039;
    int nTrials=1;

    std::random_device rd;

    A = new double [K * N];
    B = new double [ M * N];
    C = new double [ K *M];
    

    std::mt19937 mt( seed );

    set_random_matrix(A,K,N,mt );
    set_random_matrix(B,N,M,mt );

    std::fill ( C , C+K*M, 0);


    rocblas_operation trans_a =  rocblas_operation_none ;
    rocblas_operation trans_b =  rocblas_operation_none ;

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    double alpha=1;
    double beta=0;

    #pragma omp target data map(tofrom:A[0:K*N],B[0:N*M],C[0:K*M])
    {
        #pragma omp target data use_device_addr(A[:K*N],B[:K*N],C[:K*N])
        {
            for(int i=0;i<nTrials;i++)
            {
                rocblas_dgemm(  
                    handle,
                    trans_b, trans_a,
                    (int)M,
                    (int)K,
                    (int)N,
                    &alpha,
                    B,
                    (int)M,
                    A,
                    (int)N,
                    &beta,
                    C,
                    (int)M
                );
        }


        }
    
    }


    std::cout << "A" << std::endl;
    print_matrix(A,K,N);
    std::cout << "B" << std::endl;
    print_matrix(B,N,M);
    std::cout << "C" << std::endl;
    print_matrix(C,K,M);
    
    std::cout << "Success" << std::endl;

}   