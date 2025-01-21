
#include "grid.h"
#include "hip/hip_runtime.h"

__global__ void compute_jacobi_hip_kernel(double * field_phi_new, double * field_phi_old, double * field_rho, grid_t * g )
{
    int i  = blockDim.x * blockIdx.x + threadIdx.x;
    int j  = blockDim.y * blockIdx.y + threadIdx.y;
    
    if ( (i<g->n[0]) and (j < g->n[1]) )
    {
        auto k = (j +1) + (i+1)*(g->n[1]+2);

        auto nx = g->n[0];
        auto ny = g->n[1];
        auto dx = g->dx[0];
        auto dy = g->dx[1];
        double aspect2 = (dy/dx)*(dy/dx);

        field_phi_new[ k ] = 0.5*( (field_phi_old[k - 1] +  field_phi_old[k + 1 ])/(1 + 1./aspect2) + (field_phi_old[k - (ny+2) ] +  field_phi_old[(k + ny+2) ])/(1 + aspect2) - field_rho[k] * dx*dx/( 1 + aspect2   ) );
    }
    
}


void launch_compute_jacobi_hip(double * phi_new_dev, double * phi_old_dev, double * rho_dev, grid_t * g_dev, int nx, int ny )
{
    dim3 blockSize(8,8*2,1);
    dim3 gridSize((nx+1)/8,(ny+1)/(8*2),1);
    compute_jacobi_hip_kernel<<<blockSize,gridSize>>>(phi_new_dev, phi_old_dev, rho_dev,g_dev);

}