

/// Contains geometrical and discretisation information

#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <roctracer/roctx.h>
#include "grid.h"
#include "timer.h"
#include "field.h"
#include "bc.h"
#include "jacobi.h"
#include "tools.h"

#pragma omp declare mapper(grid_t g) map(g,g.n[0:2], g.dx[0:2],g.start[0:2],g.end[0:2]  )

#pragma omp declare mapper(field_t f) map(f,f.data[0:f.get_size()])

double get_distance( const field_t & field1 , const  field_t & field2, grid_t * g)
{
    auto field_phi_new=field1.get_data();
    auto field_phi_old=field2.get_data();

    double distance=0;
    
    for(int i=0; i<g->n[0] ; i++)
        for( int j=0; j<g->n[1] ; j++)
            {
                auto k = g->get_index(i,j);
                distance+= (field_phi_new[ k ] - field_phi_old[ k ]) * (field_phi_new[ k ] - field_phi_old[ k ]);
            }

    return distance;

}


int main(int argc, char ** argv)
{
    /**
     * Definitions
    */
    int nFields= 1; // number of equations to solve
    int nIterations = 10000; // Total number of iterations
    int nOutputBlocks = 10; // Approximate number of times to perform output during the simulation
    
    size_t shape[2] = { 500 , 500 }; // Grid shape
    double left_box[2]= {-1,-1}; // Coordinate of the bottom left corner
    double right_box[2]= {1,1}; // Cooridinate of the top right corner

    /**
     * Initialization
    */
    
    int nIterationsOutput = nIterations/nOutputBlocks; // Number of iterations between successive outputs
    std::cout << "Initialise" << std::endl;

    auto current_grid = make_grid(left_box,right_box,shape);
    
    field_t * rho = new field_t[nFields];
    field_t * phi1 = new field_t[nFields];
    field_t * phi2 = new field_t[nFields];

    for (int iField=0 ; iField < nFields; iField++ )
    {
        
        rho[iField].init( &current_grid);
        phi1[iField].init( &current_grid);
        phi2[iField].init(&current_grid);

        init_laplacian_gaussian( 10.0 ,rho[iField].get_data(), &current_grid );
        init_gaussian( 30 ,phi1[iField].get_data(),&current_grid, 0.1 );
        init_gaussian( 30 ,phi2[iField].get_data(),&current_grid, 0.1 );

        apply_drichlet_bc(rho[iField].get_data(), &current_grid, 0);
        apply_drichlet_bc(phi1[iField].get_data(), &current_grid, 0);
        apply_drichlet_bc(phi2[iField].get_data(), &current_grid, 0);

        print_to_file( rho[iField].get_data(), &current_grid, "rho" + std::to_string(iField) + ".dat" );
        print_to_file( phi1[iField].get_data(), &current_grid, "phi" + std::to_string(iField) + "_" + std::to_string(0) + ".dat" );
        
    }

    field_t * phi_old;
    field_t * phi_new;
    
    phi_new = phi1;
    phi_old = phi2;

    timer compute_jacobi_timer("compute_jacobi");
    timer total_time_timer("total_time");

    total_time_timer.start();

    std::cout << "Start " << nIterations << " iterations" << std::endl;
    int i=0;

    if (nIterationsOutput < 0) 
    {
        std::cout << "No iterations per block."<<std::endl;
        exit(1);
    }

    while( i<nIterations )
    {

        /**
         * Calculations 
        */

/*  
        EXIT/ENTER DATA MAP IMPLEMENTATION

        for(int i =0;i< nFields;i++)
        {
            #pragma omp target enter data map(to:current_grid.n,current_grid.dx,current_grid.start,current_grid.end,(phi_new[i].data)[0:current_grid.get_size()],(phi_old[i].data)[0:current_grid.get_size()],(rho[i].data)[0:current_grid.get_size()]  ) 

        } */


         #pragma omp target data map(tofrom: phi_new[0:nFields],phi_old[0:nFields],rho[0:nFields],current_grid)
        {
            for (int iBlock=0;(iBlock<nIterationsOutput) and (i<nIterations);iBlock++)
            {
                std::swap(phi_old,phi_new);
                
                roctx_range_id_t roctx_jacobi_id = roctxRangeStartA("jacobi");
                
                compute_jacobi_timer.start();

                compute_jacobi(phi_new,phi_old,rho,nFields,&current_grid);

                compute_jacobi_timer.stop();
                roctxRangeStop(roctx_jacobi_id);

                i++;

            }
        }
        
            /**
             * Output 
            */

 /*         
            ENTER/EXIT MAPPING IMPLEMENTATION   
            for(int i =0;i< nFields;i++)
            {
                #pragma omp target exit data map(from:current_grid.n,current_grid.dx,current_grid.start,current_grid.end,(phi_new[i].data)[0:current_grid.get_size()],(phi_old[i].data)[0:current_grid.get_size()],(rho[i].data)[0:current_grid.get_size()]  ) 
            } */

            for(int iField=0;iField<nFields;iField++)
            {

                print_to_file(phi_new[iField].get_data(),&current_grid,"phi" + std::to_string(iField) + "_" + std::to_string(i) + ".dat" );
                std::cout << "Distance old - new field " << iField << " = " << get_distance( phi_new[iField] , phi_old[iField] ,&current_grid) << std::endl;

            }
           
        
    }

    

    total_time_timer.stop();

    std::cout << "Finalize" << std::endl;

    std::cout << total_time_timer << std::endl;
    std::cout << compute_jacobi_timer << std::endl;

    
}
