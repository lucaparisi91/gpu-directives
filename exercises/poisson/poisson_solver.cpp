

#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>
#include <roctracer/roctx.h>

/**
 * Defines a grid structure.
 * The underlying data structure contains ghost cells and is ordered row-wise.
 * 
 * (-1,-1) , (-1,0) ... (-1,n[1])
 * .                    .
 * .                    .
 * .                    .
 * (n[0] + 1,-1) , ... , (n[0] + 1,n[1])
 * 
*/


struct timer
{
    timer(std::string name) : _name(name), _time(0),_start(0),_niterations(0) {}
    void start() { _start=omp_get_wtime(); }
    void stop() {_time += omp_get_wtime() - _start;_start=0;_niterations+=1;}

    auto get_time() const {return _time;}
    auto get_name() const {return _name;}
    auto get_niterations() const {return _niterations;}

    private:
    std::string _name;
    double _start;
    double _time;
    size_t _niterations;

};


std::ostream& operator<<(std::ostream & os, const timer & t)
{
    os << t.get_name() << ": ";
    if (t.get_niterations() > 1) 
    {
        os << t.get_time()*1000./t.get_niterations() << " ms/it"  ;
    }
    else
    {
        os << t.get_time() << "s";
    }

    return os;
}

struct grid
{
    /**
     * Returns the index of the location in memory associated with cell (i,j) .
     * i.e : (0,0) -> 1() + n[1] + 2
    */
    inline auto get_index(size_t i, size_t j) const { return (j +1) + (i+1)*(n[1]+2);};

    inline double x(size_t i) const {return start[0] + dx[0] * i  ;}
    inline double y(size_t j) const {return start[1] + dx[1] * j  ;}

    inline size_t size() const { return (n[0]+2) * (n[1] + 2);}
    
    
    size_t n[2];
    double dx[2];
    double start[2];
    double end[2];
};


grid make_grid(const double start[2],const double end[2], size_t n[2]  )
{
    struct grid new_grid;
    new_grid.start[0]=start[0];
    new_grid.start[1]=start[1];

    new_grid.end[0]=end[0];
    new_grid.end[1]=end[1];

    new_grid.n[0]=n[0];
    new_grid.n[1]=n[1];

    new_grid.dx[0]=(new_grid.end[0] - new_grid.start[0])/n[0];
    new_grid.dx[1]=(new_grid.end[1] - new_grid.start[1])/n[1];

    return new_grid;
}


/**
 * Makes a step of the jacobi interaction. Compute the new field, given the hold field and the know densitiy field.
*/ 
void compute_jacobi(double * phi_new, const double * phi_old, const double * rho, const grid * g)
{
    auto nx = g->n[0];
    auto ny = g->n[1];

    auto dx = g->dx[0];
    auto dy = g->dx[1];
    double aspect2 = (dy/dx)*(dy/dx);
    
    // interior
    #pragma omp target teams distribute parallel for collapse(2)
    for(int i=0;i<nx;i++)
        for( int j=0;j<ny;j++)
        {

            auto k = g->get_index(i,j);

            phi_new[ k ] = 0.5*( (phi_old[k - 1] +  phi_old[k + 1 ])/(1 + 1./aspect2) + (phi_old[k - (ny+2) ] +  phi_old[(k + ny+2) ])/(1 + aspect2) - rho[k] * dx*dx/( 1 + aspect2   ) );
        }
};


/**
 * 
 * Initialise the file field with a isotropic gaussian
 */

void init_gaussian(double alpha, double * field, const grid * g)
{
    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            double r2=g->x(i)* g->x(i) + g->y(j)*g->y(j);
            size_t k = g->get_index(i,j);
            field[ k  ] = std::exp( - alpha* r2 );
        }
};

/**
 * Applies Drichlet boundary conditions.
 * @param value Value to be set on all ghost cells
 * @param g Grid definition the space discretization of a field
 * @param field An array of double containing the field values
*/
void apply_drichlet_bc(double * field, const grid * g, double value)
{

    for(int i=0;i< g->n[0];i++)
        {
            auto k_ghost_left = g->get_index(i,-1);
            auto k_ghost_right = g->get_index(i,g->n[1] );

            field[ k_ghost_left ] = value;
            field[ k_ghost_right ] = value;            
        }
   

   for(int j=0;j<g->n[1];j++)
    {
             auto k_ghost_top = g->get_index(-1,j);
             auto k_ghost_bottom = g->get_index(g->n[0],j );

             field[ k_ghost_bottom ] = value;
             field[ k_ghost_top ] = value; 
    }


}



/**
 * Applies periodic boundary condition by filling ghost cells
*/
void apply_periodic_bc(double * field, const grid * g)
{
    #pragma omp target teams distribute parallel for
    for(int i=0;i< g->n[0];i++)
        {
            auto k_ghost_left = g->get_index(i,-1);
            auto k_right = g->get_index(i,g->n[1] -1 );
            auto k_ghost_right = g->get_index(i,g->n[1] );
            auto k_left = g->get_index(i,0);

            field[ k_ghost_left ] = field[k_right];
            field[ k_ghost_right ] = field[k_right];
            
        }


   #pragma omp target teams distribute parallel for
   for(int j=0;j<g->n[1];j++)
    {
             auto k_ghost_top = g->get_index(-1,j);
             auto k_ghost_bottom = g->get_index(g->n[0],j );
             auto k_top = g->get_index(0,j);
             auto k_bottom = g->get_index(g->n[0]-1,j );
             

             field[ k_ghost_bottom ] = field[k_top];
             field[ k_ghost_top ] = field[k_bottom];
             
    }
}



/**
 * Initialize the field with a constant
*/
void init_constant(double alpha, double * field, const grid * g)
{
    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            size_t k = g->get_index(i,j);
            field[ k  ] = alpha;
        }

};


/**
 * Save the field to a file in binary format
*/
void print_to_file(const double * field, const grid * g,std::string filename)
{
    std::ofstream f;
    size_t nx = g->n[0];
    size_t ny= g->n[1];

    
    f.open(filename,std::ios::binary | std::ios::out);

    f.write( (const char *)&nx, sizeof(size_t)  );
    f.write( (const char *)&ny, sizeof(size_t)  );

    f.write( (const char *)field, sizeof(double)*(g->n[0]+2)*(g->n[1]+2)  );

};

/**
 * Initialise the field with the laplacian of a gaussian. Mostly meant for testing.
*/
void init_laplacian_gaussian(double alpha, double * field, const grid * g)
{

    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            double r2=g->x(i)* g->x(i) + g->y(j)*g->y(j);
            size_t k = g->get_index(i,j);
            field[ k  ] = -2*alpha*( 2 - 2*alpha*r2)* std::exp( - alpha* r2 );
        }

}

double get_norm(const double * field, const grid * g)
{
    double norm=0;
    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            size_t k = g->get_index(i,j);
            norm+=field[ k  ] * (g->dx[0] * g->dx[1]);
        }
    return norm;
}



int main(int argc, char ** argv)
{
    size_t nx = 1000;
    size_t ny= 1000;

    int niterations = 10;
    
    double left_box[2]= {-1,-1};
    double right_box[2]= {1,1};
    size_t shape[2] = {nx,ny};

    auto current_grid = make_grid(left_box,right_box,shape);

    auto grid_size = (nx + 2)* (ny + 2);

    double * rho = new double[grid_size];
    double * phi1 = new double[grid_size];
    double * phi2 = new double [grid_size];

    double * phi_old;
    double * phi_new;

    init_laplacian_gaussian(10.0 ,rho,&current_grid);
    print_to_file(rho,&current_grid,"rho.dat");

    init_constant(0.0,phi1,&current_grid);
    init_gaussian(20.0 ,rho,&current_grid);
    
    apply_drichlet_bc(rho, &current_grid,0);
    apply_drichlet_bc(phi1, &current_grid,0);
    apply_drichlet_bc(phi2, &current_grid,0);


    phi_new = phi1;
    phi_old = phi2;


    timer compute_jacobi_timer("compute_jacobi");
    timer apply_periodic_bc_timer("apply_periodic_pbc");
    timer total_time_timer("total_time");


    total_time_timer.start();


    #pragma omp target data map(tofrom:phi_new[0:(nx+2)*(ny+2)],phi_old[0:(nx+2)*(ny+2)], rho[0:(nx+2)*(ny+2)],current_grid,current_grid.n[0:2],current_grid.dx[0:2],current_grid.start[0:2],current_grid.end[0:2])
    {
        
        
        for (int i=0;i<niterations;i++)
        {

            std::swap(phi_new,phi_old);
            compute_jacobi_timer.start();
            roctx_range_id_t roctx_jacobi_id = roctxRangeStartA("jacobi");
            compute_jacobi(phi_new,phi_old,rho,&current_grid);
            roctxRangeStop(roctx_jacobi_id);
            compute_jacobi_timer.stop();

            //apply_periodic_bc_timer.start();
            //roctx_range_id_t roctx_bc_id = roctxRangeStartA("periodic_bc");
            //apply_periodic_bc(phi_new, &current_grid);
            //roctxRangeStop(roctx_bc_id);
            //apply_periodic_bc_timer.stop();

        }
        
        total_time_timer.stop();

    }


    print_to_file(phi2,&current_grid,"phi.dat");

    std::cout << total_time_timer << std::endl;
    std::cout << compute_jacobi_timer << std::endl;
    std::cout << apply_periodic_bc_timer << std::endl;
    
}