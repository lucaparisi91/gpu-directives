

/// Contains geometrical and discretisation information

#include <iostream>
#include <cmath>
#include <fstream>
#include <omp.h>

/**
 * Defineds a grid structure.
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


struct grid_t
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

grid_t make_grid(const double start[2],const double end[2], size_t n[2]  )
{
    grid_t new_grid;

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


struct field_t
{

    void init(grid_t * grid_)
    {
        grid=grid_;
        data= new double[grid->size()]{0};

    }

    auto get_data() {return data;}
    const auto get_data() const  {return data;}
    
    auto get_grid() {return grid;}
    const auto get_grid() const {return grid;}

    private:

    double * data;
    grid_t * grid;

};


/**
 * Applies Drichlet boundary conditions.
 * @param value Value to be set on all ghost cells
 * @param g Grid definition the space discretization of a field
 * @param field An array of double containing the field values
*/
void apply_drichlet_bc(double * field, const grid_t * g, double value)
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
 * Makes a step of the jacobi interaction. Compute the new field, given the hold field and the know densitiy field.
*/

void compute_jacobi(field_t * phi_new, field_t * phi_old, field_t * rho, int nFields)
{

    for(int iField=0 ; iField < nFields ; iField++ )
    {
        auto g = phi_new[iField].get_grid();
        auto field_phi_new=phi_new[iField].get_data();
        auto field_phi_old=phi_old[iField].get_data();
        auto field_rho=rho[iField].get_data();

        auto nx = g->n[0];
        auto ny = g->n[1];

        auto dx = g->dx[0];
        auto dy = g->dx[1];
        double aspect2 = (dy/dx)*(dy/dx);


        for(int i=0; i<nx ; i++)
            for( int j=0; j<ny ; j++)
            {

                auto k = g->get_index(i,j);

                field_phi_new[ k ] = 0.5*( (field_phi_old[k - 1] +  field_phi_old[k + 1 ])/(1 + 1./aspect2) + (field_phi_old[k - (ny+2) ] +  field_phi_old[(k + ny+2) ])/(1 + aspect2) - field_rho[k] * dx*dx/( 1 + aspect2   ) );
            }
    }

};


/**
 * 
 * Initialise the file field with a isotropic gaussian A*exp(-alpha*(x**2 + y**2))
 * @param alpha Width of the gaussian
 * @param field Pointer to field object
 * @param A Gaussian amplitude
 */
void init_gaussian(double alpha, double  * data, grid_t * g,double A=1)
{

    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            double r2=g->x(i)* g->x(i) + g->y(j)*g->y(j);
            size_t k = g->get_index(i,j);
            data[ k  ] = A*std::exp( - alpha* r2 );
        }
};

/**
 * Applies periodic boundary condition by filling ghost cells
*/
void apply_periodic_bc(double * field, const grid_t * g)
{
    //#pragma omp target teams distribute parallel for
    for(int i=0;i< g->n[0];i++)
        {
            auto k_ghost_left = g->get_index(i,-1);
            auto k_right = g->get_index(i,g->n[1] -1 );
            auto k_ghost_right = g->get_index(i,g->n[1] );
            auto k_left = g->get_index(i,0);

            field[ k_ghost_left ] = field[k_right];
            field[ k_ghost_right ] = field[k_right];
            
        }


   //#pragma omp target teams distribute parallel for
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
 * 
*/
void init_constant(double alpha, double * field, const grid_t * g)
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
void print_to_file(const double * field, const grid_t * g,std::string filename)
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
void init_laplacian_gaussian(double alpha, double * field, const grid_t * g)
{

    for(int i=0;i<g->n[0];i++)
        for( int j=0;j<g->n[1];j++)
        {
            double r2=g->x(i)* g->x(i) + g->y(j)*g->y(j);
            size_t k = g->get_index(i,j);
            field[ k  ] = -2*alpha*( 2 - 2*alpha*r2)* std::exp( - alpha* r2 );
        }

}

double get_norm(const double * field, const grid_t * g)
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

    int nFields=1; // number of equations to solve
    int niterations = 1000;  // number of time steps
    int nIterationsOutput = niterations/10; // Number of iterations between successive outputs

    double left_box[2]= {-1,-1}; // Coordinate of the bottom left corner
    double right_box[2]= {1,1}; // Cooridinat of the top right corner
    size_t shape[2] = {100, 100 }; // Grid shape
    
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
        init_gaussian( 20 ,phi1[iField].get_data(),&current_grid, 0.5 );
        init_gaussian( 20 ,phi2[iField].get_data(),&current_grid, 0.5 );

        apply_drichlet_bc(rho[iField].get_data(), &current_grid, 0);
        apply_drichlet_bc(phi1[iField].get_data(), &current_grid, 0);
        apply_drichlet_bc(phi2[iField].get_data(), &current_grid, 0);
        
        print_to_file( rho[iField].get_data(), &current_grid, "rho" + std::to_string(iField) + ".dat" );
        print_to_file( phi1[iField].get_data(), &current_grid, "phi_initial" + std::to_string(iField) + ".dat" );

    }

    field_t * phi_old;
    field_t * phi_new;

    phi_new = phi1;
    phi_old = phi2;

    timer compute_jacobi_timer("compute_jacobi");
    timer apply_periodic_bc_timer("apply_periodic_pbc");
    timer total_time_timer("total_time");


    total_time_timer.start();

    std::cout << "Start iterations" << std::endl;
    int i=0;

    while(i<niterations)
    {

        /**
         * Calculations 
        */
        for (int iBlock=0;(iBlock<nIterationsOutput) and (i<niterations);iBlock++)
        {
            std::swap(phi_new,phi_old);
            compute_jacobi_timer.start();
            compute_jacobi(phi_new,phi_old,rho,nFields);
            compute_jacobi_timer.stop();
            i++;

        }

        /**
         * Output 
        */
        for(int iField=0;iField<nFields;iField++)
        {
            print_to_file(phi_new[iField].get_data(),&current_grid,"phi" + std::to_string(iField) + "_" + std::to_string(i) + ".dat" );
        }

    }

    total_time_timer.stop();

    std::cout << "Finalize" << std::endl;

    std::cout << total_time_timer << std::endl;
    std::cout << compute_jacobi_timer << std::endl;
    std::cout << apply_periodic_bc_timer << std::endl;
}
