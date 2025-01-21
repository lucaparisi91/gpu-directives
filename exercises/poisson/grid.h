#ifndef GRID_H
#define GRID_H
#include <cstdio>

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

#endif