#include <iostream>
#include <iomanip>
#include <chrono>
#include <mkl.h>
#include <mkl_vsl.h>
#include "mkl_cblas.h"

int main() {
    std::chrono::time_point<std::chrono::system_clock> start, end; 

    int n, i;
    double *x, *y, *r;
    double res;
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_MCG31, 1337);

    n = 300000000;
    x = (double*)mkl_malloc(n*sizeof(double),64);
    y = (double*)mkl_malloc(n*sizeof(double),64);

    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,n,x,0.,3.);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,n,y,0.,3.);

    std::cout<< "MKL: " << std::endl;
    start = std::chrono::system_clock::now(); 
    res = cblas_ddot(n, x, 1, y, 1);
    end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end - start; 
    std::cout<< "res: " << std::setprecision(12) << res << std::endl;
    std::cout<< "Time of cblas_ddot: " << elapsed_seconds.count() * 1000. << " milliseconds"  << std::endl;

    r = (double*)mkl_malloc(n*sizeof(double),64);
    std::cout<< "MKL: " << std::endl;
    start = std::chrono::system_clock::now(); 
    vdAdd(n, x, y, r);
    end = std::chrono::system_clock::now(); 
    elapsed_seconds = end - start; 
    std::cout<< "res: " << std::setprecision(12) << res << std::endl;
    std::cout<< "Time of cblas_ddot: " << elapsed_seconds.count() * 1000. << " milliseconds"  << std::endl;
    
    mkl_free(x);
    mkl_free(y);
    mkl_free(r);
    return 0;
}
