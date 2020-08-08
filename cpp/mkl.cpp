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

    n = 2100000000;
    x = (double*)mkl_malloc(n*sizeof(double),64);
    y = (double*)mkl_malloc(n*sizeof(double),64);

    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,n,x,0.,3.);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,n,y,0.,3.);
    std::cout<< "x: " << std::setprecision(12) << x[0] << ", " << x[1] << ", " << x[2] << std::endl;
    std::cout<< "y: " << std::setprecision(12) << y[0] << ", " << y[1] << ", " << y[2] << std::endl;

    start = std::chrono::system_clock::now(); 
    res = cblas_ddot(n, x, 1, y, 1);
    end = std::chrono::system_clock::now(); 
    std::chrono::duration<double> elapsed_seconds = end - start; 
    std::cout<< "res: " << std::setprecision(12) << res << std::endl;
    std::cout<< "Time of cblas_ddot: " << elapsed_seconds.count() * 1000. << " milliseconds"  << std::endl;


    r = (double*)mkl_malloc(n*sizeof(double),64);
    start = std::chrono::system_clock::now(); 
    vdAdd(n, x, y, r);
    end = std::chrono::system_clock::now(); 
    elapsed_seconds = end - start; 
    std::cout<< "res: " << std::setprecision(12) << r[0] << ", " << r[1] << ", " << r[2] << std::endl;
    std::cout<< "Time of vdAdd: " << elapsed_seconds.count() * 1000. << " milliseconds"  << std::endl;
    mkl_free(x);
    mkl_free(y);
    mkl_free(r);

    int m, k, mk;
    double *a, *b, *c, alpha, beta;

    m = 6000;
    k = 200000;
    mk = m * k;
    a = (double*)mkl_malloc(mk*sizeof(double),64);
    b = (double*)mkl_malloc(k*sizeof(double),64);

    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,mk,a,0.,3.);
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER,stream,k,b,0.,3.);

    c = (double*)mkl_malloc(m*sizeof(double),64);
    for (i=0; i<m; i++) c[i] = 2.;

    alpha = 2.;
    beta = 1.;

    start = std::chrono::system_clock::now();
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, k, alpha, a, k, b, 1, beta, c, 1);
    end = std::chrono::system_clock::now(); 
    elapsed_seconds = end - start; 
    std::cout<< "res: " << std::setprecision(12) << c[0] << ", " << c[1] << ", " << c[2] << std::endl;
    std::cout<< "Time of cblas_dgemv: " << elapsed_seconds.count() * 1000. << " milliseconds"  << std::endl;

    mkl_free(a);
    mkl_free(b);
    mkl_free(c);
    return 0;
}
