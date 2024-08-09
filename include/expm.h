#ifndef EXPM_H
#define EXPM_H
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
float* expm(float *A,const int M);
double* expm(double *A,const int M);
std::complex<float>* expm(std::complex<float> *A,const int M);
std::complex<double>* expm(std::complex<double> *A,const int M);
#endif
