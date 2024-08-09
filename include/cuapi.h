#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuComplex.h>

template<typename T>
struct cublasAPI {
    static cublasStatus_t I_amax(cublasHandle_t handle, int n, const T* x, int incx, int* result);
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);
    static cublasStatus_t Scal(cublasHandle_t handle, int n, const T* alpha, T* x, int incx);
    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const T* alpha,
                               const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc);
};

template<typename T>
struct cusolverAPI {
    static void Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, T* A, int lda, int* Lwork);
    static void Dn_getrf(cusolverDnHandle_t handle, int m, int n, T* A, int lda, T* Workspace, int* devIpiv, int* devInfo);
    static void Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const T* A, int lda, const int* devIpiv, T* B, int ldb, int* devInfo);
};

// Specializations for basic types
template<>
struct cublasAPI<float> {
    static cublasStatus_t I_amax(cublasHandle_t handle, int n, const float* x, int incx, int* result);
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy);
    static cublasStatus_t Scal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha,
                               const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc);
};

template<>
struct cublasAPI<double> {
    static cublasStatus_t I_amax(cublasHandle_t handle, int n, const double* x, int incx, int* result);
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy);
    static cublasStatus_t Scal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha,
                               const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc);
};

template<>
struct cublasAPI<cuComplex> {
    static cublasStatus_t I_amax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result);
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy);
    static cublasStatus_t Scal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx);
    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha,
                               const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc);
};

template<>
struct cublasAPI<cuDoubleComplex> {
    static cublasStatus_t I_amax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result);
    static cublasStatus_t Axpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy);
    static cublasStatus_t Scal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx);
    static cublasStatus_t Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha,
                               const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc);
};

template<>
struct cusolverAPI<float> {
    static void Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork);
    static void Dn_getrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo);
    static void Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo);
};

template<>
struct cusolverAPI<double> {
    static void Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork);
    static void Dn_getrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo);
    static void Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo);
};

template<>
struct cusolverAPI<cuComplex> {
    static void Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* Lwork);
    static void Dn_getrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo);
    static void Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo);
};

template<>
struct cusolverAPI<cuDoubleComplex> {
    static void Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork);
    static void Dn_getrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo);
    static void Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo);
};
