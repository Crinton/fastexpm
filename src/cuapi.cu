#include "cuapi.h"

// Specializations for float
cublasStatus_t cublasAPI<float>::I_amax(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    return cublasIsamax_v2(handle, n, x, incx, result);
}

cublasStatus_t cublasAPI<float>::Axpy(cublasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy) {
    return cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasAPI<float>::Scal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    return cublasSscal_v2(handle, n, alpha, x, incx);
}

cublasStatus_t cublasAPI<float>::Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float* alpha,
                                     const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc) {
    return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specializations for double
cublasStatus_t cublasAPI<double>::I_amax(cublasHandle_t handle, int n, const double* x, int incx, int* result) {
    return cublasIdamax_v2(handle, n, x, incx, result);
}

cublasStatus_t cublasAPI<double>::Axpy(cublasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy) {
    return cublasDaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasAPI<double>::Scal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    return cublasDscal_v2(handle, n, alpha, x, incx);
}

cublasStatus_t cublasAPI<double>::Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const double* alpha,
                                     const double* A, int lda, const double* B, int ldb, const double* beta, double* C, int ldc) {
    return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specializations for cuComplex
cublasStatus_t cublasAPI<cuComplex>::I_amax(cublasHandle_t handle, int n, const cuComplex* x, int incx, int* result) {
    return cublasIcamax_v2(handle, n, x, incx, result);
}

cublasStatus_t cublasAPI<cuComplex>::Axpy(cublasHandle_t handle, int n, const cuComplex* alpha, const cuComplex* x, int incx, cuComplex* y, int incy) {
    return cublasCaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasAPI<cuComplex>::Scal(cublasHandle_t handle, int n, const cuComplex* alpha, cuComplex* x, int incx) {
    return cublasCscal_v2(handle, n, alpha, x, incx);
}

cublasStatus_t cublasAPI<cuComplex>::Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuComplex* alpha,
                                         const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
    return cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specializations for cuDoubleComplex
cublasStatus_t cublasAPI<cuDoubleComplex>::I_amax(cublasHandle_t handle, int n, const cuDoubleComplex* x, int incx, int* result) {
    return cublasIzamax_v2(handle, n, x, incx, result);
}

cublasStatus_t cublasAPI<cuDoubleComplex>::Axpy(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy) {
    return cublasZaxpy_v2(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasAPI<cuDoubleComplex>::Scal(cublasHandle_t handle, int n, const cuDoubleComplex* alpha, cuDoubleComplex* x, int incx) {
    return cublasZscal_v2(handle, n, alpha, x, incx);
}

cublasStatus_t cublasAPI<cuDoubleComplex>::Gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const cuDoubleComplex* alpha,
                                               const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc) {
    return cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// Specializations for cusolverAPI functions
void cusolverAPI<float>::Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, float* A, int lda, int* Lwork) {
    cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

void cusolverAPI<float>::Dn_getrf(cusolverDnHandle_t handle, int m, int n, float* A, int lda, float* Workspace, int* devIpiv, int* devInfo) {
    cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

void cusolverAPI<float>::Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const float* A, int lda, const int* devIpiv, float* B, int ldb, int* devInfo) {
    cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}



// Add similar specializations for double, cuComplex, and cuDoubleComplex

void cusolverAPI<double>::Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, double* A, int lda, int* Lwork) {
    cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

void cusolverAPI<double>::Dn_getrf(cusolverDnHandle_t handle, int m, int n, double* A, int lda, double* Workspace, int* devIpiv, int* devInfo) {
    cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

void cusolverAPI<double>::Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const double* A, int lda, const int* devIpiv, double* B, int ldb, int* devInfo) {
    cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

void cusolverAPI<cuComplex>::Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, int* Lwork) {
    cusolverDnCgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

void cusolverAPI<cuComplex>::Dn_getrf(cusolverDnHandle_t handle, int m, int n, cuComplex* A, int lda, cuComplex* Workspace, int* devIpiv, int* devInfo) {
    cusolverDnCgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

void cusolverAPI<cuComplex>::Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuComplex* A, int lda, const int* devIpiv, cuComplex* B, int ldb, int* devInfo) {
    cusolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}


void cusolverAPI<cuDoubleComplex>::Dn_getrf_bufferSize(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, int* Lwork) {
    cusolverDnZgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

void cusolverAPI<cuDoubleComplex>::Dn_getrf(cusolverDnHandle_t handle, int m, int n, cuDoubleComplex* A, int lda, cuDoubleComplex* Workspace, int* devIpiv, int* devInfo) {
    cusolverDnZgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

void cusolverAPI<cuDoubleComplex>::Dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const cuDoubleComplex* A, int lda, const int* devIpiv, cuDoubleComplex* B, int ldb, int* devInfo) {
    cusolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}