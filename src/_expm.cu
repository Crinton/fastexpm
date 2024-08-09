#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <iomanip>
#include <cuComplex.h>
#include <time.h>
#include <cmath>
#include <utility>
#include <type_traits>
#include "expm.h"
#include "util.h"
#include "cuapi.h"
void checkCudaError(cudaError_t error) {
    if (error == cudaSuccess) {
    } else {
        const char *errorStr = cudaGetErrorString(error);
        printf("CUDA Error: %s\n", errorStr);
    }
}

void checkCublasStatus(cublasStatus_t status) {
    if (status == CUBLAS_STATUS_SUCCESS) {

    } else {
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                printf("CUBLAS library not initialized\n");
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                printf("Resource allocation failed\n");
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                printf("An invalid value was used as an argument\n");
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                printf("An unsupported CUDA architecture was used\n");
                break;
            case CUBLAS_STATUS_MAPPING_ERROR:
                printf("An access to GPU memory space failed\n");
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                printf("The GPU program failed to execute\n");
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                printf("An internal operation failed\n");
                break;
            default:
                printf("An unknown error occurred\n");
                break;
        }
    }
}

// template <typename T>
// double trace(T *A,const int M)
// {
//     T tr = 0;
//     for(int i=0; i<M; ++i)
//     {
//         tr += A[i*M + i];
//     }
//     return tr;
// }

void CuComplexToStdComplex_Array(cuComplex *A, std::complex<float> *B, int M)
{
    cudaMemcpy(B,A,M*M*sizeof(cuComplex),cudaMemcpyDeviceToHost);
    for (int i=0;i < M*M; ++i)
    {
        B[i] = std::complex<float>(A[i].x, A[i].y);
    }
}

void CuComplexToStdComplex_Array(cuDoubleComplex *A, std::complex<double> *B, int M)
{
    cudaMemcpy(B,A,M*M*sizeof(cuDoubleComplex),cudaMemcpyDeviceToHost);
    for (int i=0;i < M*M; ++i)
    {
        B[i] = std::complex<double>(A[i].x, A[i].y);
    }
}

template <typename T>
cuComplex TocuComplex(T x){
    return make_cuComplex(x,0.0f);
}

template <typename T>
cuDoubleComplex TocuDoubleComplex(T x){
    return make_cuDoubleComplex(x,0.0f);
}

static __inline__ cuDoubleComplex cuCexp(cuDoubleComplex x)
{
	double factor = exp(x.x);
	return make_cuDoubleComplex(factor * cos(x.y), factor * sin(x.y));
}

static __inline__ cuComplex cuCexp(cuComplex x)
{
	double factor = exp(x.x);
	return make_cuComplex(factor * cos(x.y), factor * sin(x.y));
}


std::pair<float, int> pre_processing(float *A, float *d_A, int M) //预处理的结果是d_A
{
    double theta = 5.371920351148152;
    size_t matrixCount = M*M*sizeof(float);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    checkCudaError(cudaMemcpy(d_A, A, matrixCount, cudaMemcpyHostToDevice));
    float A_trace,mu,*d_I;
    float L1_norm;
    cudaMalloc((void**)&d_I, M*M*sizeof(float));
    A_trace = trace(A,M); // A_trace 是 std::complex
    mu = - A_trace / (float)M; // mu = - trace(A) / M  
    eye(d_I,M);
    // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    // mu需要转化一下类型
    checkCublasStatus(cublasAPI<float>::Axpy(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    checkCudaError(cudaFree(d_I));
    L1_norm = matrix_L1_norm(d_A,M); // 得到转化后的A的第一范数
    cudaDeviceSynchronize();
    int s = (int)std::ceil(std::log2(L1_norm/theta));
    float scala = 1.0f / std::pow(2,s);

    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&scala,d_A,1)); //  A = (A / (2**s))
    checkCublasStatus(cublasAPI<float>::Scal(cublasH,M*M,&scala,d_A,1));
    cublasDestroy_v2(cublasH);
    return std::make_pair(mu,s);
}

std::pair<double, int> pre_processing(double *A, double *d_A, int M) //预处理的结果是d_A
{
    double theta = 5.371920351148152;
    size_t matrixCount = M*M*sizeof(double);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    checkCudaError(cudaMemcpy(d_A, A, matrixCount, cudaMemcpyHostToDevice));
    double A_trace,mu,*d_I;
    double L1_norm;
    cudaMalloc((void**)&d_I, M*M*sizeof(double));
    A_trace = trace(A,M); // A_trace 是 std::complex
    mu = - A_trace / (double)M; // mu = - trace(A) / M  
    eye(d_I,M);
    // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    // mu需要转化一下类型
    checkCublasStatus(cublasAPI<double>::Axpy(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    checkCudaError(cudaFree(d_I));
    L1_norm = matrix_L1_norm(d_A,M); // 得到转化后的A的第一范数
    cudaDeviceSynchronize();
    int s = (int)std::ceil(std::log2(L1_norm/theta));
    double scala = 1.0f / std::pow(2,s);

    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&scala,d_A,1)); //  A = (A / (2**s))
    checkCublasStatus(cublasAPI<double>::Scal(cublasH,M*M,&scala,d_A,1));
    cublasDestroy_v2(cublasH);
    return std::make_pair(mu,s);
}

std::pair<cuComplex, int> pre_processing(std::complex<float> *A, cuComplex *d_A, int M) //预处理的结果是d_A
{
    double theta = 5.371920351148152;
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    checkCudaError(cudaMemcpy(d_A, A, M*M*sizeof(cuComplex), cudaMemcpyHostToDevice));
    std::complex<float> A_trace,mu_host;
    cuComplex *d_I;
    double L1_norm;
    cudaMalloc((void**)&d_I, M*M*sizeof(cuComplex));
    A_trace = trace(A,M); // A_trace 是 std::complex
    mu_host = - A_trace / std::complex<float>((float)M,0.0f); // mu = - trace(A) / M 
    cuComplex mu = make_cuComplex(mu_host.real(),mu_host.imag());
    eye(d_I,M);
    // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    // mu需要转化一下类型
    checkCublasStatus(cublasAPI<cuComplex>::Axpy(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    checkCudaError(cudaFree(d_I));
    L1_norm = matrix_L1_norm(d_A,M); // 得到转化后的A的第一范数
    cudaDeviceSynchronize();
    int s = (int)std::ceil(std::log2(L1_norm/theta));
    double scala_tmp = 1.0f / std::pow(2,s);
    cuComplex scala = make_cuComplex(scala_tmp,0.0f);
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&scala,d_A,1)); //  A = (A / (2**s))
    checkCublasStatus(cublasAPI<cuComplex>::Scal(cublasH,M*M,&scala,d_A,1));
    cublasDestroy_v2(cublasH);
    return std::make_pair(mu,s);
}

std::pair<cuDoubleComplex, int> pre_processing(std::complex<double> *A, cuDoubleComplex *d_A, int M) //预处理的结果是d_A
{
    double theta = 5.371920351148152;
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    checkCudaError(cudaMemcpy(d_A, A, M*M*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    std::complex<double> A_trace,mu_host;
    cuDoubleComplex *d_I;
    double L1_norm;
    cudaMalloc((void**)&d_I, M*M*sizeof(cuDoubleComplex));
    A_trace = trace(A,M); // A_trace 是 std::complex
    mu_host = - A_trace / std::complex<double>((double)M,0.0f); // mu = - trace(A) / M 
    cuDoubleComplex mu = make_cuDoubleComplex(mu_host.real(),mu_host.imag());
    eye(d_I,M);
    // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    // mu需要转化一下类型
    checkCublasStatus(cublasAPI<cuDoubleComplex>::Axpy(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    checkCudaError(cudaFree(d_I));
    L1_norm = matrix_L1_norm(d_A,M); // 得到转化后的A的第一范数
    cudaDeviceSynchronize();
    int s = (int)std::ceil(std::log2(L1_norm/theta));
    double scala_tmp = 1.0f / std::pow(2,s);
    cuDoubleComplex scala = make_cuDoubleComplex(scala_tmp,0.0f);
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&scala,d_A,1)); //  A = (A / (2**s))
    checkCublasStatus(cublasAPI<cuDoubleComplex>::Scal(cublasH,M*M,&scala,d_A,1));
    cublasDestroy_v2(cublasH);
    return std::make_pair(mu,s);
}


float* pade_appromixmate(float *d_A, int M)
{   
    int m = 14;
    size_t matrixCount = M*M*sizeof(float);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;;
    cusolverDnCreate(&cusolverH);

    checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverDnSetStream(cusolverH, stream);
    cublasSetStream_v2(cublasH,stream);
    float poly[m] = {1,0.5,0.11999999731779099,0.018333332613110542,0.0019927537068724632,0.00016304348537232727,
                                1.035196692100726e-05, 5.175983233129955e-07, 2.0431513566525008e-08,6.306022705717595e-10,1.48377004840414e-11,2.529153491597966e-13,
                                2.8101705462199623e-15,1.5440497506703088e-17};
    float *d_matrix_tmp, *d_P, *d_Q,*ggtmp;
    cudaMallocAsync((void**)&d_matrix_tmp, matrixCount,stream);
    cudaMallocAsync((void**)&d_P, matrixCount,stream);
    cudaMallocAsync((void**)&d_Q, matrixCount,stream);
    cudaMallocAsync((void**)&ggtmp, matrixCount,stream);
    eye(d_matrix_tmp,M);
    //  d_P <- I; d_Q <- I
    cudaStreamSynchronize(stream);
    checkCudaError(cudaMemcpyAsync(d_P, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    checkCudaError(cudaMemcpyAsync(d_Q, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    cudaStreamSynchronize(stream);
    //cudaMemcpy(d_B, B, M*M * sizeof(float), cudaMemc"yHostToDevice);
    //这一步是直接覆盖的 result_P = poly[0] * torch.eye(A.shape[0],dtype=A.dtype).cuda()
    checkCublasStatus(cublasAPI<float>::Scal(cublasH,M*M,&poly[0],d_P,1));
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_Q,1));
    float alpha=1.0f,beta=0.0f;
    float alpha_add;
    for(int i=1;i<m;++i)
    {
        //matrix_tmp = matrix_tmp @ A 
        cudaStreamSynchronize(stream);
        // cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M);
        checkCublasStatus(cublasAPI<float>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M));
        cudaStreamSynchronize(stream);
        checkCudaError(cudaMemcpyAsync(d_matrix_tmp,ggtmp,matrixCount,cudaMemcpyDeviceToDevice,stream));
        // result_P = result_P + poly[i] * matrix_tmp
        // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1)); // 这里利用向量加法实现矩阵加法
        checkCublasStatus(cublasAPI<float>::Axpy(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1));
        //result_Q = result_Q + ((-1)**i) * poly[i]* matrix_tmp
        alpha_add = ((i % 2 == 0) ? 1.0 : -1.0) * poly[i];
        checkCublasStatus(cublasAPI<float>::Axpy(cublasH, M*M, &alpha_add, ggtmp, 1, d_Q, 1)); 
    }
    checkCudaError(cudaFreeAsync(d_A,stream));
    checkCudaError(cudaFreeAsync(d_matrix_tmp,stream));
    checkCudaError(cudaFreeAsync(ggtmp,stream));
    checkCublasStatus(cublasDestroy_v2(cublasH));
    int lda = M;
    int ldb = M;

    /*
    以下参数为LU分解求解Linear System的参数设置
    */
    int info = 0;
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */
    int lwork = 0;            /* size of workspace */
    float *d_work = nullptr; /* device workspace for getrf */
    // const int pivot_on = 1;
    // if (pivot_on)
    // {
    //     printf("pivot is on : compute P*A = L*U \n");
    // }
    // else
    // {
    //     printf("pivot is off: compute A = L*U (not numerically stable)\n");
    // } 
    /* step 2: copy A to device */
    checkCudaError(cudaMallocAsync((void **)(&d_Ipiv), sizeof(int) * M,stream));
    checkCudaError(cudaMallocAsync((void **)(&d_info), sizeof(int),stream));
    /* step 3: query working space of getrf */
    // (cusolverDnDgetrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork));
    cusolverAPI<float>::Dn_getrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork);
    checkCudaError(cudaMalloc((void **)(&d_work), sizeof(float) * lwork));
    /* step 4: LU factorization */

    // (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info));
    cusolverAPI<float>::Dn_getrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info);

    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    (cudaStreamSynchronize(stream));
    // if (0 > info)
    // {
    //     printf("%d-th parameter is wrong \n", -info);
    //     exit(1);
    // }
    // if (pivot_on)
    // {
    //     printf("pivoting sequence, matlab base-1\n");
    // }
    /*
     * step 5: solve A*X = B
     */
    int * Ipiv = (int *)malloc(M*sizeof(int));
    cudaMemcpy(Ipiv,d_Ipiv,M*sizeof(int),cudaMemcpyDeviceToHost);

    // (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info));
    cusolverAPI<float>::Dn_getrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info);
    (cudaStreamSynchronize(stream));
    /* free resources */
    cudaDeviceSynchronize();
    checkCudaError(cudaFree(d_Q));
    checkCudaError(cudaFree(d_Ipiv));
    checkCudaError(cudaFree(d_info));
    checkCudaError(cudaFree(d_work));
    (cusolverDnDestroy(cusolverH));
    return d_P;
}

double* pade_appromixmate(double *d_A, int M)
{   
    int m = 14;
    size_t matrixCount = M*M*sizeof(double);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;;
    cusolverDnCreate(&cusolverH);

    checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverDnSetStream(cusolverH, stream);
    cublasSetStream_v2(cublasH,stream);
    double poly[m] = {1,0.5,0.11999999731779099,0.018333332613110542,0.0019927537068724632,0.00016304348537232727,
                                1.035196692100726e-05, 5.175983233129955e-07, 2.0431513566525008e-08,6.306022705717595e-10,1.48377004840414e-11,2.529153491597966e-13,
                                2.8101705462199623e-15,1.5440497506703088e-17};
    double *d_matrix_tmp, *d_P, *d_Q,*ggtmp;
    cudaMallocAsync((void**)&d_matrix_tmp, matrixCount,stream);
    cudaMallocAsync((void**)&d_P, matrixCount,stream);
    cudaMallocAsync((void**)&d_Q, matrixCount,stream);
    cudaMallocAsync((void**)&ggtmp, matrixCount,stream);
    eye(d_matrix_tmp,M);
    //  d_P <- I; d_Q <- I
    cudaStreamSynchronize(stream);
    checkCudaError(cudaMemcpyAsync(d_P, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    checkCudaError(cudaMemcpyAsync(d_Q, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    cudaStreamSynchronize(stream);
    //cudaMemcpy(d_B, B, M*M * sizeof(float), cudaMemc"yHostToDevice);
    //这一步是直接覆盖的 result_P = poly[0] * torch.eye(A.shape[0],dtype=A.dtype).cuda()
    checkCublasStatus(cublasAPI<double>::Scal(cublasH,M*M,&poly[0],d_P,1));
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_Q,1));
    double alpha=1.0f,beta=0.0f;
    double alpha_add;
    for(int i=1;i<m;++i)
    {
        //matrix_tmp = matrix_tmp @ A 
        cudaStreamSynchronize(stream);
        // cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M);
        checkCublasStatus(cublasAPI<double>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M));
        cudaStreamSynchronize(stream);
        checkCudaError(cudaMemcpyAsync(d_matrix_tmp,ggtmp,matrixCount,cudaMemcpyDeviceToDevice,stream));
        // result_P = result_P + poly[i] * matrix_tmp
        // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1)); // 这里利用向量加法实现矩阵加法
        checkCublasStatus(cublasAPI<double>::Axpy(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1));
        //result_Q = result_Q + ((-1)**i) * poly[i]* matrix_tmp
        alpha_add = ((i % 2 == 0) ? 1.0 : -1.0) * poly[i];
        checkCublasStatus(cublasAPI<double>::Axpy(cublasH, M*M, &alpha_add, ggtmp, 1, d_Q, 1)); 
    }
    checkCudaError(cudaFreeAsync(d_A,stream));
    checkCudaError(cudaFreeAsync(d_matrix_tmp,stream));
    checkCudaError(cudaFreeAsync(ggtmp,stream));
    checkCublasStatus(cublasDestroy_v2(cublasH));
    int lda = M;
    int ldb = M;

    /*
    以下参数为LU分解求解Linear System的参数设置
    */
    int info = 0;
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */
    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */
    // const int pivot_on = 1;
    // if (pivot_on)
    // {
    //     printf("pivot is on : compute P*A = L*U \n");
    // }
    // else
    // {
    //     printf("pivot is off: compute A = L*U (not numerically stable)\n");
    // } 
    /* step 2: copy A to device */
    checkCudaError(cudaMallocAsync((void **)(&d_Ipiv), sizeof(int) * M,stream));
    checkCudaError(cudaMallocAsync((void **)(&d_info), sizeof(int),stream));
    /* step 3: query working space of getrf */
    // (cusolverDnDgetrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork));
    cusolverAPI<double>::Dn_getrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork);
    checkCudaError(cudaMalloc((void **)(&d_work), sizeof(double) * lwork));
    /* step 4: LU factorization */

    // (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info));
    cusolverAPI<double>::Dn_getrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info);

    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    (cudaStreamSynchronize(stream));
    // if (0 > info)
    // {
    //     printf("%d-th parameter is wrong \n", -info);
    //     exit(1);
    // }
    // if (pivot_on)
    // {
    //     printf("pivoting sequence, matlab base-1\n");
    // }
    /*
     * step 5: solve A*X = B
     */
    int * Ipiv = (int *)malloc(M*sizeof(int));
    cudaMemcpy(Ipiv,d_Ipiv,M*sizeof(int),cudaMemcpyDeviceToHost);

    // (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info));
    cusolverAPI<double>::Dn_getrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info);
    (cudaStreamSynchronize(stream));
    /* free resources */
    cudaDeviceSynchronize();
    checkCudaError(cudaFree(d_Q));
    checkCudaError(cudaFree(d_Ipiv));
    checkCudaError(cudaFree(d_info));
    checkCudaError(cudaFree(d_work));
    (cusolverDnDestroy(cusolverH));
    return d_P;
}

cuComplex* pade_appromixmate(cuComplex *d_A, int M)
{   
    int m = 14;
    size_t matrixCount = M*M*sizeof(cuComplex);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;;
    cusolverDnCreate(&cusolverH);

    checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverDnSetStream(cusolverH, stream);
    cublasSetStream_v2(cublasH,stream);
    double poly_tmp[m] = {1,0.5,0.11999999731779099,0.018333332613110542,0.0019927537068724632,0.00016304348537232727,
                                1.035196692100726e-05, 5.175983233129955e-07, 2.0431513566525008e-08,6.306022705717595e-10,1.48377004840414e-11,2.529153491597966e-13,
                                2.8101705462199623e-15,1.5440497506703088e-17};
    cuComplex poly[m];
    for (int i=0; i<m; ++i){
        poly[i] = TocuComplex(poly_tmp[i]);
    }

    cuComplex *d_matrix_tmp, *d_P, *d_Q,*ggtmp;
    cudaMallocAsync((void**)&d_matrix_tmp, matrixCount,stream);
    cudaMallocAsync((void**)&d_P, matrixCount,stream);
    cudaMallocAsync((void**)&d_Q, matrixCount,stream);
    cudaMallocAsync((void**)&ggtmp, matrixCount,stream);
    eye(d_matrix_tmp,M);
    //  d_P <- I; d_Q <- I
    cudaStreamSynchronize(stream);
    checkCudaError(cudaMemcpyAsync(d_P, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    checkCudaError(cudaMemcpyAsync(d_Q, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    cudaStreamSynchronize(stream);
    //cudaMemcpy(d_B, B, M*M * sizeof(float), cudaMemc"yHostToDevice);
    //这一步是直接覆盖的 result_P = poly[0] * torch.eye(A.shape[0],dtype=A.dtype).cuda()
    checkCublasStatus(cublasAPI<cuComplex>::Scal(cublasH,M*M,&poly[0],d_P,1));
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_Q,1));
    cuComplex alpha= TocuComplex(1.0f),beta=TocuComplex(0.0f);
    cuComplex alpha_add;
    for(int i=1;i<m;++i)
    {
        //matrix_tmp = matrix_tmp @ A 
        cudaStreamSynchronize(stream);
        // cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M);
        checkCublasStatus(cublasAPI<cuComplex>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M));
        cudaStreamSynchronize(stream);
        checkCudaError(cudaMemcpyAsync(d_matrix_tmp,ggtmp,matrixCount,cudaMemcpyDeviceToDevice,stream));
        // result_P = result_P + poly[i] * matrix_tmp
        // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1)); // 这里利用向量加法实现矩阵加法
        checkCublasStatus(cublasAPI<cuComplex>::Axpy(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1));
        //result_Q = result_Q + ((-1)**i) * poly[i]* matrix_tmp
        alpha_add = cuCmulf(TocuComplex((i % 2 == 0) ? 1.0 : -1.0), poly[i]);
        checkCublasStatus(cublasAPI<cuComplex>::Axpy(cublasH, M*M, &alpha_add, ggtmp, 1, d_Q, 1)); 
    }
    checkCudaError(cudaFreeAsync(d_A,stream));
    checkCudaError(cudaFreeAsync(d_matrix_tmp,stream));
    checkCudaError(cudaFreeAsync(ggtmp,stream));
    checkCublasStatus(cublasDestroy_v2(cublasH));
    int lda = M;
    int ldb = M;

    /*
    以下参数为LU分解求解Linear System的参数设置
    */
    int info = 0;
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */
    int lwork = 0;            /* size of workspace */
    cuComplex *d_work = nullptr; /* device workspace for getrf */
    // const int pivot_on = 1;
    // if (pivot_on)
    // {
    //     printf("pivot is on : compute P*A = L*U \n");
    // }
    // else
    // {
    //     printf("pivot is off: compute A = L*U (not numerically stable)\n");
    // } 
    /* step 2: copy A to device */
    checkCudaError(cudaMallocAsync((void **)(&d_Ipiv), sizeof(int) * M,stream));
    checkCudaError(cudaMallocAsync((void **)(&d_info), sizeof(int),stream));
    /* step 3: query working space of getrf */
    // (cusolverDnDgetrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork));
    cusolverAPI<cuComplex>::Dn_getrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork);
    checkCudaError(cudaMalloc((void **)(&d_work), sizeof(cuComplex) * lwork));
    /* step 4: LU factorization */

    // (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info));
    cusolverAPI<cuComplex>::Dn_getrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info);

    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    (cudaStreamSynchronize(stream));
    // if (0 > info)
    // {
    //     printf("%d-th parameter is wrong \n", -info);
    //     exit(1);
    // }
    // if (pivot_on)
    // {
    //     printf("pivoting sequence, matlab base-1\n");
    // }
    /*
     * step 5: solve A*X = B
     */
    int * Ipiv = (int *)malloc(M*sizeof(int));
    cudaMemcpy(Ipiv,d_Ipiv,M*sizeof(int),cudaMemcpyDeviceToHost);

    // (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info));
    cusolverAPI<cuComplex>::Dn_getrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info);
    (cudaStreamSynchronize(stream));
    /* free resources */
    cudaDeviceSynchronize();
    checkCudaError(cudaFree(d_Q));
    checkCudaError(cudaFree(d_Ipiv));
    checkCudaError(cudaFree(d_info));
    checkCudaError(cudaFree(d_work));
    (cusolverDnDestroy(cusolverH));
    return d_P;
}

cuDoubleComplex* pade_appromixmate(cuDoubleComplex *d_A, int M)
{   
    int m = 14;
    size_t matrixCount = M*M*sizeof(cuDoubleComplex);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;;
    cusolverDnCreate(&cusolverH);

    checkCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cusolverDnSetStream(cusolverH, stream);
    cublasSetStream_v2(cublasH,stream);
    double poly_tmp[m] = {1,0.5,0.11999999731779099,0.018333332613110542,0.0019927537068724632,0.00016304348537232727,
                                1.035196692100726e-05, 5.175983233129955e-07, 2.0431513566525008e-08,6.306022705717595e-10,1.48377004840414e-11,2.529153491597966e-13,
                                2.8101705462199623e-15,1.5440497506703088e-17};
    cuDoubleComplex poly[m];
    for (int i=0; i<m; ++i){
        poly[i] = TocuDoubleComplex(poly_tmp[i]);
    }

    cuDoubleComplex *d_matrix_tmp, *d_P, *d_Q,*ggtmp;
    cudaMallocAsync((void**)&d_matrix_tmp, matrixCount,stream);
    cudaMallocAsync((void**)&d_P, matrixCount,stream);
    cudaMallocAsync((void**)&d_Q, matrixCount,stream);
    cudaMallocAsync((void**)&ggtmp, matrixCount,stream);
    eye(d_matrix_tmp,M);
    //  d_P <- I; d_Q <- I
    cudaStreamSynchronize(stream);
    checkCudaError(cudaMemcpyAsync(d_P, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    checkCudaError(cudaMemcpyAsync(d_Q, d_matrix_tmp, matrixCount, cudaMemcpyDeviceToDevice,stream));
    cudaStreamSynchronize(stream);
    //cudaMemcpy(d_B, B, M*M * sizeof(float), cudaMemc"yHostToDevice);
    //这一步是直接覆盖的 result_P = poly[0] * torch.eye(A.shape[0],dtype=A.dtype).cuda()
    checkCublasStatus(cublasAPI<cuDoubleComplex>::Scal(cublasH,M*M,&poly[0],d_P,1));
    // checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_Q,1));
    cuDoubleComplex alpha= TocuDoubleComplex(1.0f),beta=TocuDoubleComplex(0.0f);
    cuDoubleComplex alpha_add;
    for(int i=1;i<m;++i)
    {
        //matrix_tmp = matrix_tmp @ A 
        cudaStreamSynchronize(stream);
        // cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M);
        checkCublasStatus(cublasAPI<cuDoubleComplex>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M));
        cudaStreamSynchronize(stream);
        checkCudaError(cudaMemcpyAsync(d_matrix_tmp,ggtmp,matrixCount,cudaMemcpyDeviceToDevice,stream));
        // result_P = result_P + poly[i] * matrix_tmp
        // checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1)); // 这里利用向量加法实现矩阵加法
        checkCublasStatus(cublasAPI<cuDoubleComplex>::Axpy(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1));
        //result_Q = result_Q + ((-1)**i) * poly[i]* matrix_tmp
        alpha_add = cuCmul(TocuDoubleComplex((i % 2 == 0) ? 1.0 : -1.0), poly[i]);
        checkCublasStatus(cublasAPI<cuDoubleComplex>::Axpy(cublasH, M*M, &alpha_add, ggtmp, 1, d_Q, 1)); 
    }
    checkCudaError(cudaFreeAsync(d_A,stream));
    checkCudaError(cudaFreeAsync(d_matrix_tmp,stream));
    checkCudaError(cudaFreeAsync(ggtmp,stream));
    checkCublasStatus(cublasDestroy_v2(cublasH));
    int lda = M;
    int ldb = M;

    /*
    以下参数为LU分解求解Linear System的参数设置
    */
    int info = 0;
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */
    int lwork = 0;            /* size of workspace */
    cuDoubleComplex *d_work = nullptr; /* device workspace for getrf */
    // const int pivot_on = 1;
    // if (pivot_on)
    // {
    //     printf("pivot is on : compute P*A = L*U \n");
    // }
    // else
    // {
    //     printf("pivot is off: compute A = L*U (not numerically stable)\n");
    // } 
    /* step 2: copy A to device */
    checkCudaError(cudaMallocAsync((void **)(&d_Ipiv), sizeof(int) * M,stream));
    checkCudaError(cudaMallocAsync((void **)(&d_info), sizeof(int),stream));
    /* step 3: query working space of getrf */
    // (cusolverDnDgetrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork));
    cusolverAPI<cuDoubleComplex>::Dn_getrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork);
    checkCudaError(cudaMalloc((void **)(&d_work), sizeof(cuDoubleComplex) * lwork));
    /* step 4: LU factorization */

    // (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info));
    cusolverAPI<cuDoubleComplex>::Dn_getrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info);

    (cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    (cudaStreamSynchronize(stream));
    // if (0 > info)
    // {
    //     printf("%d-th parameter is wrong \n", -info);
    //     exit(1);
    // }
    // if (pivot_on)
    // {
    //     printf("pivoting sequence, matlab base-1\n");
    // }
    /*
     * step 5: solve A*X = B
     */
    int * Ipiv = (int *)malloc(M*sizeof(int));
    cudaMemcpy(Ipiv,d_Ipiv,M*sizeof(int),cudaMemcpyDeviceToHost);

    // (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info));
    cusolverAPI<cuDoubleComplex>::Dn_getrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info);
    (cudaStreamSynchronize(stream));
    /* free resources */
    cudaDeviceSynchronize();
    checkCudaError(cudaFree(d_Q));
    checkCudaError(cudaFree(d_Ipiv));
    checkCudaError(cudaFree(d_info));
    checkCudaError(cudaFree(d_work));
    (cusolverDnDestroy(cusolverH));
    return d_P;
}


void undo_preprocessing(float *d_P, float *B, int M ,int s, float mu)
{
    float alpha = 1.0f; float beta = 0.0f; //scala = 1/ scala;
    size_t matrixCount = M*M*sizeof(float);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    float *ggtmp;
    cudaMalloc((void**)&ggtmp, matrixCount);
    

    for(int i=0;i<s;++i)
    {
        //  matrix_tmp = matrix_tmp @ A 
        // cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        cublasAPI<float>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        checkCudaError(cudaMemcpy(d_P,ggtmp,matrixCount,cudaMemcpyDeviceToDevice));
    }
    float alpha_add = std::exp(mu);
    // checkCublasStatus(cublasDscal_v2(cublasH, M*M, &alpha_add,d_P,1)); 
    checkCublasStatus(cublasAPI<float>::Scal(cublasH, M*M, &alpha_add,d_P,1));
    checkCudaError(cudaFree(ggtmp));
    checkCudaError(cudaMemcpy(B,d_P,matrixCount,cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy_v2(cublasH));
}

void undo_preprocessing(double *d_P,double *B, int M ,int s, double mu)
{
    double alpha = 1.0f; double beta = 0.0f; //scala = 1/ scala;
    size_t matrixCount = M*M*sizeof(double);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    double *ggtmp;
    cudaMalloc((void**)&ggtmp, matrixCount);
    

    for(int i=0;i<s;++i)
    {
        //  matrix_tmp = matrix_tmp @ A 
        cublasAPI<double>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        checkCudaError(cudaMemcpy(d_P,ggtmp,matrixCount,cudaMemcpyDeviceToDevice));
    }
    double alpha_add = std::exp(mu);
    checkCublasStatus(cublasAPI<double>::Scal(cublasH, M*M, &alpha_add,d_P,1));
    checkCudaError(cudaFree(ggtmp));
    checkCudaError(cudaMemcpy(B,d_P,matrixCount,cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy_v2(cublasH));
}

void undo_preprocessing(cuComplex *d_P,cuComplex *B, int M ,int s, cuComplex mu)
{
    cuComplex alpha = TocuComplex(1.0f); cuComplex beta = TocuComplex(0.0f); //scala = 1/ scala;
    size_t matrixCount = M*M*sizeof(cuComplex);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    cuComplex *ggtmp;
    cudaMalloc((void**)&ggtmp, matrixCount);
    

    for(int i=0;i<s;++i)
    {
        //  matrix_tmp = matrix_tmp @ A 
        cublasAPI<cuComplex>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        checkCudaError(cudaMemcpy(d_P,ggtmp,matrixCount,cudaMemcpyDeviceToDevice));
    }
    // std::complex<float> exp_mu = std::exp(mu);
    // cuComplex alpha_add = make_cuComplex(exp_mu.real(),exp_mu.imag());
    cuComplex alpha_add = cuCexp(mu);
    checkCublasStatus(cublasAPI<cuComplex>::Scal(cublasH, M*M, &alpha_add,d_P,1));
    checkCudaError(cudaFree(ggtmp));
    checkCudaError(cudaMemcpy(B,d_P,matrixCount,cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy_v2(cublasH));
}

void undo_preprocessing(cuDoubleComplex *d_P,cuDoubleComplex *B, int M ,int s, cuDoubleComplex mu)
{
    cuDoubleComplex alpha = TocuDoubleComplex(1.0f); cuDoubleComplex beta = TocuDoubleComplex(0.0f); //scala = 1/ scala;
    size_t matrixCount = M*M*sizeof(cuDoubleComplex);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    cuDoubleComplex *ggtmp;
    cudaMalloc((void**)&ggtmp, matrixCount);
    

    for(int i=0;i<s;++i)
    {
        //  matrix_tmp = matrix_tmp @ A 
        cublasAPI<cuDoubleComplex>::Gemm(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        checkCudaError(cudaMemcpy(d_P,ggtmp,matrixCount,cudaMemcpyDeviceToDevice));
    }
    // std::complex<double> exp_mu = std::exp(mu);
    // cuDoubleComplex alpha_add = make_cuDoubleComplex(exp_mu.real(),exp_mu.imag());
    cuDoubleComplex alpha_add = cuCexp(mu);
    checkCublasStatus(cublasAPI<cuDoubleComplex>::Scal(cublasH, M*M, &alpha_add,d_P,1));
    checkCudaError(cudaFree(ggtmp));
    checkCudaError(cudaMemcpy(B,d_P,matrixCount,cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy_v2(cublasH));
}

float* expm(float *A,const int M)
{
    float *B, *d_A;
    size_t matrixCount = M*M*sizeof(float);
    checkCudaError(cudaMalloc((void**)&d_A, matrixCount));
    std::pair<float, int> result = pre_processing(A,d_A,M); // d_A为被预处理后的数组指针
    int s; float mu;
    if (&result.first != nullptr) 
    {
        mu = result.first;
        s = result.second;
    }
    // pade 函数会在里面free d_A的显存
    float *d_P = pade_appromixmate(d_A,M); // 返回pade近似的结果 存储在d_P上，pade_appromixmate返回的是一个已经开辟好内存空间的数组指针，所以直接定义一个数值指针接受就好了，不需要额外开辟空间
    B = (float *)malloc(matrixCount); // B是最后返回的数组指针
    undo_preprocessing(d_P,B,M,s,-mu);
    checkCudaError(cudaFree(d_P));
    cudaDeviceSynchronize();
    // checkCudaError(cudaDeviceReset()); // 这一步是重置GPU设备，作为函数调用时，应该注释这一句
    return B;
}

double* expm(double *A,const int M)
{
    double *B, *d_A;
    size_t matrixCount = M*M*sizeof(double);
    checkCudaError(cudaMalloc((void**)&d_A, matrixCount));
    std::pair<double, int> result = pre_processing(A,d_A,M); // d_A为被预处理后的数组指针
    int s; double mu;
    if (&result.first != nullptr) 
    {
        mu = result.first;
        s = result.second;
    }
    // pade 函数会在里面free d_A的显存
    double *d_P = pade_appromixmate(d_A,M); // 返回pade近似的结果 存储在d_P上，pade_appromixmate返回的是一个已经开辟好内存空间的数组指针，所以直接定义一个数值指针接受就好了，不需要额外开辟空间
    B = (double *)malloc(matrixCount); // B是最后返回的数组指针
    undo_preprocessing(d_P,B,M,s,-mu);
    checkCudaError(cudaFree(d_P));
    cudaDeviceSynchronize();
    // checkCudaError(cudaDeviceReset()); // 这一步是重置GPU设备，作为函数调用时，应该注释这一句
    return B;
}

std::complex<float>* expm(std::complex<float> *A,const int M)
{
    cuComplex *d_A;
    size_t matrixCount = M*M*sizeof(std::complex<float>);
    checkCudaError(cudaMalloc((void**)&d_A, matrixCount));
    std::pair<cuComplex, int> result = pre_processing(A,d_A,M); // d_A为被预处理后的数组指针
    int s; cuComplex mu;
    if (&result.first != nullptr) 
    {
        mu = result.first;
        s = result.second;
    }
    mu.x = -mu.x;
    mu.y = -mu.y;
    // pade 函数会在里面free d_A的显存
    cuComplex *d_P = pade_appromixmate(d_A,M); // 返回pade近似的结果 存储在d_P上，pade_appromixmate返回的是一个已经开辟好内存空间的数组指针，所以直接定义一个数值指针接受就好了，不需要额外开辟空间
    
    cuComplex *B1;
    B1 = (cuComplex *)malloc(matrixCount); // B是最后返回的数组指针
    undo_preprocessing(d_P,B1,M,s,mu);
    checkCudaError(cudaFree(d_P));
    cudaDeviceSynchronize();
    std::complex<float> *B = (std::complex<float>*)malloc(matrixCount);
    CuComplexToStdComplex_Array(B1,B,M);
    // checkCudaError(cudaDeviceReset()); // 这一步是重置GPU设备，作为函数调用时，应该注释这一句
    return B;
}


std::complex<double>* expm(std::complex<double> *A,const int M)
{
    
    cuDoubleComplex *d_A;
    size_t matrixCount = M*M*sizeof(std::complex<double>);
    checkCudaError(cudaMalloc((void**)&d_A, matrixCount));
    std::pair<cuDoubleComplex, int> result = pre_processing(A,d_A,M); // d_A为被预处理后的数组指针
    cuDoubleComplex *T = (cuDoubleComplex *)malloc(matrixCount);
    checkCudaError(cudaMemcpy(T,d_A,matrixCount,cudaMemcpyDeviceToHost));
    int s; cuDoubleComplex mu;
    if (&result.first != nullptr) 
    {
        mu = result.first;
        s = result.second;
    }
    mu.x = -mu.x;
    mu.y = -mu.y;
    // pade 函数会在里面free d_A的显存
    cuDoubleComplex *d_P = pade_appromixmate(d_A,M); // 返回pade近似的结果 存储在d_P上，pade_appromixmate返回的是一个已经开辟好内存空间的数组指针，所以直接定义一个数值指针接受就好了，不需要额外开辟空间
    cuDoubleComplex *B1;
    B1 = (cuDoubleComplex *)malloc(matrixCount); // B是最后返回的数组指针
    undo_preprocessing(d_P,B1,M,s,mu);
    checkCudaError(cudaFree(d_P));
    cudaDeviceSynchronize();
    std::complex<double> *B = (std::complex<double>*)malloc(matrixCount);
    CuComplexToStdComplex_Array(B1,B,M);
    // checkCudaError(cudaDeviceReset()); // 这一步是重置GPU设备，作为函数调用时，应该注释这一句
    return B;
}