#pragma once


#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include "util.h"
#include "cuapi.h"

template <typename T>
struct NumberToAbs{
    static __host__ __device__ T abs(T x){
        return fabs(x);
    }
};

template <>
struct NumberToAbs<cuComplex> {
    static double __host__ __device__ abs(cuComplex x){
        return cuCabsf(x);
    }
};

template <>
struct NumberToAbs<cuDoubleComplex> {
    static double __host__ __device__ abs(cuDoubleComplex x){
        return cuCabs(x);
    }
};




template<typename T>
__global__ void matrix_l1_norm_kernel(T *d_A, double *d_result, int M)
{

    //extern __shared__ double shared_sum[];
    if (threadIdx.x < M)
    {   
        int col = threadIdx.x;
        double sum = 0.0f; // 这里要适应cuComplex
        //计算每一列的绝对值之和
        for (int row = 0; row < M; ++ row)
        {
            sum = sum + NumberToAbs<T>::abs(d_A[col*M + row]); //我的矩阵是以列为主序的
            // 加法也要适应cuComplex
        }
        d_result[col] = sum;
    }
}





template <typename T>
double matrix_L1_norm(T *d_A, int M)
{
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    double *d_result;
    cudaMalloc((void **)&d_result,M*sizeof(double));
    dim3 blockSize(256);
    dim3 gridSize((M+blockSize.x - 1) / blockSize.x);

    matrix_l1_norm_kernel<T><<<gridSize, blockSize>>>(d_A,d_result,M);
    cudaDeviceSynchronize();
    int L1_norm_idx;
    double *L1_result = (double *)malloc(M*sizeof(double));;
    cublasAPI<double>::I_amax(cublasH,M,d_result,1,&L1_norm_idx);
    cudaMemcpy(L1_result,d_result,M*sizeof(double),cudaMemcpyDeviceToHost);
    // 得到转化后的A的第一范数
    cudaFree(d_result);
    cublasDestroy(cublasH);
    return L1_result[L1_norm_idx-1];
}

float trace(float *A,const int M)
{
    float tr = 0.0f;
    for(int i=0; i<M; ++i)
    {
        tr += A[i*M + i];
    }
    return tr;
}

double trace(double *A,const int M)
{
    double tr = 0.0f;
    for(int i=0; i<M; ++i)
    {
        tr += A[i*M + i];
    }
    return tr;
}

std::complex<float> trace(std::complex<float> *A,const int M)
{
    std::complex<float> tr = 0.0f;
    for(int i=0; i<M; ++i)
    {
        tr += A[i*M + i];
    }
    return tr;
}

std::complex<double> trace(std::complex<double> *A,const int M)
{
    std::complex<double> tr = 0.0f;
    for(int i=0; i<M; ++i)
    {
        tr += A[i*M + i];
    }
    return tr;
}

cuComplex trace(cuComplex *A,const int M)
{
    cuComplex tr = make_cuComplex(0.0f,0.0f);
    for(int i=0; i<M; ++i)
    {
        tr =cuCaddf(tr,A[i*M + i]);
    }
    return tr;
}

cuDoubleComplex trace(cuDoubleComplex *A,const int M)
{
    cuDoubleComplex tr = make_cuDoubleComplex(0.0f,0.0f);
    for(int i=0; i<M; ++i)
    {
        tr =cuCadd(tr,A[i*M + i]);
    }
    return tr;
}

template <typename T> // 这里要特别注意复数
__global__ void setIdentityMatrix(T *d_matrix, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        d_matrix[index * N + index] = 1.0;
    }
}

template <>
__global__ void setIdentityMatrix<cuComplex>(cuComplex *d_matrix, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        d_matrix[index * N + index] = make_cuComplex(1.0f,0);
    }
}

template <>
__global__ void setIdentityMatrix<cuDoubleComplex>(cuDoubleComplex *d_matrix, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        d_matrix[index * N + index] = make_cuDoubleComplex(1.0f,0);
    }
}

template <typename T>
void eye(T *d_matrix, int M)
{
    /*
    在GPU中定义一个单位矩阵
    */
    //内存分配不应该在函数里面
    dim3 Blocksize = 256;
    dim3 gridsize = (M + Blocksize.x -1) / Blocksize.x;
    setIdentityMatrix<T><<<gridsize,Blocksize>>>(d_matrix,M);
}


// template <typename T>
// double matrix_L1_norm(T *d_A, int M);

// template <typename T>
// void eye(T *d_matrix, int M);

// template <typename T>
// double trace(T *A,const int M);
