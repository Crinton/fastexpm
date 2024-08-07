#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <cmath>
#include <utility>
#include "expm.h"
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

void init_I(double *A,int N)
{
    // A矩阵所有元素赋值为0
    for (int i = 0; i < N*N; ++i)
    {
        A[i] = 0;
    }
    for (int i =0; i<N;++i)
    {
        A[i*N+i] = 1;
    }
}
__global__ void addMatrix(double *A, double *B,double alpha,double beta,const int M)
{
    /*
    A = alpha*A + beta*B 
    */
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = tid  + bid * blockDim.x;
    if (idx < M*M)
    {
        A[idx] = alpha*A[idx] + beta*B[idx];
    }
}
__global__ void matrix_l1_norm_kernel(double *d_A, double *d_result, int M)
{

    //extern __shared__ double shared_sum[];
    if (threadIdx.x < M)
    {   
        int col = threadIdx.x;
        double sum = 0.0f;
        //计算每一列的绝对值之和
        for (int row = 0; row < M; ++ row)
        {
            sum+= fabs(d_A[col*M + row]); //我的矩阵是以列为主序的
        }
        d_result[col] = sum;
    }
}
double matrix_L1_norm(double *d_A, int M)
{
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    double *d_result;
    cudaMalloc((void **)&d_result,M*sizeof(double));
    dim3 blockSize(256);
    dim3 gridSize((M+blockSize.x - 1) / blockSize.x);
    matrix_l1_norm_kernel<<<gridSize, blockSize>>>(d_A,d_result,M);
    cudaDeviceSynchronize();
    int L1_norm_idx;
    double *L1_result = (double *)malloc(M*sizeof(double));;
    cublasIdamax_v2(cublasH,M,d_result,1,&L1_norm_idx);
    cudaMemcpy(L1_result,d_result,M*sizeof(double),cudaMemcpyDeviceToHost);
    // 得到转化后的A的第一范数
    cudaFree(d_result);
    cublasDestroy(cublasH);
    return L1_result[L1_norm_idx-1];
}
double trace(double *A,const int M)
{
    double tr = 0;
    for(int i=0; i<M; ++i)
    {
        tr += A[i*M + i];
    }
    return tr;
}
__global__ void setIdentityMatrix(double *d_matrix, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        d_matrix[index * N + index] = 1.0;
    }
}
void eye(double *d_matrix, int M)
{
    /*
    在GPU中定义一个单位矩阵
    */
    //内存分配不应该在函数里面
    dim3 Blocksize = 256;
    dim3 gridsize = (M + Blocksize.x -1) / Blocksize.x;
    setIdentityMatrix<<<gridsize,Blocksize>>>(d_matrix,M);
}
std::pair<double, int> pre_processing(double *A, double *d_A, int M) //预处理的结果是d_A
{
    double theta = 5.371920351148152;
    size_t matrixCount = M*M*sizeof(double);
    cublasHandle_t cublasH;
    cublasCreate_v2(&cublasH);
    checkCudaError(cudaMemcpy(d_A, A, matrixCount, cudaMemcpyHostToDevice));
    double A_trace,mu, L1_norm,*d_I;
    cudaMalloc((void**)&d_I, M*M*sizeof(double));
    A_trace = trace(A,M);
    mu = - A_trace / (double)M; // mu = - trace(A) / M
    eye(d_I,M);
    checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &mu, d_I, 1, d_A, 1));
    checkCudaError(cudaFree(d_I));
    L1_norm = matrix_L1_norm(d_A,M); // 得到转化后的A的第一范数
    cudaDeviceSynchronize();
    int s = (int)std::ceil(std::log2(L1_norm/theta));
    double scala = 1.0f / std::pow(2,s);
    checkCublasStatus(cublasDscal_v2(cublasH,M*M,&scala,d_A,1)); //  A = (A / (2**s))
    cublasDestroy_v2(cublasH);
    return std::make_pair(mu,s);
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
    checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_P,1));
    checkCublasStatus(cublasDscal_v2(cublasH,M*M,&poly[0],d_Q,1));
    double alpha=1.0f,beta=0.0f;
    double alpha_add;
    for(int i=1;i<m;++i)
    {
        //matrix_tmp = matrix_tmp @ A 
        cudaStreamSynchronize(stream);
        cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_matrix_tmp,M,d_A,M,&beta,ggtmp,M);
        cudaStreamSynchronize(stream);
        checkCudaError(cudaMemcpyAsync(d_matrix_tmp,ggtmp,matrixCount,cudaMemcpyDeviceToDevice,stream));
        // result_P = result_P + poly[i] * matrix_tmp
        checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &poly[i], ggtmp, 1, d_P, 1)); // 这里利用向量加法实现矩阵加法
        //result_Q = result_Q + ((-1)**i) * poly[i]* matrix_tmp
        alpha_add = ((i % 2 == 0) ? 1.0 : -1.0) * poly[i];
        checkCublasStatus(cublasDaxpy_v2(cublasH, M*M, &alpha_add, ggtmp, 1, d_Q, 1)); 
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
    const int pivot_on = 1;
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
    (cusolverDnDgetrf_bufferSize(cusolverH, M, M, d_Q, lda, &lwork));
    checkCudaError(cudaMalloc((void **)(&d_work), sizeof(double) * lwork));
    /* step 4: LU factorization */
    if (pivot_on)
    {
        (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, d_Ipiv, d_info));
    }
    else
    {
        (cusolverDnDgetrf(cusolverH, M, M, d_Q, lda, d_work, NULL, d_info));
    }

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
    if (pivot_on)
    {
        (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, d_Ipiv, d_P, ldb, d_info));
    }
    else
    {
        (cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, M, d_Q, lda, NULL, d_P, ldb, d_info));
    }
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
        cublasDgemm_v2(cublasH,CUBLAS_OP_N, CUBLAS_OP_N, M,M,M, &alpha,d_P,M,d_P,M,&beta,ggtmp,M);
        checkCudaError(cudaMemcpy(d_P,ggtmp,matrixCount,cudaMemcpyDeviceToDevice));
    }
    double alpha_add = std::exp(mu);
    checkCublasStatus(cublasDscal_v2(cublasH, M*M, &alpha_add,d_P,1)); 
    checkCudaError(cudaFree(ggtmp));
    checkCudaError(cudaMemcpy(B,d_P,matrixCount,cudaMemcpyDeviceToHost));
    checkCublasStatus(cublasDestroy_v2(cublasH));
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

