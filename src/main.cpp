#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "expm.h"
#include "iomatrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// int main(void)
// {
//     int M;
//     std::cin>>M;
//     size_t matrixCount = M*M*sizeof(float);
//     float *A,*B;
//     A = (float *)malloc(matrixCount);
//     B = (float *)malloc(matrixCount);
//     readMatrix("./matrix-A.txt",A,M,M);
//     B = expm(A,M);
//     printf("%lf\n",B[0]);
//     saveMatrix(B,M,M,"./result.txt");
//     free(B);
//     return 0; 
// }
namespace py = pybind11;


py::array_t<float> expm_float(py::array_t<float> &arr_a)
{
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != shape[1])
    {
        throw std::runtime_error("Error: The matrix is not square (rows != cols).");
    }

    int M = shape[0];
    int matrixCount = M*M*sizeof(float);
    float *ptrA = (float *)bufA.ptr;
    float *result = (float *)malloc(matrixCount);

    result = expm(ptrA ,M);
    py::capsule free_when_done(result, [](void* f){});
    py::array_t<float> output_array(
      { M*M },
      { sizeof(float) },
      result,
      free_when_done);
    return output_array;
}

py::array_t<double> expm_double(py::array_t<double> &arr_a)
{
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != shape[1])
    {
        throw std::runtime_error("Error: The matrix is not square (rows != cols).");
    }

    int M = shape[0];
    int matrixCount = M*M*sizeof(double);
    double *ptrA = (double *)bufA.ptr;
    double *result = (double *)malloc(matrixCount);

    result = expm(ptrA ,M);
    py::capsule free_when_done(result, [](void* f){});
    py::array_t<double> output_array(
      { M*M },
      { sizeof(double) },
      result,
      free_when_done);
    return output_array;
}

py::array_t<std::complex<float>> expm_complex(py::array_t<std::complex<float>> &arr_a)
{
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != shape[1])
    {
        throw std::runtime_error("Error: The matrix is not square (rows != cols).");
    }

    int M = shape[0];
    int matrixCount = M*M*sizeof(std::complex<float>);
    std::complex<float> *ptrA = (std::complex<float> *)bufA.ptr;
    std::complex<float> *result = (std::complex<float> *)malloc(matrixCount);

    result = expm(ptrA ,M);
    py::capsule free_when_done(result, [](void* f){});
    py::array_t<std::complex<float>> output_array(
      { M*M },
      { sizeof(std::complex<float>) },
      result,
      free_when_done);
    return output_array;
}

py::array_t<std::complex<double>> expm_doublecomplex(py::array_t<std::complex<double>> &arr_a)
{
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != shape[1])
    {
        throw std::runtime_error("Error: The matrix is not square (rows != cols).");
    }

    int M = shape[0];
    int matrixCount = M*M*sizeof(std::complex<double>);
    std::complex<double> *ptrA = (std::complex<double> *)bufA.ptr;
    std::complex<double> *result = (std::complex<double> *)malloc(matrixCount);

    result = expm(ptrA ,M);
    py::capsule free_when_done(result, [](void* f){});
    py::array_t<std::complex<double>> output_array(
      { M*M },
      { sizeof(std::complex<double>) },
      result,
      free_when_done);
    return output_array;
}


PYBIND11_MODULE(fastexpm,m)
{
    m.doc() = "pybind11 expm plugin";
    m.def("expm_float",&expm_float,"A function which expm for double matrix");
    m.def("expm_double",&expm_double,"A function which expm for double matrix");
    m.def("expm_complex",&expm_complex,"A function which expm for double matrix");
    m.def("expm_doublecomplex",&expm_doublecomplex,"A function which expm for double matrix");
}