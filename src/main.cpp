#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "expm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// int main(void)
// {
//     int M;
//     std::cin>>M;
//     size_t matrixCount = M*M*sizeof(double);
//     double *A,*B;
//     A = (double *)malloc(matrixCount);
//     B = (double *)malloc(matrixCount);
//     readMatrix("./matrix-A.txt",A,M,M);
//     B = expm(A,M);
//     saveMatrix(B,M,M,"./result.txt");
//     free(B);
//     return 0; 
// }
namespace py = pybind11;


// 自定义内存释放函数
void deleter(double* ptr) {
    delete[] ptr;
}


py::array_t<double> np_expm(py::array_t<double> &arr_a)
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


PYBIND11_MODULE(fastexpm,m)
{
    m.doc() = "pybind11 expm plugin";
    m.def("expm",&np_expm,"A function which expm for double matrix");
}