#include <iostream>
#ifndef IOMATIX_H
#define IOMATIX_H
bool readMatrix(const std::string& filePath, float*& matrix, int rows, int cols);
bool readMatrix(const std::string& filePath, double*& matrix, int rows, int cols);
bool saveMatrix(const float* matrix, int rows, int cols, const std::string& filePath);
bool saveMatrix(const double* matrix, int rows, int cols, const std::string& filePath);
#endif