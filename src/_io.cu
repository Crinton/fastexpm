#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "iomatrix.h"

bool readMatrix(const std::string& filePath, float*& matrix, int rows, int cols) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    std::string line;
    std::string token;
    for (int i = 0; i < rows; ++i) {
        std::getline(file, line);
        std::stringstream iss(line);
        for (int j = 0; j < cols; ++j) {
            std::getline(iss, token, ',');
            //std:: cout << token << std::endl;
            matrix[i  + j * cols] = std::stof(token);
        }
    }
    file.close();
    return true;
}


bool readMatrix(const std::string& filePath, double*& matrix, int rows, int cols) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    std::string line;
    std::string token;
    for (int i = 0; i < rows; ++i) {
        std::getline(file, line);
        std::stringstream iss(line);
        for (int j = 0; j < cols; ++j) {
            std::getline(iss, token, ',');
            //std:: cout << token << std::endl;
            matrix[i  + j * cols] = std::stod(token);
        }
    }
    file.close();
    return true;
}

bool saveMatrix(const float* matrix, int rows, int cols, const std::string& filePath) 
{
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    outFile << std::setprecision(17);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[i + j* cols];
            if (j < cols - 1) {
                outFile << ","; // 在元素之间添加逗号，除了每行的最后一个元素
            }
        }
        if (i < rows - 1) {
            outFile << "\n"; // 每行结束后添加换行符
        }
    }

    outFile.close();
    return true;
}


bool saveMatrix(const double* matrix, int rows, int cols, const std::string& filePath) 
{
    std::ofstream outFile(filePath);
    if (!outFile.is_open()) {
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return false;
    }
    outFile << std::setprecision(17);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << matrix[i + j* cols];
            if (j < cols - 1) {
                outFile << ","; // 在元素之间添加逗号，除了每行的最后一个元素
            }
        }
        if (i < rows - 1) {
            outFile << "\n"; // 每行结束后添加换行符
        }
    }

    outFile.close();
    return true;
}