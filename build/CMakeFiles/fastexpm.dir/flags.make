# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# compile CUDA with /usr/local/cuda-12.2/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_DEFINES = -Dfastexpm_EXPORTS

CUDA_INCLUDES = --options-file CMakeFiles/fastexpm.dir/includes_CUDA.rsp

CUDA_FLAGS =  "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden

CXX_DEFINES = -Dfastexpm_EXPORTS

CXX_INCLUDES = -I/home/hxy/expm/v3/include -isystem /home/hxy/expm/v3/extern/pybind11/include -isystem /home/hxy/anaconda3/include/python3.11 -isystem /usr/local/cuda-12.2/targets/x86_64-linux/include

CXX_FLAGS = -fPIC -fvisibility=hidden -flto -fno-fat-lto-objects

