cmake_minimum_required(VERSION 3.22)
project(test LANGUAGES CXX CUDA)
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c*)
include_directories(${PROJECT_SOURCE_DIR}/include) # PROJECT_SOURCE_DIR 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
find_package(Python REQUIRED)
message(STATUS "python version is ${Python_VERSION}")
find_package(CUDAToolkit REQUIRED)
add_subdirectory(extern/pybind11)


pybind11_add_module(fastexpm  ${SRC})
target_link_libraries(fastexpm  PRIVATE -lcublas -lcusolver)
# add_executable(fastexpm ${SRC})
target_link_libraries(fastexpm PRIVATE CUDA::cublas CUDA::cusolver)