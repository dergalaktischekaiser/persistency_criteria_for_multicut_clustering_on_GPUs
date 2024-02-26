cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_CUDA_ARCHITECTURES OFF)
project(PersistencyCriteriaForMulticutClusteringOnGPUs LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")


add_executable(
    PersistencyCriteriaForMulticutClusteringOnGPUs main.cu
    dCOO.cu
    device_vector_initialiser.cu
    input_reader.cu
    PersistencyCriteriaSolver.cu
)
