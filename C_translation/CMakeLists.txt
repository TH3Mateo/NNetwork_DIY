cmake_minimum_required(VERSION 3.24)
project(C_translation CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_C_STANDARD 23)
add_executable(C_translation main.cu)

set_target_properties(C_translation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcompiler /MD")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Xcompiler /MDd")
