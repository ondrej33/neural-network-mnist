cmake_minimum_required(VERSION 3.7...3.21)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(PV021-Project VERSION 1.0
                  DESCRIPTION "Neural Networks Project"
                  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# multiprocessing, for now commented, because its not used in current version
#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#endif()

# specify the mode (if you want the other one, then comment this)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Werror -Wextra -Wpedantic -Wconversion")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(main src/main.cpp include/lingebra.hpp include/neural_network.hpp include/activation_fns.hpp include/layer.hpp include/input_loading.hpp
)