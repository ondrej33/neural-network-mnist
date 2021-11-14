cmake_minimum_required(VERSION 3.7...3.21)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

SET(GCC_COVERAGE_COMPILE_FLAGS "-g")

project(PV021-Project VERSION 1.0
                  DESCRIPTION "Neural Networks Project"
                  LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(main src/main.cpp include/main.hpp include/lingebra.hpp include/neural_network.hpp include/activation_fns.hpp include/error_fns.hpp include/labels.hpp
)