cmake_minimum_required(VERSION 3.7...3.21)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(PV021-Project VERSION 1.0
                  DESCRIPTION "Neural Networks Project"
                  LANGUAGES CXX)

# specify always release (if you want debug, then comment this)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(main src/main.cpp include/main.hpp include/lingebra.hpp include/neural_network.hpp include/activation_fns.hpp include/layer.hpp include/input_loading.hpp
)