cmake_minimum_required(VERSION 3.7...3.21)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()



project(PV021-Project VERSION 1.0
                  DESCRIPTION "Neural Networks Project"
                  LANGUAGES CXX)

add_executable(main src/main.cpp include/main.hpp)