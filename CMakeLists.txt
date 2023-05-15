cmake_minimum_required(VERSION 3.25)
project(OpenCL_Programowanie_Kart)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

add_subdirectory(thirdparty)
add_subdirectory(src)
