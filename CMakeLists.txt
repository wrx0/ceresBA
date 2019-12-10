cmake_minimum_required(VERSION 3.15)
project(ceresBA)

set(CMAKE_CXX_STANDARD 11)

add_executable(ceresBA main.cpp)
add_executable(exam helloworld_analytic_diff.cc)

find_package(OpenCV 3.1 REQUIRED )
find_package(Ceres REQUIRED)

include_directories("/usr/include/eigen3")

target_link_libraries(ceresBA ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})


target_link_libraries(exam ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})