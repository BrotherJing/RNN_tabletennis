cmake_minimum_required(VERSION 2.8)
project(RNN_tabletennis)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cc TrajPredict.cpp)
set(EXECUTABLE tabletennis)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

find_package(TensorFlow REQUIRED)
find_package(Eigen REQUIRED)
find_package(Protobuf REQUIRED)

set(PROJECT_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIRS} ${Eigen_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS})
set(PROJECT_LIBRARIES ${TensorFlow_LIBRARIES} ${PROTOBUF_LIBRARIES})

include_directories(${PROJECT_INCLUDE_DIRS} /home/jing/local/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(trajpred SHARED TrajPredict.cpp)
target_link_libraries(trajpred ${PROJECT_LIBRARIES})

add_executable(${EXECUTABLE} ${SOURCE_FILES})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
