cmake_minimum_required(VERSION 3.10)
project(my_slam_project)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# Set ORB_SLAM3 path
set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ORB_SLAM3)

# Add ORB_SLAM3 cmake modules
list(APPEND CMAKE_MODULE_PATH ${ORB_SLAM3_DIR}/cmake)

# Find packages
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)

# Include directories - simplified
include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2
    ${ORB_SLAM3_DIR}/Thirdparty/g2o
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Libraries
set(LIBS 
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${ORB_SLAM3_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${ORB_SLAM3_DIR}/Thirdparty/g2o/lib/libg2o.so
    ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
    pthread
    dl
    /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
    ${realsense2_LIBRARY}
)

# RealSense executable
add_executable(slam_webcam_simple src/main_webcam_simple.cpp)
add_executable(slam_realsense_simple src/main_realsense_simple.cpp)
add_executable(main_mp4_mono src/main_mp4_mono.cpp)
# add_executable(main_relocalization_mp4 src/main_relocalization_mp4.cpp)
add_executable(relocalization src/main.cpp src/relocalization.cpp)
add_executable(test_map src/test_map.cpp)
add_executable(test src/test.cpp)

target_link_libraries(slam_webcam_simple ${LIBS})
target_link_libraries(slam_realsense_simple ${LIBS})
target_link_libraries(main_mp4_mono ${LIBS})
# target_link_libraries(main_relocalization_mp4 ${LIBS})
target_link_libraries(relocalization ${LIBS})
target_link_libraries(test_map ${LIBS})
target_link_libraries(test ${LIBS})


