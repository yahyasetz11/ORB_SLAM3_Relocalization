cmake_minimum_required(VERSION 3.10)
project(orb_slam3_relocalization)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# ── ROS2 ──────────────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nlohmann_json REQUIRED)

# ── Non-ROS dependencies ──────────────────────────────────────────────────────
set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ORB_SLAM3)
list(APPEND CMAKE_MODULE_PATH ${ORB_SLAM3_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 3.3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2 REQUIRED)

# ── Include directories ───────────────────────────────────────────────────────
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

# ── Shared native libraries ───────────────────────────────────────────────────
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

# ── Existing standalone executables (unchanged) ───────────────────────────────
add_executable(slam_webcam_simple src/main_webcam_simple.cpp)
target_link_libraries(slam_webcam_simple ${LIBS})

add_executable(slam_realsense_simple src/main_realsense_simple.cpp)
target_link_libraries(slam_realsense_simple ${LIBS})

add_executable(main_mp4_mono src/main_mp4_mono.cpp)
target_link_libraries(main_mp4_mono ${LIBS})

add_executable(relocalization src/main.cpp src/relocalization.cpp)
target_link_libraries(relocalization ${LIBS})

add_executable(test_map src/test_map.cpp)
target_link_libraries(test_map ${LIBS})

# ── ROS2 node: map_creator_node ───────────────────────────────────────────────
add_executable(map_creator_node src/map_creator_node.cpp)
target_link_libraries(map_creator_node ${LIBS})
ament_target_dependencies(map_creator_node rclcpp geometry_msgs)

# ── ROS2 node: relocalization_node ───────────────────────────────────────────
add_executable(relocalization_node src/relocalization_node.cpp src/relocalization.cpp)
target_link_libraries(relocalization_node ${LIBS} nlohmann_json::nlohmann_json)
ament_target_dependencies(relocalization_node rclcpp geometry_msgs std_msgs sensor_msgs cv_bridge)

# ── Install ROS2 targets ──────────────────────────────────────────────────────
install(TARGETS
    map_creator_node
    relocalization_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
