cmake_minimum_required(VERSION 3.8)
project(navigation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

# ---------------------------------------------------------------------------
# Core build tools
# ---------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
# ament_cmake_python lets us keep the Python nodes alongside C++ tools.
find_package(ament_cmake_python REQUIRED)

# ---------------------------------------------------------------------------
# C++ dependencies for osa_to_csv
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

set(ORB_SLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../ORB_SLAM3)

# TODO: Uncomment once ORB-SLAM3 provides a CMake config file.
#       If it does not, set ORB_SLAM3_INCLUDE_DIRS and ORB_SLAM3_LIBS
#       manually in a toolchain file or via -D flags at configure time.
#
# find_package(ORB_SLAM3 REQUIRED)

# ---------------------------------------------------------------------------
# Python package installation (preserves ament_python behaviour)
#
# ament_python_install_package() installs the `navigation` Python package
# into the ament install tree so `import navigation` works at runtime.
# Python console scripts (navigation, navigation_ui, fake_data_publisher)
# must also be installed as programs under lib/${PROJECT_NAME}/.
# See the install(PROGRAMS ...) block below.
# ---------------------------------------------------------------------------
ament_python_install_package(${PROJECT_NAME})

# Install Python entry-point scripts with RENAME so ros2 run finds them by
# their logical name (ament_cmake does not process setup.py console_scripts).
install(PROGRAMS navigation/main.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME navigation
)
install(PROGRAMS navigation/ui/map_ui_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME navigation_ui
)
install(PROGRAMS navigation/test/fake_data_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME fake_data_publisher
)
install(PROGRAMS navigation/map_constructor/map_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME map_publisher
)
install(PROGRAMS navigation/test/position_publisher.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME position_publisher
)

# ---------------------------------------------------------------------------
# osa_to_csv — offline .osa → CSV conversion tool
# ---------------------------------------------------------------------------
add_executable(osa_to_csv tools/osa_to_csv.cpp)

target_include_directories(osa_to_csv PRIVATE
  ${ORB_SLAM3_DIR}/include
  ${ORB_SLAM3_DIR}/include/CameraModels
  ${ORB_SLAM3_DIR}
  ${ORB_SLAM3_DIR}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS} 
)

target_compile_options(osa_to_csv PRIVATE
  -Wno-error
)

target_link_libraries(osa_to_csv
  ${ORB_SLAM3_DIR}/lib/libORB_SLAM3.so
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

target_compile_definitions(osa_to_csv PRIVATE
  COMPILEDWITHC11
)

target_link_options(osa_to_csv PRIVATE
  -Wl,--allow-shlib-undefined
  -Wl,--copy-dt-needed-entries
)

# Place the binary in lib/${PROJECT_NAME} so `ros2 run navigation osa_to_csv`
# can locate it through the ament install prefix.
install(TARGETS osa_to_csv
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Shared resources (config, launch files, maps)
# ---------------------------------------------------------------------------
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY navigation/maps/
  DESTINATION share/${PROJECT_NAME}/maps
)

install(FILES navigation/config/locations.yaml
              navigation/config/navigation_params.yaml
  DESTINATION share/${PROJECT_NAME}/config
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
