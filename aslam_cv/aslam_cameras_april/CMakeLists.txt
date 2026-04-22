cmake_minimum_required(VERSION 3.0.2)
project(aslam_cameras_april)

ADD_DEFINITIONS(-DASLAM_USE_ROS)

find_package(catkin REQUIRED COMPONENTS
  sm_python
  aslam_cameras
  ethz_apriltag2
  numpy_eigen
  sm_common
  sm_logging
  sm_boost
  sm_eigen
  python_module
)
find_package(OpenCV REQUIRED)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS serialization filesystem)
find_package(Eigen3 REQUIRED)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  INCLUDE_DIRS include
  DEPENDS OpenCV Boost
)

include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})

add_library(${PROJECT_NAME} 
  src/GridCalibrationTargetAprilgrid.cpp
  src/apriltag_internal/ApriltagCanonicalModel.cpp
  src/apriltag_internal/ApriltagInternalDetector.cpp
  src/apriltag_internal/DoubleSphereCameraModel.cpp
  src/apriltag_internal/MultiScaleOuterTagDetector.cpp
  src/apriltag_internal/IterativeCoarseCalibrationExperiment.cpp
)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)

add_python_export_library(${PROJECT_NAME}_python python/aslam_cameras_april
  src/module.cpp
)
target_link_libraries(${PROJECT_NAME}_python ${PROJECT_NAME})

add_executable(detect_apriltag_internal
  src/detect_apriltag_internal_main.cpp
)
target_link_libraries(detect_apriltag_internal ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(detect_apriltag_internal PUBLIC cxx_std_14)

add_executable(run_iterative_apriltag_internal_experiment
  src/run_iterative_apriltag_internal_experiment_main.cpp
)
target_link_libraries(run_iterative_apriltag_internal_experiment ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_iterative_apriltag_internal_experiment PUBLIC cxx_std_14)

##################
## Installation ##
##################

install(TARGETS ${PROJECT_NAME} detect_apriltag_internal run_iterative_apriltag_internal_experiment
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
