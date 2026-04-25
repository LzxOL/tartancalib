cmake_minimum_required(VERSION 3.0.2)
project(aslam_cameras_april)

ADD_DEFINITIONS(-DASLAM_USE_ROS)

find_package(catkin REQUIRED COMPONENTS
  sm_python
  aslam_cameras
  aslam_backend
  aslam_backend_expressions
  aslam_cv_backend
  aslam_time
  ethz_apriltag2
  numpy_eigen
  sm_common
  sm_logging
  sm_boost
  sm_eigen
  sm_kinematics
  python_module
  sparse_block_matrix
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
  src/apriltag_internal/ApriltagInternalDebugVisualization.cpp
  src/apriltag_internal/AslamBackendCalibrationRunner.cpp
  src/apriltag_internal/CalibrationStateBundle.cpp
  src/apriltag_internal/DoubleSphereCameraModel.cpp
  src/apriltag_internal/FrozenRound2BaselinePipeline.cpp
  src/apriltag_internal/JointMeasurementSelection.cpp
  src/apriltag_internal/JointReprojectionCostCore.cpp
  src/apriltag_internal/JointReprojectionMeasurementBuilder.cpp
  src/apriltag_internal/JointReprojectionOptimizer.cpp
  src/apriltag_internal/JointReprojectionResidualEvaluator.cpp
  src/apriltag_internal/KalibrBenchmark.cpp
  src/apriltag_internal/MultiBoardInternalMeasurementRegenerator.cpp
  src/apriltag_internal/MultiBoardOuterBootstrap.cpp
  src/apriltag_internal/MultiScaleOuterTagDetector.cpp
  src/apriltag_internal/OuterDetectionCache.cpp
  src/apriltag_internal/OuterOnlyCameraInitializer.cpp
  src/apriltag_internal/Stage5Runtime.cpp
  src/apriltag_internal/Stage5Benchmark.cpp
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

add_executable(run_multi_board_outer_bootstrap
  src/run_multi_board_outer_bootstrap_main.cpp
)
target_link_libraries(run_multi_board_outer_bootstrap ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_multi_board_outer_bootstrap PUBLIC cxx_std_14)

add_executable(run_multi_board_internal_regeneration
  src/run_multi_board_internal_regeneration_main.cpp
)
target_link_libraries(run_multi_board_internal_regeneration ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_multi_board_internal_regeneration PUBLIC cxx_std_14)

add_executable(run_joint_reprojection_measurement_prep
  src/run_joint_reprojection_measurement_prep_main.cpp
)
target_link_libraries(run_joint_reprojection_measurement_prep ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_joint_reprojection_measurement_prep PUBLIC cxx_std_14)

add_executable(run_joint_reprojection_optimization
  src/run_joint_reprojection_optimization_main.cpp
)
target_link_libraries(run_joint_reprojection_optimization ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_joint_reprojection_optimization PUBLIC cxx_std_14)

add_executable(run_compare_stage5_with_kalibr
  src/run_compare_stage5_with_kalibr_main.cpp
)
target_link_libraries(run_compare_stage5_with_kalibr ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_compare_stage5_with_kalibr PUBLIC cxx_std_14)

add_executable(run_stage5_benchmark
  src/run_stage5_benchmark_main.cpp
)
target_link_libraries(run_stage5_benchmark ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_stage5_benchmark PUBLIC cxx_std_14)

add_executable(run_stage5_backend
  src/run_stage5_backend_main.cpp
)
target_link_libraries(run_stage5_backend ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_stage5_backend PUBLIC cxx_std_14)

add_executable(run_iterative_apriltag_internal_experiment
  src/run_iterative_apriltag_internal_experiment_main.cpp
)
target_link_libraries(run_iterative_apriltag_internal_experiment ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_iterative_apriltag_internal_experiment PUBLIC cxx_std_14)

add_executable(run_compare_internal_seed_methods
  src/run_compare_internal_seed_methods_main.cpp
)
target_link_libraries(run_compare_internal_seed_methods ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_compare_internal_seed_methods PUBLIC cxx_std_14)

add_executable(run_compare_outer_geometry
  src/run_compare_outer_geometry_main.cpp
)
target_link_libraries(run_compare_outer_geometry ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_compare_outer_geometry PUBLIC cxx_std_14)

add_executable(run_compare_outer_adaptive_subpix_vs_sphere
  src/run_compare_outer_adaptive_subpix_vs_sphere_main.cpp
)
target_link_libraries(run_compare_outer_adaptive_subpix_vs_sphere ${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_compile_features(run_compare_outer_adaptive_subpix_vs_sphere PUBLIC cxx_std_14)

##################
## Installation ##
##################

install(TARGETS ${PROJECT_NAME} detect_apriltag_internal run_multi_board_outer_bootstrap run_multi_board_internal_regeneration run_joint_reprojection_measurement_prep run_joint_reprojection_optimization run_compare_stage5_with_kalibr run_stage5_benchmark run_stage5_backend run_iterative_apriltag_internal_experiment run_compare_internal_seed_methods run_compare_outer_geometry run_compare_outer_adaptive_subpix_vs_sphere
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
