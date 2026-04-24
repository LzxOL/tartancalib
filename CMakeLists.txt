cmake_minimum_required(VERSION 3.16)
project(tartancalib_apriltag_internal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/aslam_optimizer/sparse_block_matrix/cmake"
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS filesystem serialization thread regex)
find_package(SuiteSparse REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

set(TARTANCALIB_APRILTAG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_offline_calibration/ethz_apriltag2")
set(TARTANCALIB_APRILTAG_INTERNAL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_cv/aslam_cameras_april")
set(TARTANCALIB_ASLAM_OPTIMIZER_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_optimizer")
set(TARTANCALIB_ASLAM_CV_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_cv")
set(TARTANCALIB_SM_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/Schweizer-Messer")

include_directories(
  "${CMAKE_CURRENT_SOURCE_DIR}/compat"
  "${TARTANCALIB_APRILTAG_DIR}/include"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/include"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/include"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/include"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/sparse_block_matrix/include"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/include"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cv_backend/include"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_time/include"
  "${TARTANCALIB_SM_DIR}/sm_boost/include"
  "${TARTANCALIB_SM_DIR}/sm_common/include"
  "${TARTANCALIB_SM_DIR}/sm_eigen/include"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/include"
  "${TARTANCALIB_SM_DIR}/sm_logging/include"
  "${TARTANCALIB_SM_DIR}/sm_opencv/include"
  "${TARTANCALIB_SM_DIR}/sm_property_tree/include"
  "${TARTANCALIB_SM_DIR}/sm_random/include"
  "${TARTANCALIB_SM_DIR}/sm_timing/include"
  ${Boost_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
)

file(GLOB TARTANCALIB_APRILTAG_SOURCES CONFIGURE_DEPENDS
  "${TARTANCALIB_APRILTAG_DIR}/src/*.cc"
)

add_library(tartancalib_ethz_apriltag2_core STATIC ${TARTANCALIB_APRILTAG_SOURCES})
target_include_directories(tartancalib_ethz_apriltag2_core PUBLIC
  "${TARTANCALIB_APRILTAG_DIR}/include"
)
target_link_libraries(tartancalib_ethz_apriltag2_core PUBLIC
  Eigen3::Eigen
  ${OpenCV_LIBS}
)

add_library(tartancalib_sm_boost STATIC
  "${TARTANCALIB_SM_DIR}/sm_boost/src/portable_binary_iarchive.cpp"
  "${TARTANCALIB_SM_DIR}/sm_boost/src/portable_binary_oarchive.cpp"
  "${TARTANCALIB_SM_DIR}/sm_boost/src/JobQueue.cpp"
)
target_link_libraries(tartancalib_sm_boost PUBLIC
  Boost::thread
  Threads::Threads
)

add_library(tartancalib_sm_random STATIC
  "${TARTANCALIB_SM_DIR}/sm_random/src/random.cpp"
)
target_link_libraries(tartancalib_sm_random PUBLIC Eigen3::Eigen)

add_library(tartancalib_sm_timing STATIC
  "${TARTANCALIB_SM_DIR}/sm_timing/src/NsecTimeUtilities.cpp"
  "${TARTANCALIB_SM_DIR}/sm_timing/src/Timer.cpp"
)
target_link_libraries(tartancalib_sm_timing PUBLIC Threads::Threads)

add_library(tartancalib_sm_logging STATIC
  "${TARTANCALIB_SM_DIR}/sm_logging/src/Logger.cpp"
  "${TARTANCALIB_SM_DIR}/sm_logging/src/StdOutLogger.cpp"
  "${TARTANCALIB_SM_DIR}/sm_logging/src/LoggingEvent.cpp"
  "${TARTANCALIB_SM_DIR}/sm_logging/src/LoggingGlobals.cpp"
  "${TARTANCALIB_SM_DIR}/sm_logging/src/Formatter.cpp"
  "${TARTANCALIB_SM_DIR}/sm_logging/src/Tokens.cpp"
)
target_link_libraries(tartancalib_sm_logging PUBLIC
  Boost::thread
  Boost::regex
  Threads::Threads
)

add_library(tartancalib_sm_property_tree STATIC
  "${TARTANCALIB_SM_DIR}/sm_property_tree/src/PropertyTree.cpp"
  "${TARTANCALIB_SM_DIR}/sm_property_tree/src/BoostPropertyTreeImplementation.cpp"
  "${TARTANCALIB_SM_DIR}/sm_property_tree/src/BoostPropertyTree.cpp"
  "${TARTANCALIB_SM_DIR}/sm_property_tree/src/PropertyTreeImplementation.cpp"
  "${TARTANCALIB_SM_DIR}/sm_property_tree/src/BoostPropertyTreeSupport.cpp"
)
target_link_libraries(tartancalib_sm_property_tree PUBLIC
  Boost::filesystem
  Threads::Threads
)

add_library(tartancalib_sm_kinematics STATIC
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/quaternion_algebra.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/rotations.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/transformations.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/RotationalKinematics.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/RotationVector.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/EulerAnglesZYX.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/EulerAnglesYawPitchRoll.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/EulerRodriguez.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/Transformation.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/homogeneous_coordinates.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/HomogeneousPoint.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/UncertainTransformation.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/UncertainHomogeneousPoint.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/three_point_methods.cpp"
  "${TARTANCALIB_SM_DIR}/sm_kinematics/src/EulerAnglesZXY.cpp"
)
target_link_libraries(tartancalib_sm_kinematics PUBLIC
  tartancalib_sm_random
  Eigen3::Eigen
  Boost::serialization
  Boost::filesystem
)

add_library(tartancalib_aslam_time STATIC
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_time/src/time.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_time/src/duration.cpp"
)
target_link_libraries(tartancalib_aslam_time PUBLIC Threads::Threads)

add_library(tartancalib_sparse_block_matrix STATIC
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/sparse_block_matrix/src/matrix_structure.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/sparse_block_matrix/src/sparse_helper.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/sparse_block_matrix/src/marginal_covariance_cholesky.cpp"
)
target_link_libraries(tartancalib_sparse_block_matrix PUBLIC
  Eigen3::Eigen
  TBB::tbb
  ${SUITESPARSE_LIBRARIES}
)

add_library(tartancalib_aslam_backend STATIC
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/MEstimatorPolicies.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/JacobianContainer.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/DesignVariable.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/ErrorTerm.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/OptimizationProblemBase.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/Optimizer.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/sparse_matrix_functions.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend/src/OptimizationProblem.cpp"
)
target_compile_definitions(tartancalib_aslam_backend PUBLIC QRSOLVER_DISABLED)
target_link_libraries(tartancalib_aslam_backend PUBLIC
  tartancalib_sparse_block_matrix
  tartancalib_sm_logging
  tartancalib_sm_property_tree
  tartancalib_sm_timing
  Eigen3::Eigen
  Boost::thread
  Threads::Threads
  TBB::tbb
  ${SUITESPARSE_LIBRARIES}
)

add_library(tartancalib_aslam_backend_expressions STATIC
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/RotationExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/RotationExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/RotationQuaternion.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MappedRotationQuaternion.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/Vector2RotationQuaternionExpressionAdapter.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MatrixExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MatrixExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MatrixTransformation.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/TransformationExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/TransformationExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/TransformationBasic.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/EuclideanExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/EuclideanExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/EuclideanPoint.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MappedEuclideanPoint.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/HomogeneousExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/HomogeneousExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/HomogeneousPoint.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MappedHomogeneousPoint.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/ScalarExpression.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/ScalarExpressionNode.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/Scalar.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/EuclideanDirection.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/ErrorTermTransformation.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/ErrorTermEuclidean.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MapTransformation.cpp"
  "${TARTANCALIB_ASLAM_OPTIMIZER_DIR}/aslam_backend_expressions/src/MatrixBasic.cpp"
)
target_link_libraries(tartancalib_aslam_backend_expressions PUBLIC
  tartancalib_aslam_backend
  tartancalib_sm_kinematics
  Threads::Threads
)

add_library(tartancalib_aslam_cameras STATIC
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/CameraGeometryBase.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GlobalShutter.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/RollingShutter.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/NoMask.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/NoDistortion.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/RadialTangentialDistortion.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/EquidistantDistortion.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/FovDistortion.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/ImageMask.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GridCalibrationTargetObservation.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GridCalibrationTargetBase.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GridCalibrationTargetCheckerboard.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GridCalibrationTargetCirclegrid.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/GridDetector.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/Triangulation.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/FrameBase.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/KeypointBase.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/DescriptorBase.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/BackProjection.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/Landmark.cpp"
  "${TARTANCALIB_ASLAM_CV_DIR}/aslam_cameras/src/Image.cpp"
)
target_compile_definitions(tartancalib_aslam_cameras PUBLIC ASLAM_USE_ROS)
target_compile_options(tartancalib_aslam_cameras PUBLIC "-Wno-deprecated-copy")
target_link_libraries(tartancalib_aslam_cameras PUBLIC
  tartancalib_aslam_time
  tartancalib_sm_boost
  tartancalib_sm_logging
  tartancalib_sm_property_tree
  tartancalib_sm_kinematics
  Eigen3::Eigen
  ${OpenCV_LIBS}
  Boost::serialization
  Boost::filesystem
)

add_library(tartancalib_apriltag_internal_core STATIC
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/AslamBackendCalibrationRunner.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/ApriltagCanonicalModel.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/ApriltagInternalDebugVisualization.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/ApriltagInternalDetector.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/CalibrationStateBundle.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/DoubleSphereCameraModel.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/FrozenRound2BaselinePipeline.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/JointMeasurementSelection.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/JointReprojectionCostCore.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/JointReprojectionMeasurementBuilder.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/JointReprojectionOptimizer.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/JointReprojectionResidualEvaluator.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/KalibrBenchmark.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/MultiBoardInternalMeasurementRegenerator.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/MultiBoardOuterBootstrap.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/MultiScaleOuterTagDetector.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/OuterOnlyCameraInitializer.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/Stage5Benchmark.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/IterativeCoarseCalibrationExperiment.cpp"
)
target_include_directories(tartancalib_apriltag_internal_core PUBLIC
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/include"
)
target_link_libraries(tartancalib_apriltag_internal_core PUBLIC
  tartancalib_ethz_apriltag2_core
  tartancalib_aslam_backend
  tartancalib_aslam_backend_expressions
  tartancalib_aslam_cameras
  Eigen3::Eigen
  ${OpenCV_LIBS}
  Boost::filesystem
  Boost::serialization
)

add_executable(detect_apriltag_internal
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/detect_apriltag_internal_main.cpp"
)
target_link_libraries(detect_apriltag_internal PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_iterative_apriltag_internal_experiment
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_iterative_apriltag_internal_experiment_main.cpp"
)
target_link_libraries(run_iterative_apriltag_internal_experiment PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_multi_board_outer_bootstrap
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_multi_board_outer_bootstrap_main.cpp"
)
target_link_libraries(run_multi_board_outer_bootstrap PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_multi_board_internal_regeneration
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_multi_board_internal_regeneration_main.cpp"
)
target_link_libraries(run_multi_board_internal_regeneration PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_joint_reprojection_measurement_prep
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_joint_reprojection_measurement_prep_main.cpp"
)
target_link_libraries(run_joint_reprojection_measurement_prep PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_joint_reprojection_optimization
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_joint_reprojection_optimization_main.cpp"
)
target_link_libraries(run_joint_reprojection_optimization PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_compare_stage5_with_kalibr
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_compare_stage5_with_kalibr_main.cpp"
)
target_link_libraries(run_compare_stage5_with_kalibr PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_stage5_benchmark
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_stage5_benchmark_main.cpp"
)
target_link_libraries(run_stage5_benchmark PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_stage5_backend
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_stage5_backend_main.cpp"
)
target_link_libraries(run_stage5_backend PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_compare_internal_seed_methods
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_compare_internal_seed_methods_main.cpp"
)
target_link_libraries(run_compare_internal_seed_methods PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_compare_outer_geometry
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_compare_outer_geometry_main.cpp"
)
target_link_libraries(run_compare_outer_geometry PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)

add_executable(run_compare_outer_adaptive_subpix_vs_sphere
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/run_compare_outer_adaptive_subpix_vs_sphere_main.cpp"
)
target_link_libraries(run_compare_outer_adaptive_subpix_vs_sphere PRIVATE
  tartancalib_apriltag_internal_core
  ${OpenCV_LIBS}
  Boost::filesystem
)
