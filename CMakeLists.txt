cmake_minimum_required(VERSION 3.16)
project(tartancalib_apriltag_internal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED COMPONENTS filesystem)

set(TARTANCALIB_APRILTAG_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_offline_calibration/ethz_apriltag2")
set(TARTANCALIB_APRILTAG_INTERNAL_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/aslam_cv/aslam_cameras_april")

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

add_library(tartancalib_apriltag_internal_core STATIC
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/ApriltagCanonicalModel.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/ApriltagInternalDetector.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/DoubleSphereCameraModel.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/MultiScaleOuterTagDetector.cpp"
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/src/apriltag_internal/IterativeCoarseCalibrationExperiment.cpp"
)
target_include_directories(tartancalib_apriltag_internal_core PUBLIC
  "${TARTANCALIB_APRILTAG_INTERNAL_DIR}/include"
)
target_link_libraries(tartancalib_apriltag_internal_core PUBLIC
  tartancalib_ethz_apriltag2_core
  Eigen3::Eigen
  ${OpenCV_LIBS}
  Boost::filesystem
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
