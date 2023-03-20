cmake_minimum_required(VERSION 3.14)

# Project properties
set(PROJECT_NAMESPACE gepetto)
set(PROJECT_NAME quadruped-reactive-walking)
set(PROJECT_DESCRIPTION "Reactive walking control architecture for quadrupeds")
set(PROJECT_URL https://gitlab.laas.fr/${PROJECT_NAMESPACE}/${PROJECT_NAME})

# JRL-cmakemodule setup
include(cmake/base.cmake)
include(cmake/boost.cmake)
include(cmake/apple.cmake)
include(CMakeDependentOption)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Project options
option(BUILD_PYTHON_INTERFACE "Build the python binding" ON)
option(BUILD_JOYSTICK "Build with Gamepad support." ON)
cmake_dependent_option(INSTALL_PYTHON_INTERFACE_ONLY "Install *ONLY* the python bindings"
                       OFF BUILD_PYTHON_INTERFACE OFF)
cmake_dependent_option(
  GENERATE_PYTHON_STUBS "Generate the Python stubs associated to the Python library" ON
  BUILD_PYTHON_INTERFACE ON)
option(SUFFIX_SO_VERSION "Suffix library name with its version" ON)
set(INSTALL_DOCUMENTATION OFF)
option(BUILD_WITH_MQTT_SUPPORT "Add support for MQTT interface" OFF)

# Project configuration
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  set(PROJECT_USE_CMAKE_EXPORT TRUE)
endif()

# Disable -Werror on Unix.
set(CXX_DISABLE_WERROR True)

# Project definition
compute_project_args(PROJECT_ARGS LANGUAGES CXX)
project(${PROJECT_NAME} ${PROJECT_ARGS})
check_minimal_cxx_standard(11 REQUIRED)
apply_default_apple_configuration()

# Project dependencies
set(BOOST_COMPONENTS unit_test_framework)
add_project_dependency(Eigen3 REQUIRED)
add_project_dependency(example-robot-data REQUIRED)

if(BUILD_JOYSTICK)
  add_compile_definitions(QRW_JOYSTICK_SUPPORT)
endif(BUILD_JOYSTICK)

if(BUILD_PYTHON_INTERFACE)
  add_project_dependency(eigenpy REQUIRED)
  string(REGEX REPLACE "-" "_" PY_NAME ${PROJECT_NAME})
  set(${PY_NAME}_LIB "${PY_NAME}_pywrap")
  set(${PY_NAME}_INSTALL_DIR ${PYTHON_SITELIB}/${PY_NAME})
endif()

# Main Library
set(${PROJECT_NAME}_HEADERS
    include/bindings/python.hpp
    include/bindings/yaml-node.hpp
    include/qrw/Types.h
    include/qrw/Params.hpp
    include/qrw/yaml-eigen.hpp
    include/qrw/Animator.hpp
    include/qrw/Estimator.hpp
    include/qrw/Filter.hpp
    include/qrw/ComplementaryFilter.hpp
    include/qrw/utils.hpp)

set(${PROJECT_NAME}_SOURCES src/Params.cpp src/Animator.cpp src/Estimator.cpp
                            src/Filter.cpp src/ComplementaryFilter.cpp)

if(BUILD_JOYSTICK)
  list(APPEND ${PROJECT_NAME}_HEADERS include/qrw/Joystick.hpp)
  list(APPEND ${PROJECT_NAME}_SOURCES src/Joystick.cpp)
endif(BUILD_JOYSTICK)

add_library(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include>)
target_include_directories(${PROJECT_NAME} PUBLIC $<INSTALL_INTERFACE:include/qrw>)
target_link_libraries(${PROJECT_NAME} PUBLIC example-robot-data::example-robot-data)

# Include Eigen3 directories
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)

# Find pinocchio library
add_project_dependency(pinocchio REQUIRED)

# Link pinocchio library
target_link_libraries(${PROJECT_NAME} PUBLIC pinocchio::pinocchio)

# Link odri_control_interface library
# find_package(odri_control_interface REQUIRED)
# target_link_libraries(${PROJECT_NAME}
#                       PUBLIC odri_control_interface::odri_control_interface)

# Link master board library
# target_link_libraries(${PROJECT_NAME} PUBLIC master_board_sdk::master_board_sdk)

add_project_dependency(yaml-cpp CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp)

if(SUFFIX_SO_VERSION)
  set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION ${PROJECT_VERSION})
endif()

target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC CONFIG_SOLO12_YAML="${PROJECT_SOURCE_DIR}/config/config_solo12.yaml")
target_compile_definitions(
  ${PROJECT_NAME}
  PUBLIC WALK_PARAMETERS_YAML="${PROJECT_SOURCE_DIR}/config/walk_parameters.yaml")

# MQTT Interface
if(BUILD_WITH_MQTT_SUPPORT)
  add_subdirectory(paho.mqtt.c)
  add_library(mqtt-interface SHARED include/qrw/mqtt-interface.hpp src/mqtt-interface.cpp)
  target_link_libraries(mqtt-interface PRIVATE paho-mqtt3c paho-mqtt3a)
  target_include_directories(mqtt-interface
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)
endif()

# Python Bindings
if(BUILD_PYTHON_INTERFACE)
  add_subdirectory(python)
endif()

# Installation
if(NOT INSTALL_PYTHON_INTERFACE_ONLY)
  install(
    TARGETS ${PROJECT_NAME}
    EXPORT ${TARGETS_EXPORT_NAME}
    DESTINATION lib)
endif()
