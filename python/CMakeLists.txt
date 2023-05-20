include(${PROJECT_SOURCE_DIR}/cmake/python-helpers.cmake)
if(GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_SOURCES
    module.cpp
    expose-animator.cpp
    expose-estimator.cpp
    expose-filter.cpp
    expose-mpc-interface.cpp
    expose-mpc-result.cpp
    expose-ocp-interface.cpp
    expose-params.cpp)

set(PY_LIB ${${PY_NAME}_LIB})
message(STATUS "PY_LIB = ${PY_LIB}")
add_library(${PY_LIB} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_LIB} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_LIB} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(${PY_LIB} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PY_NAME} PREFIX ""
                                           SUFFIX ${PYTHON_EXT_SUFFIX})
if(UNIX)
  get_relative_rpath(${${PY_NAME}_INSTALL_DIR} ${PY_NAME}_INSTALL_RPATH)
  set_target_properties(${PY_LIB} PROPERTIES INSTALL_RPATH "${${PY_NAME}_INSTALL_RPATH}")
endif()
target_link_boost_python(${PY_LIB} PRIVATE)
install(TARGETS ${PY_LIB} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
    __init__.py
    controller.py
    main_solo12_control.py
    wbmpc_wrapper_abstract.py
    wbmpc_wrapper_sync.py
    wbmpc_wrapper_multiprocess.py
    wbmpc_wrapper_ros_mp.py
    wbmpc_wrapper_ros.py)

set(${PY_NAME}_WB_MPC_PYTHON
    __init__.py
    ocp_abstract.py
    ocp_proxddp.py
    ocp_crocoddyl.py
    ocp_casadi.py
    jump.py
    walking.py
    task_spec.py
    target.py)

set(${PY_NAME}_TOOLS_PYTHON
    gamepadClient.py
    __init__.py
    logger_control.py
    pybullet_sim.py
    utils.py
    qualisys_client.py
    ros_tools.py
    kinematics_utils.py
    meshcat_viewer.py
    Interpolator.py)

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME} ${PYTHON_SITELIB} ${PY_LIB})
endif(GENERATE_PYTHON_STUBS)

foreach(python ${${PY_NAME}_PYTHON})
  python_install_on_site(${PY_NAME} ${python})
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/${python}
      ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/${python})
endforeach()

set(WBMPC_DIR ${PY_NAME}/wb_mpc)

foreach(python ${${PY_NAME}_WB_MPC_PYTHON})
  python_install_on_site(${PY_NAME}/wb_mpc ${python})
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${WBMPC_DIR}/${python}
      ${CMAKE_CURRENT_BINARY_DIR}/${WBMPC_DIR}/${python})
endforeach()

foreach(python ${${PY_NAME}_TOOLS_PYTHON})
  python_install_on_site(${PY_NAME}/tools ${python})
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/tools/${python}
      ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/tools/${python})
endforeach()
