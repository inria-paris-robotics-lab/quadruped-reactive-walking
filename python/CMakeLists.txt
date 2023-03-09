if(GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

set(${PY_NAME}_SOURCES module.cpp Animator.cpp Params.cpp Estimator.cpp Filter.cpp)

set(PY_LIB ${${PY_NAME}_LIB})
message(STATUS "PY_LIB = ${PY_LIB}")
add_library(${PY_LIB} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_LIB} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_LIB} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(
  ${PY_LIB}
  PROPERTIES INSTALL_RPATH "\$ORIGIN/../../.." LIBRARY_OUTPUT_DIRECTORY ${PY_NAME} PREFIX
                                                                                   ""
             SUFFIX ${PYTHON_EXT_SUFFIX})
target_link_boost_python(${PY_LIB} PRIVATE)
install(TARGETS ${PY_LIB} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON __init__.py controller.py main_solo12_control.py wbmpc_wrapper_abstract.py wbmpc_wrapper_sync.py)

set(${PY_NAME}_WB_MPC_PYTHON
    __init__.py
    ocp_abstract.py
    ocp_proxddp.py
    ocp_crocoddyl.py
    ocp_casadi.py
    problem_data.py
    target.py)

set(${PY_NAME}_TOOLS_PYTHON
    gamepadClient.py
    __init__.py
    logger_control.py
    pybullet_sim.py
    Utils.py
    qualisys_client.py
    kinematics_utils.py
    meshcat_viewer.py
    Interpolator.py)

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME} ${PYTHON_SITELIB})
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
