if (GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif (GENERATE_PYTHON_STUBS)

set(${PY_NAME}_SOURCES
  main.cpp
  Joystick.cpp
  Params.cpp
  Estimator.cpp
  Filter.cpp
)

set(PY_LIB ${${PY_NAME}_LIB})
message(STATUS "PY_LIB = ${PY_LIB}")
add_library(${PY_LIB} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_LIB} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_LIB} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(${PY_LIB} PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})
TARGET_LINK_BOOST_PYTHON(${PY_LIB} PRIVATE)
install(TARGETS ${PY_LIB} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  __init__.py
  Controller.py
  main_solo12_control.py
  WB_MPC_Wrapper.py
  )

set(${PY_NAME}_WB_MPC_PYTHON
  __init__.py
  BenchmarkCrocoddylOCP.py
  ocp_abstract.py
  ocp_proxddp.py
  ocp_crocoddyl.py
  ocp_casadi.py
  ShootingNode.py
  problem_data.py
  Target.py
  )

set(${PY_NAME}_TOOLS_PYTHON
  gamepadClient.py
  __init__.py
  LoggerControl.py
  pybullet_sim.py
  Utils.py
  qualisys_client.py
  kinematics_utils.py
  Interpolator.py
  )

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME} ${PYTHON_SITELIB})
endif(GENERATE_PYTHON_STUBS)

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/${python}
      ${${PROJECT_NAME}_BINARY_DIR}/python/${PY_NAME}/${python})
endforeach()

foreach(python ${${PY_NAME}_WB_MPC_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/WB_MPC ${python})
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/WB_MPC/${python}
      ${${PROJECT_NAME}_BINARY_DIR}/python/${PY_NAME}/WB_MPC/${python})
endforeach()

foreach(python ${${PY_NAME}_TOOLS_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/tools ${python})
  EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/tools/${python}
      ${${PROJECT_NAME}_BINARY_DIR}/python/${PY_NAME}/tools/${python})
endforeach()
