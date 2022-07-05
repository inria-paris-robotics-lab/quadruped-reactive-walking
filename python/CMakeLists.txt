set(${PY_NAME}_SOURCES
  main.cpp
  Joystick.cpp
  Params.cpp
)

add_library(${PY_NAME} SHARED ${${PY_NAME}_SOURCES})
target_link_libraries(${PY_NAME} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_NAME} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(${PY_NAME} PROPERTIES
  INSTALL_RPATH "\$ORIGIN/../../.."
  LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
  PREFIX ""
  SUFFIX ${PYTHON_EXT_SUFFIX})
TARGET_LINK_BOOST_PYTHON(${PY_NAME} PRIVATE)
install(TARGETS ${PY_NAME} DESTINATION ${${PY_NAME}_INSTALL_DIR})

set(${PY_NAME}_PYTHON
  __init__.py
  Controller.py
  main_solo12_control.py
  WB_MPC_Wrapper.py
  )

set(${PY_NAME}_WB_MPC_PYTHON
  __init__.py
  CrocoddylOCP.py
  CasadiOCP.py
  ShootingNode.py
  )

set(${PY_NAME}_TOOLS_PYTHON
  gamepadClient.py
  __init__.py
  LoggerControl.py
  PyBulletSimulator.py
  qualisysClient.py
  )

foreach(python ${${PY_NAME}_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME} ${python})
endforeach()

foreach(python ${${PY_NAME}_WB_MPC_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/WB_MPC ${python})
endforeach()

foreach(python ${${PY_NAME}_TOOLS_PYTHON})
  PYTHON_INSTALL_ON_SITE(${PY_NAME}/tools ${python})
endforeach()
