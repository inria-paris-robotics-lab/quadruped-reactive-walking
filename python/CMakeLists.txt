include(${PROJECT_SOURCE_DIR}/cmake/python-helpers.cmake)
if(GENERATE_PYTHON_STUBS)
  include(${PROJECT_SOURCE_DIR}/cmake/stubs.cmake)
endif(GENERATE_PYTHON_STUBS)

set(PY_HEADERS
    ${PROJECT_SOURCE_DIR}/include/qrw/bindings/python.hpp
    ${PROJECT_SOURCE_DIR}/include/qrw/bindings/visitors.hpp
    ${PROJECT_SOURCE_DIR}/include/qrw/bindings/custom-members.hpp
    ${PROJECT_SOURCE_DIR}/include/qrw/bindings/yaml-node.hpp)

set(PY_SOURCES
    module.cpp
    expose-animator.cpp
    expose-estimator.cpp
    expose-filter.cpp
    expose-mpc-interface.cpp
    expose-mpc-result.cpp
    expose-ocp-interface.cpp
    expose-params.cpp)

set(PY_LIB ${${PY_NAME}_LIB})
add_library(${PY_LIB} SHARED ${PY_SOURCES} ${PY_HEADERS})
target_link_libraries(${PY_LIB} PUBLIC ${PROJECT_NAME} ${PYTHON_LIBRARY} eigenpy::eigenpy)
target_include_directories(${PY_LIB} SYSTEM PRIVATE ${PYTHON_INCLUDE_DIR})
set_target_properties(
  ${PY_LIB}
  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PY_NAME}
             PREFIX ""
             SUFFIX ${PYTHON_EXT_SUFFIX}
             VERSION ${PROJECT_VERSION})
if(UNIX)
  get_relative_rpath(${${PY_NAME}_INSTALL_DIR} ${PY_NAME}_INSTALL_RPATH)
  set_target_properties(${PY_LIB} PROPERTIES INSTALL_RPATH "${${PY_NAME}_INSTALL_RPATH}")
endif()
target_link_boost_python(${PY_LIB} PRIVATE)
install(
  TARGETS ${PY_LIB}
  EXPORT ${TARGETS_EXPORT_NAME}
  DESTINATION ${${PY_NAME}_INSTALL_DIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/qrw/bindings DESTINATION include/qrw)

file(
  GLOB ${PY_NAME}_PYTHON
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}
  ${PY_NAME}/*.py)

file(
  GLOB ${PY_NAME}_WB_MPC_PYTHON
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}
  ${PY_NAME}/wb_mpc/*.py)

file(
  GLOB ${PY_NAME}_TOOLS_PYTHON
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}
  ${PY_NAME}/tools/*.py)

file(
  GLOB ${PY_NAME}_OCP
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}
  ${PY_NAME}/ocp_defs/*.py)

if(GENERATE_PYTHON_STUBS)
  load_stubgen()
  generate_stubs(${CMAKE_CURRENT_BINARY_DIR} ${PY_NAME} ${PYTHON_SITELIB} ${PY_LIB})
endif(GENERATE_PYTHON_STUBS)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/wb_mpc
     ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/tools
     ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/ocp_defs)

# util to add python file with their dir as MODULE var
function(add_pyfile pyfile)
  get_filename_component(fname ${pyfile} NAME)
  get_filename_component(fdir ${pyfile} DIRECTORY)
  python_install_on_site(${PY_NAME}/${fdir} ${fname})
  execute_process(
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/${PY_NAME}/${pyfile}
      ${CMAKE_CURRENT_BINARY_DIR}/${PY_NAME}/${pyfile})
endfunction(add_pyfile ${pyfile})

foreach(python ${${PY_NAME}_PYTHON})
  add_pyfile(${python})
endforeach()

foreach(python ${${PY_NAME}_WB_MPC_PYTHON})
  add_pyfile(${python})
endforeach()

foreach(python ${${PY_NAME}_TOOLS_PYTHON})
  add_pyfile(${python})
endforeach()

foreach(python ${${PY_NAME}_OCP})
  add_pyfile(${python})
endforeach()
