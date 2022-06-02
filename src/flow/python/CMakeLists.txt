###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# Specify the sources of the pure python and compiled portions of our module.
SET(flow_py_python_sources  py_src/__init__.py py_src/helpers.py)
SET(flow_py_cpp_sources     flow_python.cpp)


# Setup our module
PYTHON_ADD_HYBRID_MODULE(NAME flow_python
                         DEST_DIR python-modules
                         PY_MODULE_DIR flow
                         PY_SETUP_FILE setup.py
                         PY_SOURCES  "${flow_py_python_sources}"
                         SOURCES      ${flow_py_cpp_sources})

# link with the proper libs (beyond python)
target_link_libraries(flow_python ascent_flow)


# TODO: ADD SUPPORT FOR SUPPORT PYTHON_MODULE_INSTALL_PREFIX
# install the capi header so other python modules can use it
# support alt install dir for python module via PYTHON_MODULE_INSTALL_PREFIX
# if(PYTHON_MODULE_INSTALL_PREFIX)
#     install(FILES ${flow_py_headers} DESTINATION ${PYTHON_MODULE_INSTALL_PREFIX}/flow/)
# else()
#     install(FILES ${flow_py_headers} DESTINATION python-modules/flow/)
# endif()




