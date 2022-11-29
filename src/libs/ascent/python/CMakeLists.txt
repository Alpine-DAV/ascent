###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

add_subdirectory(ascent_module)

# install the jupyter bridge kernel client source
# we copy the source because users may want to install
# client into a different python environment
install(DIRECTORY ascent_jupyter_bridge
        DESTINATION share/ascent)

# create scripts that make it easy to launch the proper 
# python interp with the ascent module already in PYTHONPATH
set(_PYMOD_DIR ${PROJECT_BINARY_DIR}/python-modules)

# gen python helper to ${PROJECT_BINARY_DIR}/bin
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_python_with_ascent.sh.in"
                "${PROJECT_BINARY_DIR}/bin/run_python_with_ascent_build.sh" @ONLY)

# gen jupyter notebook helper script to ${PROJECT_BINARY_DIR}/bin
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_jupyter_notebook_with_ascent.sh.in"
                "${PROJECT_BINARY_DIR}/bin/run_jupyter_notebook_with_ascent.sh" @ONLY)

# gen jupyter lab helper script to ${PROJECT_BINARY_DIR}/bin
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_jupyter_lab_with_ascent.sh.in"
                "${PROJECT_BINARY_DIR}/bin/run_jupyter_lab_with_ascent_build.sh" @ONLY)


# gen helper that works with the ascent install
if(PYTHON_MODULE_INSTALL_PREFIX)
    set(_PYMOD_DIR ${PYTHON_MODULE_INSTALL_PREFIX})
else()
    set(_PYMOD_DIR ${CMAKE_INSTALL_PREFIX}/python-modules/)
endif()


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_python_with_ascent.sh.in"
                "${CMAKE_CURRENT_BINARY_DIR}/run_python_with_ascent_install.sh" @ONLY)

unset(_PYMOD_DIR)

# install the script that works for install paths
# 'PROGRAMS' makes sure exec perms are set
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/run_python_with_ascent_install.sh"
        DESTINATION bin
        RENAME run_python_with_ascent.sh)


