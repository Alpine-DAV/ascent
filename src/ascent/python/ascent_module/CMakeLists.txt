###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

# Specify the sources of the pure python portions of our module.
SET(ascent_py_python_sources  py_src/__init__.py 
                              py_src/ascent.py
                              py_src/bridge_kernel/__init__.py
                              py_src/bridge_kernel/mpi_server.py
                              py_src/bridge_kernel/server.py
                              py_src/bridge_kernel/utils.py
                              )

# built the pure python module
PYTHON_ADD_DISTUTILS_SETUP(NAME ascent_python_py
                           DEST_DIR python-modules
                           PY_MODULE_DIR ascent
                           PY_SETUP_FILE setup.py
                           PY_SOURCES "${ascent_py_python_sources}")

# build non-mpi linked compiled ascent module 
if (ENABLE_SERIAL)
    SET(ascent_py_cpp_sources     ascent_python.cpp)
    # Setup our module
    PYTHON_ADD_COMPILED_MODULE(NAME ascent_python
                               DEST_DIR python-modules
                               PY_MODULE_DIR ascent
                               PY_SOURCES "${ascent_py_cpp_sources}"
                               SOURCES ${ascent_py_cpp_sources})

    # link with the proper libs (beyond python)
    target_link_libraries(ascent_python ascent)
endif()


if(MPI_FOUND)
    # Specify the sources of compiled portions of our module.
    SET(ascent_mpi_py_python_sources  py_src/__init__.py py_src/mpi/ascent_mpi.py)
    SET(ascent_mpi_py_cpp_sources     ascent_mpi_python.cpp)


    # Setup our module
    PYTHON_ADD_COMPILED_MODULE(NAME ascent_mpi_python
                               DEST_DIR python-modules
                               PY_MODULE_DIR ascent/mpi
                               PY_SOURCES "${ascent_mpi_py_python_sources}"
                               SOURCES ${ascent_mpi_py_cpp_sources})

    # link with the proper libs (beyond python)
    target_link_libraries(ascent_mpi_python ascent_mpi)
endif()



