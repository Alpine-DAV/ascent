###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Performance Tests
################################

message(STATUS "Adding performance tests")

if (NOT TARGET cloverleaf3d_par)
  # Do something when target found
  message(STATUS "cloverleaf3d not built, skipping performance tests that leverage cloverleaf3d")
else()

    message(STATUS " [*] Adding performance test: t_perf_clover")

    ########################################################################
    # Here we setup a python script that can run performance tests for us
    # and wire that into ctest
    #
    # this is pretty messy, it could be a macro-ized but
    # avoiding cart before horse, i expect we will want to evolve to
    # something cleaner
    ########################################################################

    #####
    # Gen opts.json file for our python perf test script
    #####

    # Grab proper MPI Command and bake into the options file
    if( ${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.10.0" )
            set(_mpiexec ${MPIEXEC_EXECUTABLE})
    else()
            set(_mpiexec ${MPIEXEC})
    endif()
    set(_PERF_MPIEXEC_CMD "${_mpiexec} ${MPIEXEC_NUMPROC_FLAG} {0} ${BLT_MPI_COMMAND_APPEND}")

    # Grab path to clover and bake into the options file
    set(_PERF_CLOVER_CMD  ${CMAKE_BINARY_DIR}/examples/proxies/cloverleaf3d-ref/cloverleaf3d_par)

    configure_file ("opts.json.in"
                    "${CMAKE_CURRENT_BINARY_DIR}/opts.json" @ONLY)

    # copy over script and supporting files
    configure_file ("run_ascent_clover_perf_tests.py"
                    "${CMAKE_CURRENT_BINARY_DIR}/run_ascent_clover_perf_tests.py"
                    COPYONLY)
    set(_PERF_ACTIONS_FILES "ascent_actions_contour_and_render.yaml")
    list(APPEND _PERF_ACTIONS_FILES "ascent_actions_sampling_and_render.yaml")

    foreach(_ACTION_FILE ${_PERF_ACTIONS_FILES})
        configure_file (${_ACTION_FILE}
                        "${CMAKE_CURRENT_BINARY_DIR}/${_ACTION_FILE}"
                        COPYONLY)
    endforeach()

    # wire up our script as a labeled test
    # working dir is ${CMAKE_CURRENT_BINARY_DIR}
    # if we built python use our python, otherwise defer to system python
    if(PYTHON_FOUND)
        set(py_command "${PYTHON_EXECUTABLE}")
    else()
        set(py_command "python")
    endif()


    set(_test_name "t_perf_clover")

    add_test(NAME ${_test_name}
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
             CONFIGURATIONS Perf
             COMMAND ${py_command} "run_ascent_clover_perf_tests.py")

    set_tests_properties(${_test_name} PROPERTIES LABELS "Perf")

    # setup other env vars (borrowed from our test macros)

    if(PYTHON_FOUND AND ENABLE_PYTHON)
        # make sure python can pick up the modules we built
        # use proper env var path sep for current platform
        if(WIN32)
            set(ENV_PATH_SEP "\\;")
        else()
            set(ENV_PATH_SEP ":")
        endif()

        # if python path is already set -- we need to append to it
        # this is important for running in spack's build-env
        set(PYTHON_TEST_PATH "")

        if(DEFINED ENV{PYTHONPATH})
            set(PYTHON_TEST_PATH "$ENV{PYTHONPATH}${ENV_PATH_SEP}")
        endif()

        set(PYTHON_TEST_PATH "${PYTHON_TEST_PATH}${CMAKE_BINARY_DIR}/python-modules/${ENV_PATH_SEP}${CMAKE_CURRENT_SOURCE_DIR}")
        if(EXTRA_PYTHON_MODULE_DIRS)
            set(PYTHON_TEST_PATH "${EXTRA_PYTHON_MODULE_DIRS}${ENV_PATH_SEP}${PYTHON_TEST_PATH}")
        endif()
        set_property(TEST ${_test_name} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
    endif()

    ###########################################################################
    # Newer versions of OpenMPI require OMPI_MCA_rmaps_base_oversubscribe=1
    # to run with more tasks than actual cores
    # Since this is an OpenMPI specific env var, it shouldn't interfere
    # with other mpi implementations.
    ###########################################################################
    set_property(TEST ${_test_name}
                 APPEND PROPERTY ENVIRONMENT  "OMPI_MCA_rmaps_base_oversubscribe=1")

endif()

