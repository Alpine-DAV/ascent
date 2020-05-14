###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-716457
#
# All rights reserved.
#
# This file is part of Ascent.
#
# For details, see: http://ascent.readthedocs.io/.
#
# Please also read ascent/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
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
    set(_PERF_MPIEXEC_CMD "${_mpiexec} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_MPI_TASKS} ${BLT_MPI_COMMAND_APPEND}")

    # Grab path to clover and bake into the options file
    set(_PERF_CLOVER_CMD  ${CMAKE_BINARY_DIR}/examples/proxies/cloverleaf3d-ref/cloverleaf3d_par)

    configure_file ("opts.json.in"
                    "${CMAKE_CURRENT_BINARY_DIR}/opts.json" @ONLY)
    
    # copy over script and supporting files 
    configure_file ("run_ascent_clover_perf_tests.py"
                    "${CMAKE_CURRENT_BINARY_DIR}/run_ascent_clover_perf_tests.py"
                    COPYONLY)
    set(_PERF_ACTIONS_FILES "ascent_actions_contour_and_render.yaml")

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

