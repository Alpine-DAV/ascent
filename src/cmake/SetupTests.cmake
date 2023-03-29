###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

################################################################
# SetupTests.cmake is called after Setup3rdParty.cmake
# Here we check ENABLE_TESTS vs ASCENT_ENABLE_TESTS.
# They should be the same, since we simply cache
# the value of ENABLE_TESTS in ASCENT_ENABLE_TESTS at 
# the very beginning of configure (in CMakeBasics.cmake)
#
# But if Setup3rdParty.cmake imports something that changes
# ENABLE_TESTS, they could be different.
# Trust ASCENT_ENABLE_TESTS
################################################################

if(ASCENT_ENABLE_TESTS AND NOT ENABLE_TESTS)
    set(ENABLE_TESTS ON)
endif()

if(ENABLE_TESTS)
    message(STATUS "Tests are enabled (ENABLE_TESTS=ON)")
else()
    message(STATUS "Tests are disabled (ENABLE_TESTS=OFF)")
endif()


#####################################################################
if(ASCENT_ENABLE_TESTS AND WIN32 AND BUILD_SHARED_LIBS)
    # Copy DLLs into our bin dir so we can satisfy 
    # deps to run tests.
    #
    # Note: There are per target ways to do this, however
    #       all of our many, many tests share these dlls so 
    #       I opted for a single copy step, instead of
    #       trying to track and copy each test. 
    #
    # Show TPL DLL Paths
    message(STATUS "ASCENT_TPL_DLL_PATHS: ${ASCENT_TPL_DLL_PATHS}")
    # glob and gather dlls from all TPL dirs
    set(tpl_all_dlls)
    foreach( tpl_dll_path in ${ASCENT_TPL_DLL_PATHS})
        file(GLOB tpl_glob_dlls ${tpl_dll_path}/*.dll)
        foreach( tpl_dll ${tpl_glob_dlls})
            list(APPEND tpl_all_dlls ${tpl_dll})
        endforeach()
    endforeach()
    add_custom_target(tpl_dlls_dir ALL
                      COMMAND ${CMAKE_COMMAND} -E make_directory
                      ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
    add_custom_target(tpl_dlls ALL
                      COMMAND ${CMAKE_COMMAND} -E copy 
                      ${tpl_all_dlls}
                      ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
    add_dependencies(tpl_dlls tpl_dlls_dir)
endif()

##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest
##
## add_cpp_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cpp_test)

    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Unit Test: ${arg_TEST}")

    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest)

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                    )

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
        set_property(TEST ${arg_TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
    endif()
endfunction()


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest
##
## add_cuda_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cuda_test)

    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding CUDA Unit Test: ${arg_TEST}")

    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest cuda)

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                    )

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
        set_property(TEST ${arg_TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
    endif()

endfunction()


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest and mpi
##
## add_cpp_mpi_test( TEST test NUM_MPI_TASKS 2 DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cpp_mpi_test)

    set(options)
    set(singleValueArgs TEST NUM_MPI_TASKS)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Unit Test: ${arg_TEST}")


    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.cpp ${fortran_driver_source}
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON "${arg_DEPENDS_ON}" gtest mpi)

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST}
                  NUM_MPI_TASKS ${arg_NUM_MPI_TASKS})

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
        set_property(TEST ${arg_TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
    endif()

    ###########################################################################
    # Newer versions of OpenMPI require OMPI_MCA_rmaps_base_oversubscribe=1
    # to run with more tasks than actual cores
    # Since this is an OpenMPI specific env var, it shouldn't interfere
    # with other mpi implementations.
    ###########################################################################
    set_property(TEST ${arg_TEST}
                 APPEND PROPERTY ENVIRONMENT  "OMPI_MCA_rmaps_base_oversubscribe=1")
endfunction()


##------------------------------------------------------------------------------
## - Adds a python based unit test
##
## add_python_test( TEST test)
##------------------------------------------------------------------------------
function(add_python_test TEST)

    message(STATUS " [*] Adding Python-based Unit Test: ${TEST}")
    add_test( NAME ${TEST}
              COMMAND ${PYTHON_EXECUTABLE} -B -m unittest -v ${TEST})
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
    set_property(TEST ${TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")

endfunction(add_python_test)


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest and mpi
##
## add_python_mpi_test( TEST test NUM_MPI_TASKS 2 )
##------------------------------------------------------------------------------
function(add_python_mpi_test)

    set(options)
    set(singleValueArgs TEST NUM_MPI_TASKS)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Python-based MPI Unit Test: ${arg_TEST}")
    set(test_command ${PYTHON_EXECUTABLE} -B -m unittest -v ${arg_TEST})

    # Handle mpi
    if ( ${arg_NUM_MPI_TASKS} )
          set(test_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_MPI_TASKS} ${test_command} )
    endif()

    add_test(NAME ${arg_TEST}
             COMMAND ${test_command} )

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
     set_property(TEST ${arg_TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")

     ###########################################################################
     # Newer versions of OpenMPI require OMPI_MCA_rmaps_base_oversubscribe=1
     # to run with more tasks than actual cores
     # Since this is an OpenMPI specific env var, it shouldn't interfere
     # with other mpi implementations.
     ###########################################################################
     set_property(TEST ${arg_TEST}
                  PROPERTY ENVIRONMENT  "OMPI_MCA_rmaps_base_oversubscribe=1")

endfunction()



##------------------------------------------------------------------------------
## - Adds a fortran based unit test
##
## add_fortran_test( TEST test DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
macro(add_fortran_test)
    set(options)
    set(singleValueArgs TEST)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Fortran Unit Test: ${arg_TEST}")
    blt_add_executable( NAME ${arg_TEST}
                        SOURCES ${arg_TEST}.f90
                        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
                        DEPENDS_ON fruit "${arg_DEPENDS_ON}")

    blt_add_test( NAME ${arg_TEST}
                  COMMAND ${arg_TEST})

endmacro(add_fortran_test)



