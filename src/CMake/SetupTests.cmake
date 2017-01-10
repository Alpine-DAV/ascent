###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Strawman. 
# 
# For details, see: http://software.llnl.gov/strawman/.
# 
# Please also read strawman/LICENSE
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


set(UNIT_TEST_BASE_LIBS gtest_main gtest)

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

    add_executable( ${arg_TEST} ${arg_TEST}.cpp )

    target_link_libraries( ${arg_TEST} ${UNIT_TEST_BASE_LIBS})
    target_link_libraries( ${arg_TEST} "${arg_DEPENDS_ON}" )
   
    add_test( ${arg_TEST} ${arg_TEST} )

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

    cuda_compile( ${arg_TEST}_O ${arg_TEST}.cu)

    cuda_add_executable(${arg_TEST} ${${arg_TEST}_O})

    target_link_libraries( ${arg_TEST} ${UNIT_TEST_BASE_LIBS})
    target_link_libraries( ${arg_TEST} "${arg_DEPENDS_ON}" )
   
    add_test( ${arg_TEST} ${arg_TEST} )

endfunction()


##------------------------------------------------------------------------------
## - Builds and adds a test that uses gtest and mpi
##
## add_cpp_mpi_test( TEST test NUM_PROCS 2 DEPENDS_ON dep1 dep2... )
##------------------------------------------------------------------------------
function(add_cpp_mpi_test)

    set(options)
    set(singleValueArgs TEST NUM_PROCS)
    set(multiValueArgs DEPENDS_ON)

    # parse our arguments
    cmake_parse_arguments(arg
                         "${options}" 
                         "${singleValueArgs}" 
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Unit Test: ${arg_TEST}")

    # make sure the test can see the mpi headers
    include_directories(${MPI_CXX_INCLUDE_PATH})
    # guard against empty mpi params
    if(NOT "${MPI_CXX_COMPILE_FLAGS}" STREQUAL "")
        set_source_files_properties(${arg_TEST}.cpp PROPERTIES COMPILE_FLAGS  ${MPI_CXX_COMPILE_FLAGS} )
    endif()
    if(NOT "${MPI_CXX_LINK_FLAGS}" STREQUAL "")
        set_source_files_properties(${arg_TEST}.cpp PROPERTIES LINK_FLAGS  ${MPI_CXX_LINK_FLAGS} )
    endif()
    
    
    add_executable( ${arg_TEST} ${arg_TEST}.cpp )

    target_link_libraries( ${arg_TEST} ${UNIT_TEST_BASE_LIBS} )
    target_link_libraries( ${arg_TEST} ${MPI_CXX_LIBRARIES} )
    target_link_libraries( ${arg_TEST} "${arg_DEPENDS_ON}" )

    # setup custom test command to launch the test via mpi
    set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${arg_NUM_PROCS} "./${arg_TEST}")
    add_test(NAME ${arg_TEST} COMMAND ${MPIEXEC} ${test_parameters})

endfunction()


##------------------------------------------------------------------------------
## - Adds a python based unit test
##
## add_python_test( TEST test)
##------------------------------------------------------------------------------
function(add_python_test TEST)
    message(STATUS " [*] Adding Python-based Unit Test: ${TEST}")
    add_test(NAME ${TEST} COMMAND 
             ${PYTHON_EXECUTABLE} -B -m unittest -v ${TEST})
    # make sure python can pick up the modules we built
    set(PYTHON_TEST_PATH "${CMAKE_BINARY_DIR}/python-modules/:${CMAKE_CURRENT_SOURCE_DIR}")
    if(EXTRA_PYTHON_MODULE_DIRS)
        set(PYTHON_TEST_PATH "${EXTRA_PYTHON_MODULE_DIRS}:${PYTHON_TEST_PATH}")
    endif()
    set_property(TEST ${TEST} PROPERTY ENVIRONMENT  "PYTHONPATH=${PYTHON_TEST_PATH}")
endfunction(add_python_test)


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
    set(fortran_driver_source
        ${CMAKE_SOURCE_DIR}/thirdparty_builtin/fruit-3.3.9/gtest_fortran_driver.cpp)

    add_executable( ${arg_TEST} ${arg_TEST}.f ${fortran_driver_source})
    set_target_properties(${arg_TEST} PROPERTIES Fortran_FORMAT "FREE")

    target_link_libraries( ${arg_TEST} fruit)
    target_link_libraries( ${arg_TEST} ${UNIT_TEST_BASE_LIBS})
    target_link_libraries( ${arg_TEST} "${arg_DEPENDS_ON}" )

    add_test( ${arg_TEST} ${arg_TEST})

endmacro(add_fortran_test)



