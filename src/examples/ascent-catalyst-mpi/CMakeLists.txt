# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Example that shows how to use an installed instance of Ascent in another
# CMake-based build system.
#
# To build:
#  cmake -DAscent_DIR={ascent install path} -B build -S .
#  cp ascent_actions.yaml build
#  cd build
#  make
#  ./ascent_catalyst_mpi_example
#
# In order to run directly in a sub directory below ascent-catalyst_mpi in an ascent install, 
# set Ascent_DIR to ../../..
# 
#  cmake -DAscent_DIR=../../.. -B build -S .
#  cp ascent_actions.yaml build
#  cd build
#  make
#  ./ascent_catalyst_mpi_example
#
###############################################################################

cmake_minimum_required(VERSION 3.9)

project(ascent_catalyst_mpi)

#
# Use CMake's find_package to import ascent's targets
#
# PATHS is just a hint if someone runs this example from the Ascent install
# tree without setting up an environment hint to find Ascent
find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CMAKE_SOURCE_DIR}/../../../)

find_package(MPI REQUIRED COMPONENTS CXX)

find_package(catalyst 2.0 REQUIRED)

# create our example
add_executable(ascent_catalyst_mpi_example ascent_catalyst_mpi_example.cpp)

# link to ascent and catalyst
target_link_libraries(ascent_catalyst_mpi_example ascent::ascent_mpi catalyst::catalyst)

# if cuda is in the mix:
# we need to make sure CUDA_RESOLVE_DEVICE_SYMBOLS is on for our target
# (it propgates some way magically in 3.14, but not in 3.21)
if(CMAKE_CUDA_COMPILER)
    set_property(TARGET ascent_catalyst_mpi_example PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
