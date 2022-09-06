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
#  mkdir build
#  cd build
#  cmake -DAscent_DIR={ascent install path} ../
#  make
#  ./ascent_render_example
#
# In order to run directly in a sub directory below using-with-cmake in an ascent install, 
# set Ascent_DIR to ../../..
# 
#  mkdir build
#  cd build
#  cmake .. -DAscent_DIR=../../..
#  make
#  ./ascent_render_example
#
###############################################################################

cmake_minimum_required(VERSION 3.9)

project(using_with_cmake)

#
# Use CMake's find_package to import ascent's targets
#
# PATHS is just a hint if someone runs this example from the Ascent install
# tree without setting up an environment hint to find Ascent
find_package(Ascent REQUIRED
             NO_DEFAULT_PATH
             PATHS ${CMAKE_SOURCE_DIR}/../../../)

# create our example
add_executable(ascent_render_example ascent_render_example.cpp)

# link to ascent
target_link_libraries(ascent_render_example ascent::ascent)

# if cuda is in the mix:
# we need to make sure CUDA_RESOLVE_DEVICE_SYMBOLS is on for our target
# (it propgates some way magically in 3.14, but not in 3.21)
if(CMAKE_CUDA_COMPILER)
    set_property(TARGET ascent_render_example PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
