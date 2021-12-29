# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Example that shows how to use an installed instance of Ascent in Makefile
# based build system.
#
# To build:
#  env ASCENT_DIR={ascent install path} make
#  ./ascent_render_example
#
# From within an ascent install:
#  make 
#  mpiexec -n 2 ./ascent_mpi_render_example
#
# Which corresponds to:
#
#  make ASCENT_DIR=../../../ 
#  mpiexec -n 2 ./ascent_mpi_render_example
#
###############################################################################

ASCENT_DIR ?= ../../..

# See $(ASCENT_DIR)/share/ascent/ascent_config.mk for detailed linking info 
include $(ASCENT_DIR)/share/ascent/ascent_config.mk

# make sure to enable c++11 support (conduit's interface now requires it)
CXX_FLAGS = -std=c++11
INC_FLAGS = $(ASCENT_INCLUDE_FLAGS)
LNK_FLAGS = $(ASCENT_LINK_RPATH) $(ASCENT_MPI_LIB_FLAGS)

main:
	$(CXX) $(CXX_FLAGS) $(INC_FLAGS) ascent_mpi_render_example.cpp $(LNK_FLAGS) -o ascent_mpi_render_example


clean:
	rm -f ascent_mpi_render_example


