###############################################################################
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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

###############################################################################
#
# Example that shows how to use an installed instance of Ascent in Makefile
# based build system.
#
# To build:
#  env ASCENT_DIR={ascent install path} \
#      CONDUIT_DIR={conduit install path} \
#      HDF5_DIR={hdf5 install path} \
#      ZLIB_DIR={zlib install path} \
#      VTKH_DIR={vtkh install path} \
#      VTKM_DIR={vtkm install path} \
#      VTKM_VERSION={vtkm version (ex: 1.3)} \
#   make
#  ./ascent_render_example
# 
#
###############################################################################


#################
# Include Flags
#################
ASCENT_INCLUDE_FLAGS = -I$(CONDUIT_DIR)/include/conduit \
                       -I$(ASCENT_DIR)/include/ascent

#################
# Linking Flags
#################

##########
# Conduit
##########
# If using Conduit build without MPI
CONDUIT_LIB_FLAGS = -L $(CONDUIT_DIR)/lib \
                    -lconduit_blueprint \
                    -lconduit_relay \
                    -lconduit

# If using Conduit build with MPI
CONDUIT_MPI_LIB_FLAGS = -L $(CONDUIT_DIR)/lib \
                        -lconduit_relay_mpi_io \
                        -lconduit_relay_mpi \
                        -lconduit_blueprint \
                        -lconduit_relay \
                        -lconduit 

# zlib
ZLIB_LIB_FLAGS = -L $(ZLIB_DIR)/lib -lz

# hdf5
HDF5_LIB_FLAGS = -L $(HDF5_DIR)/lib -lhdf5

##########
# VTKH
##########
# If using VTKH without mpi
VTKH_LIB_FLAGS=-L $(VTKH_DIR)/lib \
                -lvtkh_rendering \
                -lvtkh_filters \
                -lvtkh_core \
                -lvtkh_lodepng \
                -lvtkh_utils

# If using VTKH with mpi
VTKH_MPI_LIB_FLAGS=-L $(VTKH_DIR)/lib \
                   -lvtkh_rendering_mpi \
                   -lvtkh_filters_mpi \
                   -lvtkh_core_mpi \
                   -lvtkh_lodepng \
                   -lvtkh_utils_mpi

#vtkm
VTKM_LIB_FLAGS = -L $(VTKM_DIR)/lib \
                 -lvtkm_cont-$(VTKM_VERSION) \
                 -lvtkm_rendering-$(VTKM_VERSION)

##########
# Ascent
##########
# if using Ascent without mpi
ASCENT_LIB_FLAGS = -L $(ASCENT_DIR)/lib \
                   -lascent \
                   -lflow \
                   -lrover \
                   $(VTKH_LIB_FLAGS) \
                   $(VTKM_LIB_FLAGS) \
                   $(CONDUIT_LIB_FLAGS) \
                   $(HDF5_LIB_FLAGS) \
                   $(ZLIB_LIB_FLAGS)

# if using Ascent with mpi
ASCENT_MPI_LIB_FLAGS = -L $(ASCENT_DIR)/lib \
                       -lascent_mpi \
                       -lflow \
                       -lrover_mpi \
                       $(VTKH_MPI_LIB_FLAGS) \
                       $(VTKM_LIB_FLAGS) \
                       $(CONDUIT_MPI_LIB_FLAGS) \
                       $(HDF5_LIB_FLAGS) \
                       $(ZLIB_LIB_FLAGS)


main:
	$(CXX) $(ASCENT_INCLUDE_FLAGS) ascent_render_example.cpp $(ASCENT_LIB_FLAGS) -o ascent_render_example

# TODO: Add mpi render example



