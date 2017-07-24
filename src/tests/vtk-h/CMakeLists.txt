###############################################################################
# Copyright (c) 2015-2017, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-716457
# 
# All rights reserved.
# 
# This file is part of Alpine. 
# 
# For details, see: http://software.llnl.gov/alpine/.
# 
# Please also read alpine/LICENSE
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
# Unit Tests
################################

################################
# Core VTK-h Unit Tests
################################
set(BASIC_TESTS t_vtk-h_smoke
                t_vtk-h_dataset
                t_vtk-h_clip
                t_vtk-h_marching_cubes
                t_vtk-h_raytracer
                t_vtk-h_volume_renderer
                t_vtk-h_wavelet_compressor
                )


set(MPI_TESTS t_vtk-h_smoke_par
              t_vtk-h_dataset_par
              t_vtk-h_marching_cubes_par
              t_vtk-h_raytracer_par
              t_vtk-h_volume_renderer_par
              )


################################
# Add main tests
################################
message(STATUS "Adding vtk-h lib unit tests")
foreach(TEST ${BASIC_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON vtkh)
endforeach()


################################
# Add optional tests
################################

if(MPI_FOUND AND ENABLE_MPI)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 2 procs
        add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2 DEPENDS_ON vtkh_par) 
        blt_add_target_compile_flags(TO ${TEST} FLAGS "-D PARALLEL")
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()



