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
# Kripke CMake Build for Ascent
#
###############################################################################


set(KRIPKE_SOURCES
    Kripke/Directions.cpp  
    Kripke/Grid.cpp  
    Kripke/Kernel.cpp  
    Kripke/Layout.cpp
    Kripke/Subdomain.cpp  
    Kripke/Sweep_Solver.cpp
    Kripke/ParallelComm.cpp  
    Kripke/Timing.cpp  
    
    Kripke/Kernel/Kernel_3d_GDZ.cpp
    Kripke/Kernel/Kernel_3d_DGZ.cpp
    Kripke/Kernel/Kernel_3d_ZDG.cpp
    Kripke/Kernel/Kernel_3d_DZG.cpp
    Kripke/Kernel/Kernel_3d_ZGD.cpp
    Kripke/Kernel/Kernel_3d_GZD.cpp
  
    Kripke/ParallelComm/BlockJacobiComm.cpp
    Kripke/ParallelComm/SweepComm.cpp 
    tools/kripke.cpp
    tools/testKernels.cpp
)

# kripke reqs mpi
if(MPI_FOUND)
    # copy over example command line
    configure_file(run_kripke_simple_example.sh 
                   ${CMAKE_CURRENT_BINARY_DIR}/run_kripke_simple_example.sh
                   COPYONLY)
    configure_file(ascent_actions.json ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.json COPYONLY)
    configure_file(ascent_options.json ${CMAKE_CURRENT_BINARY_DIR}/ascent_options.json COPYONLY)

    if(ENABLE_OPENMP)
       set(kripke_openmp_flags "-DKRIPKE_USE_OPENMP")
    else()
       set(kripke_openmp_flags "")
    endif()

    include_directories(.)
    include_directories("tools")


    set(kripke_par_deps ascent_mpi mpi)
    if(ENABLE_OPENMP)
           list(APPEND kripke_par_deps openmp)
    endif()
    
    blt_add_executable(
        NAME        kripke_par
        SOURCES     ${KRIPKE_SOURCES}
        DEPENDS_ON  ${kripke_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO kripke_par FLAGS "-D PARALLEL ${kripke_openmp_flags}")

    # install target for kripke mpi
    install(TARGETS kripke_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/proxies/kripke
            ARCHIVE DESTINATION examples/proxies/kripke
            RUNTIME DESTINATION examples/proxies/kripke
    )

    install(FILES ASCENT_README.md 
                  ascent_actions.json
                  ascent_options.json
            DESTINATION examples/proxies/kripke)

endif()






