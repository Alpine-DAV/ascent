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

###############################################################################
#
# LULESH CMake Build for Ascent
#
###############################################################################

set(LULESH_SOURCES
    lulesh.cc
    lulesh-comm.cc
    lulesh-viz.cc
    lulesh-util.cc
    lulesh-init.cc)

configure_file(ascent_actions.json ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.json COPYONLY)
configure_file(ascent_options.json ${CMAKE_CURRENT_BINARY_DIR}/ascent_options.json COPYONLY)

set(lulesh_deps ascent)

if(OPENMP_FOUND)
   set(lulesh_openmp_flags "-DLULESH_USE_OPENMP")
   list(APPEND lulesh_deps openmp)
else()
   set(lulesh_openmp_flags "")
endif()

blt_add_executable(
    NAME        lulesh_ser
    SOURCES     ${LULESH_SOURCES}
    DEPENDS_ON  ${lulesh_deps}
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})


blt_add_target_compile_flags(TO lulesh_ser FLAGS "-DUSE_MPI=0 ${lulesh_openmp_flags}")

# install target for lulesh serial
install(TARGETS lulesh_ser
        EXPORT  ascent
        LIBRARY DESTINATION examples/ascent/proxies/lulesh
        ARCHIVE DESTINATION examples/ascent/proxies/lulesh
        RUNTIME DESTINATION examples/ascent/proxies/lulesh
)

install(FILES ASCENT_README.md
              ascent_actions.json
              ascent_options.json
        DESTINATION examples/ascent/proxies/lulesh)


if(MPI_FOUND)

    set(lulesh_par_deps ascent_mpi mpi)
    if(ENABLE_OPENMP)
           list(APPEND lulesh_par_deps openmp)
    endif()

    blt_add_executable(
        NAME        lulesh_par
        SOURCES     ${LULESH_SOURCES}
        DEPENDS_ON  ${lulesh_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO lulesh_par FLAGS "-DUSE_MPI=1 ${lulesh_openmp_flags}")


    # install target for lulesh mpi
    install(TARGETS lulesh_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/proxies/lulesh
            ARCHIVE DESTINATION examples/ascent/proxies/lulesh
            RUNTIME DESTINATION examples/ascent/proxies/lulesh
    )


endif()






