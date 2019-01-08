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
# Noise CMake Build for Ascent
#
###############################################################################

set(noise_sources
    noise.cpp
    open_simplex_noise.c)

set(noise_header
    open_simplex_noise.h)

set(noise_deps ascent)
if(OPENMP_FOUND)
   set(noise_openmp_flags "-DNOISE_USE_OPENMP")
   list(APPEND noise_deps openmp)
else()
   set(noise_openmp_flags "")
endif()

blt_add_executable(
    NAME        noise_ser
    SOURCES     ${noise_sources}
    DEPENDS_ON  ${noise_deps}
    OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR})

blt_add_target_compile_flags(TO noise_ser FLAGS "${noise_openmp_flags}")

# install target for noise example
install(TARGETS noise_ser
        EXPORT  ascent
        LIBRARY DESTINATION examples/ascent/synthetic/noise
        ARCHIVE DESTINATION examples/ascent/synthetic/noise
        RUNTIME DESTINATION examples/ascent/synthetic/noise
)

install(FILES ASCENT_README.md DESTINATION examples/ascent/synthetic/noise)

if(MPI_FOUND)

    set(noise_par_deps ascent_mpi mpi)
    if(OPENMP_FOUND)
           list(APPEND noise_par_deps openmp)
    endif()

    blt_add_executable(
        NAME        noise_par
        SOURCES     ${noise_sources}
        DEPENDS_ON  ${noise_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO noise_par FLAGS "-DPARALLEL  ${noise_openmp_flags}")

    # install target for mpi noise example
    install(TARGETS noise_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/synthetic/noise
            ARCHIVE DESTINATION examples/ascent/synthetic/noise
            RUNTIME DESTINATION examples/ascent/synthetic/noise
    )


endif()






