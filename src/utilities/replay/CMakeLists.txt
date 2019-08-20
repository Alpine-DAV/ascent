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
# Replay CMake Build for Ascent
#
###############################################################################

set(REPLAY_SOURCES
    replay.cpp)

set(replay_deps ascent)

if(OPENMP_FOUND)
   list(APPEND deps openmp)
endif()

if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        replay_ser
        SOURCES     ${REPLAY_SOURCES}
        DEPENDS_ON  ${replay_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    
    # install target for replay serial
    install(TARGETS replay_ser
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/replay
            ARCHIVE DESTINATION utilities/ascent/replay
            RUNTIME DESTINATION utilities/ascent/replay
    )
endif()

if(MPI_FOUND)

    set(replay_mpi_deps ascent_mpi mpi)
    if(ENABLE_OPENMP)
           list(APPEND replay_mpi_deps openmp)
    endif()

    blt_add_executable(
        NAME        replay_mpi
        SOURCES     ${REPLAY_SOURCES}
        DEPENDS_ON  ${replay_mpi_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    blt_add_target_compile_flags(TO replay_mpi FLAGS "-DREPLAY_MPI=1")

    # install target for replay mpi
    install(TARGETS replay_mpi
            EXPORT  ascent
            LIBRARY DESTINATION utilities/ascent/replay
            ARCHIVE DESTINATION utilities/ascent/replay
            RUNTIME DESTINATION utilities/ascent/replay
    )


endif()






