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


###############################################################################
#
# file: src/alpine/flow/CMakeLists.txt
#
###############################################################################

set(alpine_flow_sources
    # alpine flow interface
    alpine_flow_data.cpp
    alpine_flow_registry.cpp
    alpine_flow_filter.cpp
    alpine_flow_filters.cpp
    alpine_flow_graph.cpp
    alpine_flow_workspace.cpp
    filters/alpine_flow_registry_filters.cpp)
    
set(alpine_flow_headers
    alpine_flow_data.hpp
    alpine_flow_registry.hpp
    alpine_flow_filter.hpp
    alpine_flow_filters.hpp
    alpine_flow_graph.hpp
    alpine_flow_workspace.hpp
    filters/alpine_flow_registry_filters.hpp
    alpine_flow.hpp)

set(alpine_flow_thirdparty_libs 
    conduit)

##########################################
# Build a serial version of flow
##########################################

if(BUILD_SHARED_LIBS)
    message(STATUS "Building alpine_flow as a shared library")
     add_library(alpine_flow SHARED 
                 ${alpine_flow_sources} 
                 ${alpine_flow_headers})
else()
    message(STATUS "Building alpine_flow as a static library")
    add_library(alpine_flow STATIC 
                ${alpine_flow_sources} 
                ${alpine_flow_headers})
endif()


target_link_libraries(alpine_flow ${alpine_flow_thirdparty_libs})

# install target for alpine flow lib
install(TARGETS alpine_flow
        EXPORT  alpine
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION lib
)

install(FILES ${alpine_flow_headers} DESTINATION include/alpine)


