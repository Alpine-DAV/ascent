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

# Specify the sources of the pure python and compiled portions of our module.
SET(flow_py_python_sources  py_src/__init__.py py_src/helpers.py)
#SET(ascent_py_headers        flow_python.hpp)
SET(flow_py_cpp_sources     flow_python.cpp)


# Setup our module
PYTHON_ADD_HYBRID_MODULE(flow_python
                         python-modules
                         flow
                         setup.py
                         "${flow_py_python_sources}"
                         #${flow_py_headers}
                         ${flow_py_cpp_sources})

# link with the proper libs (beyond python)
target_link_libraries(flow_python ascent_flow)


# TODO: ADD SUPPORT FOR SUPPORT PYTHON_MODULE_INSTALL_PREFIX
# install the capi header so other python modules can use it
# support alt install dir for python module via PYTHON_MODULE_INSTALL_PREFIX
# if(PYTHON_MODULE_INSTALL_PREFIX)
#     install(FILES ${flow_py_headers} DESTINATION ${PYTHON_MODULE_INSTALL_PREFIX}/flow/)
# else()
#     install(FILES ${flow_py_headers} DESTINATION python-modules/flow/)
# endif()




