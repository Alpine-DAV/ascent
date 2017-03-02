#!/bin/bash
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


#------------------------------------------------------------------------------
#clean up existing build + install directories
#------------------------------------------------------------------------------
rm -rf build-debug install-debug

#------------------------------------------------------------------------------
# create a new build directory
#------------------------------------------------------------------------------
mkdir build-debug
mkdir install-debug
cd build-debug

#------------------------------------------------------------------------------
# setup desired basic cmake options
#------------------------------------------------------------------------------
export CMAKE_OPTS=" -DCMAKE_BUILD_TYPE=Debug"
export CMAKE_OPTS="$CMAKE_OPTS -DCMAKE_INSTALL_PREFIX=../install-debug"

#------------------------------------------------------------------------------
# Check if a host config was direclty passed
#------------------------------------------------------------------------------
if [ $# -ge 1 ]; then
    if [ "${1: -6}" == ".cmake" ]; then
        export HOST_CONFIG=../$1
        echo "Looking for host-config file: $HOST_CONFIG"
        if [[ -e  "$HOST_CONFIG" ]]; then
            echo "FOUND: $HOST_CONFIG"
            export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
        fi
    fi
fi

if [[ ! -e  "$HOST_CONFIG" ]]; then
#------------------------------------------------------------------------------
# if no host config was passed, try include an initial cmake settings file 
# if appropriate
#------------------------------------------------------------------------------
    # first look for a specific config for this machine
    export HOSTNAME=`hostname`
    export HOST_CONFIG=`ls ../host-configs/$HOSTNAME*.cmake`
    echo "Looking for host-config file: $HOST_CONFIG"
    if [[ -e  "$HOST_CONFIG" ]]; then
        echo "FOUND: $HOST_CONFIG"
        export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
    # then check for a sys-type based config
    elif [[ "$SYS_TYPE" != "" ]]; then
        export HOST_CONFIG=../host-configs/$SYS_TYPE.cmake
        echo "Looking for SYS_TYPE based host-config file: $HOST_CONFIG"
        if [[ -e  "$HOST_CONFIG" ]]; then
            echo "FOUND: $HOST_CONFIG"
            export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
        fi
    else 
        # fallback to simple a uname based config (Linux / Darwin / etc)
        export HOST_CONFIG=../host-configs/`uname`.cmake
        echo "Looking for uname based host-config file: $HOST_CONFIG"
        if [[ -e  "$HOST_CONFIG" ]]; then
            echo "FOUND: $HOST_CONFIG"
            export CMAKE_OPTS="$CMAKE_OPTS -C $HOST_CONFIG"
        fi
    fi
fi

#------------------------------------------------------------------------------
# run cmake to configure
#------------------------------------------------------------------------------
echo "cmake $CMAKE_OPTS ../src"
cmake  $CMAKE_OPTS \
       ../src 
# return to the starting dir
cd ../

#------------------------------------------------------------------------------
# add extended builds when a non host config argument is passed
#------------------------------------------------------------------------------
if [ $# -ge 1 ]; then
    if [ "${1: -6}" == ".cmake" ]; then
        # skip this case
        echo ""
    else
        # also create an xcode build for debugging on osx
        if [ "$TERM_PROGRAM" = "Apple_Terminal" ]; then
            rm -rf build-debug-xcode
            mkdir build-debug-xcode
            cd build-debug-xcode
            cmake -G Xcode $CMAKE_OPTS ../src 
            cd ../
        fi
    fi
fi

