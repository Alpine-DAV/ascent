#!/bin/bash
###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
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

