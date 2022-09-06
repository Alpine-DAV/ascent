###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


################################
# Standard CMake Options
################################


# Fail if someone tries to config an in-source build.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported. Please remove "
                       "CMakeCache.txt from the 'src' dir and configure an "
                       "out-of-source build in another directory.")
endif()

set(_default_build_type "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
     message(STATUS "Setting build type to '${d_default_build_type}' as none was specified.")
     set(CMAKE_BUILD_TYPE "${_default_build_type}" CACHE STRING "Choose the type of build." FORCE)
     # Set the possible values of build type for cmake-gui
     set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
                  STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# enable creation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# always use position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMake build tool name: ${CMAKE_BUILD_TOOL}")

macro(ENABLE_WARNINGS)
    # set the warning levels we want to abide by
    if("${CMAKE_BUILD_TOOL}" MATCHES "(msdev|devenv|nmake|MSBuild)")
        add_definitions(/W4)
    else()
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"   OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel" OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
            # use these flags for clang, gcc, icc, or icx
            add_definitions(-Wall -Wextra)
        endif()
    endif()
endmacro()


################################
# Shared vs Static Libs
################################
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries (BUILD_SHARED_LIBS == ON)")
else()
    message(STATUS "Building static libraries (BUILD_SHARED_LIBS == OFF)")
endif()

################################
# Coverage Flags
################################
if(ENABLE_COVERAGE)
    message(STATUS "Building using coverage flags (ENABLE_COVERAGE == ON)")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
else()
    message(STATUS "Building without coverage flags (ENABLE_COVERAGE == OFF)")
endif()

################################
# Tests Option.
# save option here to defend if
# a TPL flips it a part a
# cmake import
################################
if(ENABLE_TESTS)
    set(ASCENT_ENABLE_TESTS ON)
else()
    set(ASCENT_ENABLE_TESTS OFF)
endif()

################################
# Win32 Output Dir Settings
################################
# On windows we place all of the libs and execs in one dir.
# dlls need to be located next to the execs since there is no
# rpath equiv on windows. I tried some gymnastics to extract
# and append the output dir of each dependent lib to the PATH for
# each of our tests and bins, but that was futile.
################################
if(WIN32)
    set(EXECUTABLE_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
    set(ARCHIVE_OUTPUT_PATH     ${CMAKE_BINARY_DIR}/bin)
    set(LIBRARY_OUTPUT_PATH     ${CMAKE_BINARY_DIR}/bin)
endif()

################################
# Standard CTest Options
################################
if(ENABLE_TESTS)
    set(MEMORYCHECK_SUPPRESSIONS_FILE "${CMAKE_SOURCE_DIR}/cmake/valgrind.supp" CACHE PATH "")
    include(CTest)
endif()

##############################################################################
# Try to extract the current git sha and other info
#
# This solution is derived from:
#  http://stackoverflow.com/a/21028226/203071
#
# This does not have full dependency tracking - it wont auto update when the
# git HEAD changes or when a branch is checked out, unless a change causes
# cmake to reconfigure.
#
# However, this limited approach will still be useful in many cases, 
# including building and for installing  conduit as a tpl
#
##############################################################################
find_package(Git)
if(GIT_FOUND)
    message(STATUS "git executable: ${GIT_EXECUTABLE}")
    # try to get sha1
    execute_process(COMMAND
        "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE ASCENT_GIT_SHA1
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if("${ASCENT_GIT_SHA1}" STREQUAL "")
       set(ASCENT_GIT_SHA1 "unknown")
    endif()
    message(STATUS "git SHA1: " ${ASCENT_GIT_SHA1})

    execute_process(COMMAND
        "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=5 --dirty
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE ASCENT_GIT_SHA1_ABBREV
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if("${ASCENT_GIT_SHA1_ABBREV}" STREQUAL "")
       set(ASCENT_GIT_SHA1_ABBREV "unknown")
    endif()
    message(STATUS "git SHA1-abbrev: " ${ASCENT_GIT_SHA1_ABBREV})

    # try to get tag
    execute_process(COMMAND
            "${GIT_EXECUTABLE}" describe --exact-match --tags
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE ASCENT_GIT_TAG
            ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if("${ASCENT_GIT_TAG}" STREQUAL "")
       set(ASCENT_GIT_TAG "unknown")
    endif()
    message(STATUS "git tag: " ${ASCENT_GIT_TAG})
  
endif()

###############################################################################
# This macro converts a cmake path to a platform specific string literal
# usable in C++. (For example, on windows C:/Path will be come C:\\Path)
###############################################################################

macro(convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro()

###############################################
# Protect ourselves from vtkm warning with cuda
###############################################
if(CUDA_FOUND)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
         #nvcc 9 introduced specific controls to disable the stack size warning
         #otherwise we let the warning occur. We have to set this in CMAKE_CUDA_FLAGS
         #as it is passed to the device link step, unlike compile_options
         set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink=--suppress-stack-size-warning")
    endif()

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

    if(ENABLE_CUDA_DEBUG_CPU_ONLY)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DDEBUG_CPU_ONLY")
    endif()

endif()
