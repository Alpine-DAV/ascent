# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

# Find the interpreter first
if(PYTHON_DIR AND NOT PYTHON_EXECUTABLE)
    if(UNIX)
        # look for python 3 first
        set(PYTHON_EXECUTABLE ${PYTHON_DIR}/bin/python3)
        # if this doesn't exist, look for python
        if(NOT EXISTS "${PYTHON_EXECUTABLE}")
            set(PYTHON_EXECUTABLE ${PYTHON_DIR}/bin/python)
        endif()
    elseif(WIN32)
        set(PYTHON_EXECUTABLE ${PYTHON_DIR}/python.exe)
    endif()
endif()

find_package(PythonInterp REQUIRED)
if(PYTHONINTERP_FOUND)
        
        MESSAGE(STATUS "PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE}")

        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                        "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('VERSION'))"
                        OUTPUT_VARIABLE PYTHON_CONFIG_VERSION
                        ERROR_VARIABLE  ERROR_FINDING_PYTHON_VERSION)
        MESSAGE(STATUS "PYTHON_CONFIG_VERSION ${PYTHON_CONFIG_VERSION}")

        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_python_inc;sys.stdout.write(get_python_inc())"
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                        ERROR_VARIABLE ERROR_FINDING_INCLUDES)
        MESSAGE(STATUS "PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}")
        
        if(NOT EXISTS ${PYTHON_INCLUDE_DIR})
            MESSAGE(FATAL_ERROR "Reported PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR} does not exist!")
        endif()

        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_python_lib;sys.stdout.write(get_python_lib())"
                        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
                        ERROR_VARIABLE ERROR_FINDING_SITE_PACKAGES_DIR)
        MESSAGE(STATUS "PYTHON_SITE_PACKAGES_DIR ${PYTHON_SITE_PACKAGES_DIR}")

        if(NOT EXISTS ${PYTHON_SITE_PACKAGES_DIR})
            MESSAGE(FATAL_ERROR "Reported PYTHON_SITE_PACKAGES_DIR ${PYTHON_SITE_PACKAGES_DIR} does not exist!")
        endif()

        
        # check if we need "-undefined dynamic_lookup" by inspecting LDSHARED flags
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
                                "import sys;import sysconfig;sys.stdout.write(sysconfig.get_config_var('LDSHARED'))"
                        OUTPUT_VARIABLE PYTHON_LDSHARED_FLAGS
                        ERROR_VARIABLE ERROR_FINDING_PYTHON_LDSHARED_FLAGS)

        MESSAGE(STATUS "PYTHON_LDSHARED_FLAGS ${PYTHON_LDSHARED_FLAGS}")

        if(PYTHON_LDSHARED_FLAGS MATCHES "-undefined dynamic_lookup")
             MESSAGE(STATUS "PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG is ON")
            set(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG ON)
        else()
             MESSAGE(STATUS "PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG is OFF")
            set(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG OFF)
        endif()

        # our goal is to find the specific python lib, based on info
        # we extract from distutils.sysconfig from the python executable
        #
        # check for python libs differs for windows python installs
        if(NOT WIN32)
            # we may build a shared python module against a static python
            # check for both shared and static libs cases

            # combos to try:
            # shared:
            #  LIBDIR + LDLIBRARY
            #  LIBPL + LDLIBRARY
            # static:
            #  LIBDIR + LIBRARY
            #  LIBPL + LIBRARY

            execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                    "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LIBDIR'))"
                            OUTPUT_VARIABLE PYTHON_CONFIG_LIBDIR
                            ERROR_VARIABLE  ERROR_FINDING_PYTHON_LIBDIR)

            execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                    "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LIBPL'))"
                            OUTPUT_VARIABLE PYTHON_CONFIG_LIBPL
                            ERROR_VARIABLE  ERROR_FINDING_PYTHON_LIBPL)

            execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                    "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LDLIBRARY'))"
                            OUTPUT_VARIABLE PYTHON_CONFIG_LDLIBRARY
                            ERROR_VARIABLE  ERROR_FINDING_PYTHON_LDLIBRARY)

            execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                    "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LIBRARY'))"
                            OUTPUT_VARIABLE PYTHON_CONFIG_LIBRARY
                            ERROR_VARIABLE  ERROR_FINDING_PYTHON_LIBRARY)

            message(STATUS "PYTHON_CONFIG_LIBDIR:     ${PYTHON_CONFIG_LIBDIR}")
            message(STATUS "PYTHON_CONFIG_LIBPL:      ${PYTHON_CONFIG_LIBPL}")
            message(STATUS "PYTHON_CONFIG_LDLIBRARY:  ${PYTHON_CONFIG_LDLIBRARY}")
            message(STATUS "PYTHON_CONFIG_LIBRARY:    ${PYTHON_CONFIG_LIBRARY}")

            set(PYTHON_LIBRARY "")
            # look for shared libs first
            # shared libdir + ldlibrary
            if(NOT EXISTS ${PYTHON_LIBRARY})
                if(IS_DIRECTORY ${PYTHON_CONFIG_LIBDIR})
                    set(_PYTHON_LIBRARY_TEST  "${PYTHON_CONFIG_LIBDIR}/${PYTHON_CONFIG_LDLIBRARY}")
                    message(STATUS "Checking for python library at: ${_PYTHON_LIBRARY_TEST}")
                    if(EXISTS ${_PYTHON_LIBRARY_TEST})
                        set(PYTHON_LIBRARY ${_PYTHON_LIBRARY_TEST})
                    endif()
                endif()
            endif()

            # shared libpl + ldlibrary
            if(NOT EXISTS ${PYTHON_LIBRARY})
                if(IS_DIRECTORY ${PYTHON_CONFIG_LIBPL})
                    set(_PYTHON_LIBRARY_TEST  "${PYTHON_CONFIG_LIBPL}/${PYTHON_CONFIG_LDLIBRARY}")
                    message(STATUS "Checking for python library at: ${_PYTHON_LIBRARY_TEST}")
                    if(EXISTS ${_PYTHON_LIBRARY_TEST})
                        set(PYTHON_LIBRARY ${_PYTHON_LIBRARY_TEST})
                    endif()
                endif()
            endif()

            # static: libdir + library
            if(NOT EXISTS ${PYTHON_LIBRARY})
                if(IS_DIRECTORY ${PYTHON_CONFIG_LIBDIR})
                    set(_PYTHON_LIBRARY_TEST  "${PYTHON_CONFIG_LIBDIR}/${PYTHON_CONFIG_LIBRARY}")
                    message(STATUS "Checking for python library at: ${_PYTHON_LIBRARY_TEST}")
                    if(EXISTS ${_PYTHON_LIBRARY_TEST})
                        set(PYTHON_LIBRARY ${_PYTHON_LIBRARY_TEST})
                    endif()
                endif()
            endif()

            # static: libpl + library
            if(NOT EXISTS ${PYTHON_LIBRARY})
                if(IS_DIRECTORY ${PYTHON_CONFIG_LIBPL})
                    set(_PYTHON_LIBRARY_TEST  "${PYTHON_CONFIG_LIBPL}/${PYTHON_CONFIG_LIBRARY}")
                    message(STATUS "Checking for python library at: ${_PYTHON_LIBRARY_TEST}")
                    if(EXISTS ${_PYTHON_LIBRARY_TEST})
                        set(PYTHON_LIBRARY ${_PYTHON_LIBRARY_TEST})
                    endif()
                endif()
            endif()
        else() # windows 
            get_filename_component(PYTHON_ROOT_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
            # Note: this assumes that two versions of python are not installed in the same dest dir
            set(_PYTHON_LIBRARY_TEST  "${PYTHON_ROOT_DIR}/libs/python${PYTHON_CONFIG_VERSION}.lib")
            message(STATUS "Checking for python library at: ${_PYTHON_LIBRARY_TEST}")
            if(EXISTS ${_PYTHON_LIBRARY_TEST})
                set(PYTHON_LIBRARY ${_PYTHON_LIBRARY_TEST})
            endif()
        endif()

        if(NOT EXISTS ${PYTHON_LIBRARY})
            MESSAGE(FATAL_ERROR "Failed to find main library using PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
        endif()

        MESSAGE(STATUS "{PythonLibs from PythonInterp} using: PYTHON_LIBRARY=${PYTHON_LIBRARY}")
        find_package(PythonLibs)

        if(NOT PYTHONLIBS_FOUND)
            MESSAGE(FATAL_ERROR "Failed to find Python Libraries using PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
        endif()
        
endif()


find_package_handle_standard_args(Python  DEFAULT_MSG
                                  PYTHON_LIBRARY PYTHON_INCLUDE_DIR)



##############################################################################
# Macro to use a pure python distutils setup script
##############################################################################
FUNCTION(PYTHON_ADD_DISTUTILS_SETUP)
    set(singleValuedArgs NAME DEST_DIR PY_MODULE_DIR PY_SETUP_FILE FOLDER)
    set(multiValuedArgs  PY_SOURCES)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
            "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    # check req'd args
    if(NOT DEFINED args_NAME)
       message(FATAL_ERROR
               "PYTHON_ADD_HYBRID_MODULE: Missing required argument NAME")
    endif()

    if(NOT DEFINED args_DEST_DIR)
       message(FATAL_ERROR
               "PYTHON_ADD_HYBRID_MODULE: Missing required argument DEST_DIR")
    endif()

    if(NOT DEFINED args_PY_MODULE_DIR)
       message(FATAL_ERROR
       "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_MODULE_DIR")
    endif()

    if(NOT DEFINED args_PY_SETUP_FILE)
       message(FATAL_ERROR
       "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_SETUP_FILE")
    endif()

    if(NOT DEFINED args_PY_SOURCES)
       message(FATAL_ERROR
       "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_SOURCES")
    endif()

    MESSAGE(STATUS "Configuring python distutils setup: ${args_NAME}")

    # dest for build dir
    set(abs_dest_path ${CMAKE_BINARY_DIR}/${args_DEST_DIR})
    if(WIN32)
        # on windows, distutils seems to need standard "\" style paths
        string(REGEX REPLACE "/" "\\\\" abs_dest_path  ${abs_dest_path})
    endif()

    add_custom_command(OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}_build
            COMMAND ${PYTHON_EXECUTABLE} ${args_PY_SETUP_FILE} -v
            build
            --build-base=${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}_build
            install
            --install-purelib="${abs_dest_path}"
            DEPENDS  ${args_PY_SETUP_FILE} ${args_PY_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_custom_target(${args_NAME} ALL DEPENDS
                      ${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}_build)

    # also use distutils for the install ...
    # if PYTHON_MODULE_INSTALL_PREFIX is set, install there
    if(PYTHON_MODULE_INSTALL_PREFIX)
        set(py_mod_inst_prefix ${PYTHON_MODULE_INSTALL_PREFIX})
        # make sure windows style paths don't ruin our day (or night)
        if(WIN32)
            string(REGEX REPLACE "/" "\\\\" py_mod_inst_prefix  ${PYTHON_MODULE_INSTALL_PREFIX})
        endif()
        INSTALL(CODE
            "
            EXECUTE_PROCESS(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${PYTHON_EXECUTABLE} ${args_PY_SETUP_FILE} -v
                    build   --build-base=${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}_build_install
                    install --install-purelib=${py_mod_inst_prefix}
                OUTPUT_VARIABLE PY_DIST_UTILS_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_DIST_UTILS_INSTALL_OUT}\")
            ")
    else()
        # else install to the dest dir under CMAKE_INSTALL_PREFIX
        INSTALL(CODE
            "
            EXECUTE_PROCESS(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${PYTHON_EXECUTABLE} ${args_PY_SETUP_FILE} -v
                    build   --build-base=${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}_build_install
                    install --install-purelib=\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${args_DEST_DIR}
                OUTPUT_VARIABLE PY_DIST_UTILS_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_DIST_UTILS_INSTALL_OUT}\")
            ")
    endif()

    # set folder if passed
    if(DEFINED args_FOLDER)
        blt_set_target_folder(TARGET ${args_NAME} FOLDER ${args_FOLDER})
    endif()

ENDFUNCTION(PYTHON_ADD_DISTUTILS_SETUP)

##############################################################################
# Macro to create a compiled python module 
##############################################################################
#
# we use this instead of the std ADD_PYTHON_MODULE cmake command 
# to setup proper install targets.
#
##############################################################################
FUNCTION(PYTHON_ADD_COMPILED_MODULE)
    set(singleValuedArgs NAME DEST_DIR PY_MODULE_DIR FOLDER)
    set(multiValuedArgs  SOURCES)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
            "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    # check req'd args
    if(NOT DEFINED args_NAME)
       message(FATAL_ERROR
               "PYTHON_ADD_COMPILED_MODULE: Missing required argument NAME")
    endif()

    if(NOT DEFINED args_DEST_DIR)
       message(FATAL_ERROR
               "PYTHON_ADD_COMPILED_MODULE: Missing required argument DEST_DIR")
    endif()

    if(NOT DEFINED args_PY_MODULE_DIR)
       message(FATAL_ERROR
       "PYTHON_ADD_COMPILED_MODULE: Missing required argument PY_MODULE_DIR")
    endif()

    if(NOT DEFINED args_SOURCES)
       message(FATAL_ERROR
               "PYTHON_ADD_COMPILED_MODULE: Missing required argument SOURCES")
    endif()

    MESSAGE(STATUS "Configuring python module: ${args_NAME}")
    PYTHON_ADD_MODULE(${args_NAME} ${args_SOURCES})

    set_target_properties(${args_NAME} PROPERTIES
                                       LIBRARY_OUTPUT_DIRECTORY
                                       ${CMAKE_BINARY_DIR}/${args_DEST_DIR}/${args_PY_MODULE_DIR})

    # set folder if passed
    if(DEFINED args_FOLDER)
        blt_set_target_folder(TARGET ${args_NAME} FOLDER ${args_FOLDER})
    endif()

    foreach(CFG_TYPE ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CFG_TYPE} CFG_TYPE)
        set_target_properties(${args_NAME} PROPERTIES
                                           LIBRARY_OUTPUT_DIRECTORY_${CFG_TYPE}
                                           ${CMAKE_BINARY_DIR}/${args_DEST_DIR}/${args_PY_MODULE_DIR})
    endforeach()

    MESSAGE(STATUS "${args_NAME} build location: ${CMAKE_BINARY_DIR}/${args_DEST_DIR}/${args_PY_MODULE_DIR}")

    # macOS and linux
    # defer linking with python, let the final python interpreter
    # provide the proper symbols

    # on osx we need to use the following flag to 
    # avoid undefined linking errors
    if(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG)
        set_target_properties(${args_NAME} PROPERTIES
                              LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    
    # win32, link to python
    if(WIN32)
        target_link_libraries(${args_NAME} ${PYTHON_LIBRARIES})
    endif()

    # support installing the python module components to an
    # an alternate dir, set via PYTHON_MODULE_INSTALL_PREFIX 
    set(py_install_dir ${args_DEST_DIR})
    if(PYTHON_MODULE_INSTALL_PREFIX)
        set(py_install_dir ${PYTHON_MODULE_INSTALL_PREFIX})
    endif()

    install(TARGETS ${args_NAME}
            EXPORT  conduit
            LIBRARY DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
            ARCHIVE DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
            RUNTIME DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
    )

ENDFUNCTION(PYTHON_ADD_COMPILED_MODULE)

##############################################################################
# Macro to create a compiled distutils and compiled python module
##############################################################################
FUNCTION(PYTHON_ADD_HYBRID_MODULE)
    set(singleValuedArgs NAME DEST_DIR PY_MODULE_DIR PY_SETUP_FILE FOLDER)
    set(multiValuedArgs  PY_SOURCES SOURCES)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
            "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

     # check req'd args
    if(NOT DEFINED args_NAME)
        message(FATAL_ERROR
                "PYTHON_ADD_HYBRID_MODULE: Missing required argument NAME")
    endif()

    if(NOT DEFINED args_DEST_DIR)
        message(FATAL_ERROR
                "PYTHON_ADD_HYBRID_MODULE: Missing required argument DEST_DIR")
    endif()

    if(NOT DEFINED args_PY_MODULE_DIR)
        message(FATAL_ERROR
        "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_MODULE_DIR")
    endif()

    if(NOT DEFINED args_PY_SETUP_FILE)
        message(FATAL_ERROR
        "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_SETUP_FILE")
    endif()

    if(NOT DEFINED args_PY_SOURCES)
        message(FATAL_ERROR
        "PYTHON_ADD_HYBRID_MODULE: Missing required argument PY_SOURCES")
    endif()

    if(NOT DEFINED args_SOURCES)
        message(FATAL_ERROR
                "PYTHON_ADD_HYBRID_MODULE: Missing required argument SOURCES")
    endif()

    MESSAGE(STATUS "Configuring hybrid python module: ${args_NAME}")

    PYTHON_ADD_DISTUTILS_SETUP(NAME          "${args_NAME}_py_setup"
                               DEST_DIR      ${args_DEST_DIR}
                               PY_MODULE_DIR ${args_PY_MODULE_DIR}
                               PY_SETUP_FILE ${args_PY_SETUP_FILE}
                               PY_SOURCES    ${args_PY_SOURCES}
                               FOLDER        ${args_FOLDER})

    PYTHON_ADD_COMPILED_MODULE(NAME          ${args_NAME}
                               DEST_DIR      ${args_DEST_DIR}
                               PY_MODULE_DIR ${args_PY_MODULE_DIR}
                               SOURCES       ${args_SOURCES}
                               FOLDER        ${args_FOLDER})

ENDFUNCTION(PYTHON_ADD_HYBRID_MODULE)




#
# Also register python as a BLT dep,to support the case were we link python,
# as opposed to creating python modules via the above macros.
#

blt_register_library(NAME python
                     INCLUDES ${PYTHON_INCLUDE_DIR}
                     LIBRARIES ${PYTHON_LIBRARY} )


