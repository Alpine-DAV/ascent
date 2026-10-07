# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.

#
# Downstream packages may configure using PYTHON_DIR or PYTHON_EXECUTABLE
#
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

# allow PYTHON_EXECUTABLE to init Python3_EXECUTABLE
if(PYTHON_EXECUTABLE AND NOT Python3_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

if(Python3_EXECUTABLE)
    message(STATUS "Python Executable: ${Python3_EXECUTABLE}")
endif()

find_package(Python3
             REQUIRED
             COMPONENTS Interpreter Development NumPy)

# normalize python found to all caps
if(Python3_FOUND)
    set(PYTHON_FOUND TRUE)
endif()

##############################################################################
# Manual, bare-minimum check of setuptools version
#
# Installing with `pip install --no-build-isolation` causes pip to use the
# setuptools already installed in this python rather than fetching the version
# specified in the pyproject.toml. Versions of setuptools older than 61.0.0
# do not understand pyproject.toml metadata, but instead of failing, those
# versions quietly build an empty package.
#
# As a result, 61.0.0 is the hard floor enforced explicitly here. The "requires"
# section of the pyproject.toml may pin a higher recommended version, but that
# is a separate concern from this check.
##############################################################################
set(SETUPTOOLS_MIN_VERSION 61.0.0)

execute_process(COMMAND ${Python3_EXECUTABLE} -c
                        "import setuptools; print(setuptools.__version__)"
                RESULT_VARIABLE setuptools_probe_result
                OUTPUT_VARIABLE setuptools_version
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT setuptools_probe_result EQUAL 0)
    message(FATAL_ERROR
            "Could not determine the setuptools version in ${Python3_EXECUTABLE} "
            "(it may be missing or broken). Install or repair setuptools there, "
            "or build with ENABLE_PYTHON=OFF.")
elseif(setuptools_version VERSION_LESS SETUPTOOLS_MIN_VERSION)
    message(FATAL_ERROR
            "Python modules require setuptools "
            "${SETUPTOOLS_MIN_VERSION} or newer, but "
            "${Python3_EXECUTABLE} provides ${setuptools_version}. Upgrade "
            "setuptools there, use a python that provides a newer one, or "
            "build with ENABLE_PYTHON=OFF.")
endif()

##############################################################################
# Macro to use a pure python pip setup script
##############################################################################
FUNCTION(PYTHON_ADD_PIP_SETUP)
    set(singleValuedArgs NAME DEST_DIR PY_MODULE_DIR PY_SETUP_FILE FOLDER)
    set(multiValuedArgs  PY_SOURCES)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
            "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    # check req'd args
    if(NOT DEFINED args_NAME)
       message(FATAL_ERROR
               "PYTHON_ADD_PIP_SETUP: Missing required argument NAME")
    endif()

    if(NOT DEFINED args_DEST_DIR)
       message(FATAL_ERROR
               "PYTHON_ADD_PIP_SETUP: Missing required argument DEST_DIR")
    endif()

    if(NOT DEFINED args_PY_MODULE_DIR)
       message(FATAL_ERROR
       "PYTHON_ADD_PIP_SETUP: Missing required argument PY_MODULE_DIR")
    endif()

    if(NOT DEFINED args_PY_SETUP_FILE)
       message(FATAL_ERROR
       "PYTHON_ADD_PIP_SETUP: Missing required argument PY_SETUP_FILE")
    endif()

    if(NOT DEFINED args_PY_SOURCES)
       message(FATAL_ERROR
       "PYTHON_ADD_PIP_SETUP: Missing required argument PY_SOURCES")
    endif()

    MESSAGE(STATUS "Configuring python pip setup: ${args_NAME}")

    # dest for build dir
    set(abs_dest_path ${CMAKE_BINARY_DIR}/${args_DEST_DIR})
    if(WIN32)
        # on windows, python seems to need standard "\" style paths
        string(REGEX REPLACE "/" "\\\\" abs_dest_path  ${abs_dest_path})
    endif()

    # Use a timestamp file to track when the following pip
    # command was last executed w.r.t. its dependencies.
    set(stamp ${CMAKE_CURRENT_BINARY_DIR}/${args_NAME}.stamp)

    # NOTE: With pip, you can't directly control build dir with an arg
    # like we were able to do with distutils, you have to use TMPDIR
    # TODO: we might want to  explore this in the future
    add_custom_command(OUTPUT ${stamp}
            COMMAND ${Python3_EXECUTABLE} -m pip install . -V 
            --no-cache-dir
            --disable-pip-version-check
            --no-index
            --no-deps
            --no-build-isolation
            --no-warn-script-location
            --upgrade
            --target "${abs_dest_path}"
            COMMAND ${CMAKE_COMMAND} -E touch ${stamp}
            DEPENDS  ${args_PY_SETUP_FILE} ${args_PY_SOURCES}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    # The above pip command wipes the --target directory,
    # so any dependent modules need to be linked afterwards.
    # Propagate this this dependency as a usage requirement.
    add_library(${args_NAME} INTERFACE ${stamp})
    set_property(TARGET ${args_NAME} APPEND PROPERTY INTERFACE_LINK_DEPENDS ${stamp})

    # also use pip for the install ...
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
                COMMAND ${Python3_EXECUTABLE} -m pip install . -V --upgrade
                --disable-pip-version-check --no-warn-script-location
                --no-index --no-deps --no-build-isolation
                --target ${py_mod_inst_prefix}
                RESULT_VARIABLE PY_MODULE_INSTALL_RESULT
                OUTPUT_VARIABLE PY_MODULE_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_MODULE_INSTALL_OUT}\")
            # If pip install failed, that's actually an error we should stop at
            IF(NOT PY_MODULE_INSTALL_RESULT EQUAL 0)
                MESSAGE(FATAL_ERROR \"Staging python module ${args_PY_MODULE_DIR} failed (pip exited \${PY_MODULE_INSTALL_RESULT}); see output above.\")
            ENDIF()
            # If pip succeeded but expected sources are missing, that's an error too
            IF(NOT EXISTS \"${py_mod_inst_prefix}/${args_PY_MODULE_DIR}/__init__.py\")
                MESSAGE(FATAL_ERROR \"Staging python module ${args_PY_MODULE_DIR} produced no python sources (${py_mod_inst_prefix}/${args_PY_MODULE_DIR}/__init__.py is missing).\")
            ENDIF()
            ")
    else()
        # else install to the dest dir under CMAKE_INSTALL_PREFIX
        INSTALL(CODE
            "
            EXECUTE_PROCESS(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${Python3_EXECUTABLE} -m pip install . -V --upgrade
                --disable-pip-version-check --no-warn-script-location
                --no-index --no-deps --no-build-isolation
                --target \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${args_DEST_DIR}
                RESULT_VARIABLE PY_MODULE_INSTALL_RESULT
                OUTPUT_VARIABLE PY_MODULE_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_MODULE_INSTALL_OUT}\")
            # If pip install failed, that's actually an error we should stop at
            IF(NOT PY_MODULE_INSTALL_RESULT EQUAL 0)
                MESSAGE(FATAL_ERROR \"Staging python module ${args_PY_MODULE_DIR} failed (pip exited \${PY_MODULE_INSTALL_RESULT}); see output above.\")
            ENDIF()
            # If pip succeeded but expected sources are missing, that's an error too
            IF(NOT EXISTS \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${args_DEST_DIR}/${args_PY_MODULE_DIR}/__init__.py\")
                MESSAGE(FATAL_ERROR \"Staging python module ${args_PY_MODULE_DIR} produced no python sources (${args_PY_MODULE_DIR}/__init__.py is missing under the install prefix).\")
            ENDIF()
            ")
    endif()

    # set folder if passed
    if(DEFINED args_FOLDER)
        blt_set_target_folder(TARGET ${args_NAME} FOLDER ${args_FOLDER})
    endif()

ENDFUNCTION(PYTHON_ADD_PIP_SETUP)

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
    set(sabi)
    if (ASCENT_PYTHON_USE_LIMITED_API)
        set(sabi USE_SABI 3.8)
    endif ()
    Python3_add_library(${args_NAME} MODULE ${sabi} WITH_SOABI ${args_SOURCES})

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
    
    # link to python as a module
    target_link_libraries(${args_NAME} PRIVATE Python3::Module)

    # support installing the python module components to an
    # an alternate dir, set via PYTHON_MODULE_INSTALL_PREFIX 
    set(py_install_dir ${args_DEST_DIR})
    if(PYTHON_MODULE_INSTALL_PREFIX)
        set(py_install_dir ${PYTHON_MODULE_INSTALL_PREFIX})
    endif()

    install(TARGETS ${args_NAME}
            EXPORT  ascent
            LIBRARY DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
            ARCHIVE DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
            RUNTIME DESTINATION ${py_install_dir}/${args_PY_MODULE_DIR}
    )

ENDFUNCTION(PYTHON_ADD_COMPILED_MODULE)

##############################################################################
# Macro to create a pip script and compiled python module
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

    PYTHON_ADD_PIP_SETUP(NAME          "${args_NAME}_py_setup"
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

    # "${args_NAME}" depends on "${args_NAME}_py_setup"
    target_link_libraries("${args_NAME}" PRIVATE "${args_NAME}_py_setup")

ENDFUNCTION(PYTHON_ADD_HYBRID_MODULE)



