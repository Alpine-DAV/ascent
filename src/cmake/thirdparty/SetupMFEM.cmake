###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Setup MFEM
#
###############################################################################

# first Check for MFEM_DIR

if(NOT MFEM_DIR)
    MESSAGE(FATAL_ERROR "MFEM support needs explicit MFEM_DIR")
endif()

MESSAGE(STATUS "Looking for MFEM using MFEM_DIR = ${MFEM_DIR}")


# when mfem is built w/o cmake, we can get the details of deps from its
# config.mk file
find_path(MFEM_CFG_DIR config.mk
          PATHS ${MFEM_DIR}/share/mfem/
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

if(NOT MFEM_CFG_DIR)
    MESSAGE(FATAL_ERROR "Failed to find MFEM share/mfem/config.mk")
endif()

# read config.mk file
file(READ "${MFEM_CFG_DIR}/config.mk" mfem_cfg_file_txt)

# parse include flags
string(REGEX MATCHALL "MFEM_TPLFLAGS .+\n" mfem_tpl_inc_flags ${mfem_cfg_file_txt})
string(REGEX REPLACE  "MFEM_TPLFLAGS +=" "" mfem_tpl_inc_flags ${mfem_tpl_inc_flags})
string(FIND  ${mfem_tpl_inc_flags} "\n" mfem_tpl_inc_flags_end_pos)
string(SUBSTRING ${mfem_tpl_inc_flags} 0 ${mfem_tpl_inc_flags_end_pos} mfem_tpl_inc_flags)
string(STRIP ${mfem_tpl_inc_flags} mfem_tpl_inc_flags)
# this must b be a list style var, otherwise blt/cmake will quote it
# some where down the line and undermine the flags
string (REPLACE " " ";" mfem_tpl_inc_flags "${mfem_tpl_inc_flags}")

# parse link flags
string(REGEX MATCHALL "MFEM_EXT_LIBS .+\n" mfem_tpl_lnk_flags ${mfem_cfg_file_txt})
string(REGEX REPLACE  "MFEM_EXT_LIBS +=" "" mfem_tpl_lnk_flags ${mfem_tpl_lnk_flags})
string(FIND  ${mfem_tpl_lnk_flags} "\n" mfem_tpl_lnl_flags_end_pos )
string(SUBSTRING ${mfem_tpl_lnk_flags} 0 ${mfem_tpl_lnl_flags_end_pos} mfem_tpl_lnk_flags)

string(FIND  ${mfem_tpl_lnk_flags} "\n" mfem_tpl_lnl_flags_end_pos )
string(SUBSTRING ${mfem_tpl_lnk_flags} 0 ${mfem_tpl_lnl_flags_end_pos} mfem_tpl_lnk_flags)

# filter out any -L s to system libs, they can appear
# at wrong oder on link line, leading to accidental
# inclusion of system libs vs those we want
# note: we may discover other cases we need to defend against
string(REGEX REPLACE "\-L\/lib64" ""  mfem_tpl_lnk_flags ${mfem_tpl_lnk_flags})
string(STRIP ${mfem_tpl_lnk_flags} mfem_tpl_lnk_flags)



# make sure mfem was built with with conduit support:
message(STATUS "Checking for MFEM conduit support")
string(REGEX MATCHALL "MFEM_USE_CONDUIT += +YES" mfem_use_conduit ${mfem_cfg_file_txt})
if(mfem_use_conduit STREQUAL "")
    message(FATAL_ERROR "MFEM config.mk missing MFEM_USE_CONDUIT = YES")
else()
    message(STATUS "Found MFEM_USE_CONDUIT = YES in MFEM config.mk")
endif()


# see if mfem was built with mpi support, if so we need to propgate mpi deps
# even for serial case
string(REGEX MATCHALL "MFEM_USE_MPI += +YES" mfem_use_mpi ${mfem_cfg_file_txt})

if(mfem_use_mpi STREQUAL "")
    set(MFEM_MPI_ENABLED FALSE)
else()
    set(MFEM_MPI_ENABLED TRUE)
endif()


# next check for ZLIB_DIR 
# (spack builds of mfem now depend on zlib, and that is not propgating)
#
# TODO: Decide if we want to be strict about this
# if(NOT ZLIB_DIR)
#     MESSAGE(FATAL_ERROR "MFEM support needs explicit ZLIB_DIR")
# endif()
#

if(ZLIB_DIR)
    set(ZLIB_ROOT ${ZLIB_DIR})
    find_package(ZLIB REQUIRED)
endif()

#
# Add ZLIB to the mfem_tpl_lnk_flags
#
if(ZLIB_FOUND)
    list(APPEND mfem_tpl_lnk_flags ${ZLIB_LIBRARIES})
endif()

#find includes
find_path(MFEM_INCLUDE_DIRS mfem.hpp
          PATHS ${MFEM_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

#find libs
find_library(MFEM_LIBRARIES LIBRARIES NAMES mfem
             PATHS ${MFEM_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


if(MFEM_LIBRARIES)
    include(FindPackageHandleStandardArgs)
    # handle the QUIETLY and REQUIRED arguments and set MFEM_FOUND to TRUE
    # if all listed variables are TRUE
    find_package_handle_standard_args(MFEM DEFAULT_MSG
                                      MFEM_LIBRARIES MFEM_INCLUDE_DIRS)
endif()

# add mpi if mfem uses mpi
if(MFEM_MPI_ENABLED)
    if(NOT MPI_FOUND)
    message(FATAL_ERROR "MFEM was build with MPI support (config.mk has MFEM_USE_MPI = YES)"
                        " but MPI::MPI_CXX target is missing.")
    endif()
    list(APPEND MFEM_LIBRARIES MPI::MPI_CXX)
endif()

if(NOT MFEM_FOUND)
    message(FATAL_ERROR "MFEM_FOUND is not a path to a valid MFEM install")
endif()

    # assume mfem is built with mpi support for now
blt_register_library(NAME mfem
                     INCLUDES ${MFEM_INCLUDE_DIRS}
                     COMPILE_FLAGS ${mfem_tpl_inc_flags}
                     LIBRARIES ${MFEM_LIBRARIES} ${mfem_tpl_lnk_flags})
