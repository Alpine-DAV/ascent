###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

set(ASCENT_INCLUDE_DIRS "${ASCENT_INSTALL_PREFIX}/include/ascent")

#
# Probe Ascent Features
#

if(ASCENT_SERIAL_ENABLED)
    # create convenience target that bundles all reg ascent deps (ascent::ascent)
    add_library(ascent::ascent INTERFACE IMPORTED)

    set_property(TARGET ascent::ascent
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

    set_property(TARGET ascent::ascent
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/ascent/")

    set_property(TARGET ascent::ascent
                 PROPERTY INTERFACE_LINK_LIBRARIES
                 ascent)

    # try to include conduit with new exports
    if(TARGET conduit::conduit)
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit::conduit)
    else()
        # if not, bottle conduit
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})

        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit conduit_relay conduit_blueprint)
    endif()

    if(ASCENT_VTKH_ENABLED)
        set_property(TARGET ascent::ascent
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     vtkh)
    endif()

    if(ASCENT_DRAY_ENABLED)
        # we may still have external dray, so guard
        if(NOT TARGET dray::dray)
            # create convenience target that bundles all reg dray deps (dray::dray)
            add_library(dray::dray INTERFACE IMPORTED)

            set_property(TARGET dray::dray
                         APPEND PROPERTY
                         INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

            set_property(TARGET dray::dray
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         dray)
        endif()
     endif()

     if(ASCENT_VTKH_ENABLED)
         # we may still have external, so guard
         if(NOT TARGET vtkh::vtkh)
             # create convenience target that bundles all reg deps
             add_library(vtkh::vtkh INTERFACE IMPORTED)

             set_property(TARGET vtkh::vtkh
                          APPEND PROPERTY
                          INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

             set_property(TARGET vtkh::vtkh
                          PROPERTY INTERFACE_LINK_LIBRARIES
                          vtkh)
         endif()
      endif()
     
endif()

# and if mpi enabled, a convenience target mpi case (ascent::cascent_mpi)
if(ASCENT_MPI_ENABLED)
    add_library(ascent::ascent_mpi INTERFACE IMPORTED)

    set_property(TARGET ascent::ascent_mpi
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

    set_property(TARGET ascent::ascent_mpi
                 APPEND PROPERTY
                 INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/ascent/")

    set_property(TARGET ascent::ascent_mpi
                 PROPERTY INTERFACE_LINK_LIBRARIES
                 ascent_mpi)
    # try to include conduit with new exports
    if(TARGET conduit::conduit)
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit::conduit conduit::conduit_mpi)
    else()
        # if not, bottle conduit
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY
                     INTERFACE_INCLUDE_DIRECTORIES ${CONDUIT_INCLUDE_DIRS})

        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     conduit conduit_relay conduit_relay_mpi conduit_blueprint conduit_blueprint_mpi)
    endif()

    if(ASCENT_VTKH_ENABLED)
        set_property(TARGET ascent::ascent_mpi
                     APPEND PROPERTY INTERFACE_LINK_LIBRARIES
                     vtkh_mpi)
    endif()


    if(ASCENT_DRAY_ENABLED)
        # we may still have external dray, so guard
        if(NOT TARGET dray::dray_mpi)
            # create convenience target that bundles all reg dray deps (dray::dray)
            add_library(dray::dray_mpi INTERFACE IMPORTED)

            set_property(TARGET dray::dray_mpi
                         APPEND PROPERTY
                         INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

            set_property(TARGET dray::dray_mpi
                         PROPERTY INTERFACE_LINK_LIBRARIES
                         dray_mpi)
         endif()
     endif()

     if(ASCENT_VTKH_ENABLED)
         # we may still have external, so guard
         if(NOT TARGET vtkh::vtkh_mpi)
             # create convenience target that bundles all reg deps
             add_library(vtkh::vtkh_mpi INTERFACE IMPORTED)

             set_property(TARGET vtkh::vtkh_mpi
                          APPEND PROPERTY
                          INTERFACE_INCLUDE_DIRECTORIES "${ASCENT_INSTALL_PREFIX}/include/")

             set_property(TARGET vtkh::vtkh_mpi
                          PROPERTY INTERFACE_LINK_LIBRARIES
                          vtkh_mpi)
         endif()
      endif()

 endif()

if(NOT Ascent_FIND_QUIETLY)

    message(STATUS "ASCENT_VERSION             = ${ASCENT_VERSION}")
    message(STATUS "ASCENT_INSTALL_PREFIX      = ${ASCENT_INSTALL_PREFIX}")
    message(STATUS "ASCENT_INCLUDE_DIRS        = ${ASCENT_INCLUDE_DIRS}")
    message(STATUS "ASCENT_SERIAL_ENABLED      = ${ASCENT_SERIAL_ENABLED}")
    message(STATUS "ASCENT_MPI_ENABLED         = ${ASCENT_MPI_ENABLED}")
    message(STATUS "ASCENT_FORTRAN_ENABLED     = ${ASCENT_FORTRAN_ENABLED}")
    message(STATUS "ASCENT_VTKH_ENABLED        = ${ASCENT_VTKH_ENABLED}")
    message(STATUS "ASCENT_PYTHON_ENABLED      = ${ASCENT_PYTHON_ENABLED}")
    message(STATUS "ASCENT_PYTHON_EXECUTABLE   = ${ASCENT_PYTHON_EXECUTABLE}")
    message(STATUS "ASCENT_DRAY_ENABLED        = ${ASCENT_DRAY_ENABLED}")
    message(STATUS "ASCENT_APCOMP_ENABLED      = ${ASCENT_APCOMP_ENABLED}")
    message(STATUS "ASCENT_OCCA_ENABLED        = ${ASCENT_OCCA_ENABLED}")
    message(STATUS "ASCENT_UMPIRE_ENABLED      = ${ASCENT_UMPIRE_ENABLED}")
    message(STATUS "ASCENT_BABELFLOW_ENABLED   = ${ASCENT_BABELFLOW_ENABLED}")
    message(STATUS "ASCENT_FIDES_ENABLED       = ${ASCENT_FIDES_ENABLED}")
    message(STATUS "ASCENT_MFEM_ENABLED        = ${ASCENT_MFEM_ENABLED}")
    message(STATUS "ASCENT_MFEM_MPI_ENABLED    = ${ASCENT_MFEM_MPI_ENABLED}")


    set(_print_targets "")
    if(ASCENT_SERIAL_ENABLED)
        set(_print_targets "ascent::ascent ")
    endif()

    if(ASCENT_MPI_ENABLED)
        set(_print_targets "${_print_targets}ascent::ascent_mpi")
    endif()

    message(STATUS "Ascent imported targets: ${_print_targets}")
    unset(_print_targets)

endif()


