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
# file: src/ascent/CMakeLists.txt
#
###############################################################################

#if(ENABLE_CUDA AND NOT "${BUILD_SHARED_LIBS}" STREQUAL "OFF")
#  message(FATAL_ERROR "Ascent can only be built statically when cuda is enabled")
#endif()

convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX}
                                    ASCENT_INSTALL_PREFIX)

set(ASCENT_INSTALL_PREFIX ${ASCENT_INSTALL_PREFIX} CACHE STRING "" FORCE )

if(FORTRAN_FOUND)
    set(ASCENT_FORTRAN_COMPILER ${CMAKE_Fortran_COMPILER})
endif()

set(ASCENT_SOURCE_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_clients)

set(ASCENT_VTKM_ENABLED ${VTKM_FOUND})
set(ASCENT_VTKH_ENABLED ${VTKM_FOUND})

set(ASCENT_HDF5_ENABLED ${HDF5_FOUND})
set(ASCENT_MFEM_ENABLED ${MFEM_FOUND})
set(ASCENT_DRAY_ENABLED ${DRAY_FOUND})
set(ASCENT_PYTHON_ENABLED ${PYTHON_FOUND})
set(ASCENT_FIDES_ENABLED ${FIDES_FOUND})

if(ASCENT_VTKM_ENABLED)
    set(ASCENT_VTKM_USE_CUDA ${CUDA_FOUND})
    set(ASCENT_VTKM_USE_OPENMP ${OPENMP_FOUND})
endif()

set(ASCENT_USE_OPENMP  ${OPENMP_FOUND})

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h")

set(ascent_sources
    # ascent interface
    ascent.cpp
    c/ascent_c.cpp
    # runtimes
    ascent_runtime.cpp
    runtimes/ascent_main_runtime.cpp
    runtimes/ascent_metadata.cpp
    runtimes/ascent_empty_runtime.cpp
    runtimes/ascent_expression_eval.cpp
    runtimes/ascent_transmogrifier.cpp
    runtimes/expressions/ascent_blueprint_architect.cpp
    runtimes/expressions/ascent_conduit_reductions.cpp
    runtimes/expressions/ascent_expression_filters.cpp
    runtimes/expressions/ascent_expressions_ast.cpp
    runtimes/expressions/ascent_expressions_tokens.cpp
    runtimes/expressions/ascent_expressions_parser.cpp
    runtimes/ascent_flow_runtime.cpp
    runtimes/flow_filters/ascent_runtime_filters.cpp
    runtimes/flow_filters/ascent_runtime_param_check.cpp
    runtimes/flow_filters/ascent_runtime_relay_filters.cpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.cpp
    runtimes/flow_filters/ascent_runtime_trigger_filters.cpp
    runtimes/flow_filters/ascent_runtime_query_filters.cpp
    runtimes/flow_filters/ascent_runtime_utils.cpp
    # utils
    utils/ascent_actions_utils.cpp
    utils/ascent_file_system.cpp
    utils/ascent_block_timer.cpp
    utils/ascent_logging.cpp
    utils/ascent_png_compare.cpp
    utils/ascent_png_decoder.cpp
    utils/ascent_png_encoder.cpp
    utils/ascent_mpi_utils.cpp
    utils/ascent_string_utils.cpp
    utils/ascent_web_interface.cpp
    utils/ascent_resources.cpp
    # hola
    hola/ascent_hola.cpp)

set(ascent_headers
    # ascent interface
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h
    ascent_exports.h
    ascent.hpp
    c/ascent.h
    # runtimes
    ascent_runtime.hpp
    runtimes/ascent_empty_runtime.hpp
    runtimes/ascent_expression_eval.hpp
    runtimes/ascent_transmogrifier.hpp
    runtimes/expressions/ascent_blueprint_architect.hpp
    runtimes/expressions/ascent_conduit_reductions.hpp
    runtimes/expressions/ascent_expression_filters.hpp
    runtimes/expressions/ascent_expressions_ast.hpp
    runtimes/expressions/ascent_expressions_tokens.hpp
    runtimes/expressions/ascent_expressions_parser.hpp
    # flow
    runtimes/ascent_main_runtime.hpp
    runtimes/ascent_metadata.hpp
    runtimes/ascent_flow_runtime.hpp
    runtimes/flow_filters/ascent_runtime_filters.hpp
    runtimes/flow_filters/ascent_runtime_param_check.hpp
    runtimes/flow_filters/ascent_runtime_relay_filters.hpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.hpp
    runtimes/flow_filters/ascent_runtime_trigger_filters.hpp
    runtimes/flow_filters/ascent_runtime_query_filters.hpp
    runtimes/flow_filters/ascent_runtime_vtkh_utils.hpp
    runtimes/flow_filters/ascent_runtime_utils.hpp
    # utils
    utils/ascent_actions_utils.hpp
    utils/ascent_logging.hpp
    utils/ascent_file_system.hpp
    utils/ascent_block_timer.hpp
    utils/ascent_png_compare.hpp
    utils/ascent_png_decoder.hpp
    utils/ascent_png_encoder.hpp
    utils/ascent_mpi_utils.hpp
    utils/ascent_string_utils.hpp
    utils/ascent_web_interface.hpp
    utils/ascent_resources.hpp
    utils/ascent_resources_cinema_web.hpp
    # hola
    hola/ascent_hola.hpp)

if(VTKM_FOUND)

    set(ascent_vtkh_dep_headers
        runtimes/ascent_data_object.hpp
        runtimes/ascent_vtkh_data_adapter.hpp
        runtimes/ascent_vtkh_collection.hpp
        runtimes/flow_filters/ascent_runtime_vtkh_filters.hpp
        runtimes/flow_filters/ascent_runtime_vtkh_utils.hpp
        runtimes/flow_filters/ascent_runtime_rendering_filters.hpp
        runtimes/flow_filters/ascent_runtime_rover_filters.hpp
        runtimes/flow_filters/ascent_runtime_conduit_to_vtkm_parsing.hpp
        runtimes/ascent_main_runtime.hpp)

    set(ascent_vtkh_dep_sources
        runtimes/ascent_data_object.cpp
        runtimes/ascent_vtkh_data_adapter.cpp
        runtimes/ascent_vtkh_collection.cpp
        runtimes/flow_filters/ascent_runtime_vtkh_filters.cpp
        runtimes/flow_filters/ascent_runtime_vtkh_utils.cpp
        runtimes/flow_filters/ascent_runtime_rendering_filters.cpp
        runtimes/flow_filters/ascent_runtime_rover_filters.cpp
        runtimes/flow_filters/ascent_runtime_conduit_to_vtkm_parsing.cpp
        runtimes/ascent_main_runtime.cpp)

    foreach( _itm ${ascent_vtkh_dep_headers} )
        list(APPEND ascent_headers ${_itm})
    endforeach()

    foreach( _itm ${ascent_vtkh_dep_sources} )
        list(APPEND ascent_sources ${_itm})
    endforeach()
else()
    list(APPEND ascent_headers runtimes/ascent_data_object.hpp)
    list(APPEND ascent_sources runtimes/ascent_data_object.cpp)
endif()

if(MFEM_FOUND)
    # data adapter
    list(APPEND ascent_headers runtimes/ascent_mfem_data_adapter.hpp)
    list(APPEND ascent_sources runtimes/ascent_mfem_data_adapter.cpp)
endif()

if(DRAY_FOUND)
  list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_dray_filters.hpp)
  list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_dray_filters.cpp)
endif()

##############################
# optional features
#############################

if(FORTRAN_FOUND)
    add_library(ascent_fortran OBJECT fortran/ascent_fortran.f90)
    list(APPEND ascent_sources $<TARGET_OBJECTS:ascent_fortran>)

    #####################################################
    # Setup install to copy the fortran modules
    #####################################################
    install(FILES
            ${CMAKE_Fortran_MODULE_DIRECTORY}/ascent.mod
            DESTINATION include/ascent)
endif()



################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
   add_subdirectory(python)

   list(APPEND ascent_sources
        runtimes/flow_filters/ascent_python_script_filter.cpp)

   list(APPEND ascent_headers
        runtimes/flow_filters/ascent_python_script_filter.hpp)
endif()

##################
# req'd libs
##################

set(ascent_thirdparty_libs
    conduit
    ascent_flow
    ascent_lodepng)

##################
# optional libs
##################
if(VTKM_FOUND)
    list(APPEND ascent_thirdparty_libs vtkm vtkh rover)
endif()

if(MFEM_FOUND)
    list(APPEND ascent_thirdparty_libs mfem)
    # workaround for MPI headers
    if(ENABLE_MPI)
        list(APPEND ascent_thirdparty_libs mpi)
    endif()
endif()

if(CUDA_FOUND)
    list(APPEND ascent_thirdparty_libs cuda_runtime)
endif()

if(OPENMP_FOUND)
    list(APPEND ascent_thirdparty_libs openmp)
endif()

if(DRAY_FOUND)
    list(APPEND ascent_thirdparty_libs dray)
endif()


##########################################
# Build a serial version of ascent
##########################################
if (ENABLE_SERIAL)
    blt_add_library(
        NAME        ascent
        SOURCES     ${ascent_sources}
        HEADERS     ${ascent_headers}
        DEPENDS_ON  ${ascent_thirdparty_libs})

    set_target_properties(ascent PROPERTIES CXX_VISIBILITY_PRESET hidden)
    target_compile_definitions(ascent PRIVATE ASCENT_EXPORTS_FLAG)

    if(VTKM_FOUND)
        set(ascent_device_sources ${ascent_vtkh_dep_sources})
        list(APPEND ascent_device_sources runtimes/flow_filters/ascent_runtime_blueprint_filters.cpp)

        vtkm_add_target_information(ascent DEVICE_SOURCES ${ascent_device_sources})
    endif()
    # install target for serial ascent lib
    install(TARGETS ascent
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )


    if(VTKM_FOUND AND CUDA_FOUND)
      set(locations_on_disk "")

      set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ascent.dir")

      set(targets_to_link rover
                          vtkh_rendering
                          vtkh_compositing
                          vtkh_filters
                          vtkh_core
                          vtkh_lodepng
                          vtkh_utils
                          vtkm_io
                          vtkm_rendering
                          vtkm_filter_common
                          vtkm_filter_contour
                          vtkm_filter_gradient
                          vtkm_filter_extra
                          vtkm_worklet
                          vtkm_source
                          vtkm_cont
                          vtkmdiympi_nompi
                          )

      source_objects(${binary_dir}
                     "${ascent_device_sources}"
                     "${CMAKE_BINARY_DIR}/lib/librover.a"
                     "${targets_to_link}"
                     locations_on_disk)

      separate_arguments(params NATIVE_COMMAND "${CMAKE_CUDA_FLAGS} ${locations_on_disk}")
      add_custom_command(
        TARGET ascent PRE_LINK
        DEPENDS ascent rover
        COMMAND ${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm.o
        COMMENT "manual device link step for Ascent"
        )
      install(FILES ${CMAKE_BINARY_DIR}/ascent/bcal_vtkm.o DESTINATION lib)
    endif()

endif()

################################################
# Build Parallel (MPI) version of ascent
################################################
if(MPI_FOUND)

    # req'd libs
    set(ascent_mpi_thirdparty_libs mpi conduit_relay_mpi)

    if (ENABLE_BABELFLOW)
	    if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
		    add_compile_definitions("BABELFLOW_DEBUG")
	    endif()

      list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_babelflow_filters.hpp)
      list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_pmt.cpp)
      list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_comp_utils.hpp)
      list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_comp_utils.cpp)
      list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_compose.cpp)

      list(APPEND ascent_mpi_thirdparty_libs BabelFlow::babelflow BabelFlow::babelflow_mpi)
      list(APPEND ascent_mpi_thirdparty_libs PMT::pmt)
      add_compile_definitions(ASCENT_BABELFLOW_ENABLED)
      include_directories(${BABELFLOW_INCLUDE_DIRS})
      include_directories(${PMT_INCLUDE_DIRS})
    endif ()

    # hola mpi support is only in par (mpi ver)

    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_hola_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_hola_filters.cpp)

    list(APPEND ascent_headers hola/ascent_hola_mpi.hpp)
    list(APPEND ascent_sources hola/ascent_hola_mpi.cpp)

    # adios support is only in par (mpi ver)
    if (ASCENT_FIDES_ENABLED AND MPI_FOUND)
       list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_adios2_filters.hpp)
       list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_adios2_filters.cpp)
       list(APPEND ascent_mpi_thirdparty_libs adios2::adios2 fides)
    endif()

    list(APPEND ascent_mpi_thirdparty_libs ${ascent_thirdparty_libs})

    if(VTKM_FOUND)
        if (ENABLE_BABELFLOW)
            list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_iso.cpp)
        endif()

        # swap vtkh_mpi for vtkh
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs vtkh)
        list(APPEND ascent_mpi_thirdparty_libs vtkh_mpi)
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs rover)
        list(APPEND ascent_mpi_thirdparty_libs rover_mpi)
    endif()
    if(DRAY_FOUND)
        # swap dray_mpi for dray
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs dray)
        list(APPEND ascent_mpi_thirdparty_libs dray_mpi)
    endif()

    blt_add_library(
        NAME        ascent_mpi
        SOURCES     ${ascent_sources} ${ascent_mpi_sources}
        HEADERS     ${ascent_headers} ${ascent_mpi_headers}
        DEPENDS_ON  ${ascent_mpi_thirdparty_libs})

    set_target_properties(ascent_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    target_compile_definitions(ascent_mpi PRIVATE ASCENT_EXPORTS_FLAG)

    if(VTKM_FOUND)
        vtkm_add_target_information(ascent_mpi DEVICE_SOURCES ${ascent_device_sources})
    endif()
    blt_add_target_compile_flags(TO ascent_mpi FLAGS "-DASCENT_MPI_ENABLED")

    # install target for mpi ver
    install(TARGETS ascent_mpi
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
    )

    if(VTKM_FOUND AND CUDA_FOUND)
      set(locations_on_disk "")

      set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ascent_mpi.dir")

      set(targets_to_link rover
                          vtkh_rendering_mpi
                          vtkh_compositing_mpi
                          vtkh_filters_mpi
                          vtkh_core_mpi
                          vtkh_lodepng
                          vtkh_utils_mpi
                          vtkm_io
                          vtkm_rendering
                          vtkm_filter_common
                          vtkm_filter_contour
                          vtkm_filter_gradient
                          vtkm_filter_extra
                          vtkm_worklet
                          vtkm_source
                          vtkm_cont
                          vtkmdiympi_nompi
                          )

      source_objects(${binary_dir}
                     "${ascent_device_sources}"
                     "${CMAKE_BINARY_DIR}/lib/librover_mpi.a"
                     "${targets_to_link}"
                     locations_on_disk)

      separate_arguments(params NATIVE_COMMAND "${CMAKE_CUDA_FLAGS} ${locations_on_disk}")
      #message(STATUS "${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm_mpi.o")
      add_custom_command(
        TARGET ascent_mpi PRE_LINK
        DEPENDS ascent_mpi rover_mpi
        COMMAND ${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm_mpi.o
        COMMENT "manual device link step for Ascent parallel"
        )
      install(FILES ${CMAKE_BINARY_DIR}/ascent/bcal_vtkm_mpi.o DESTINATION lib)
    endif()

endif()

########################################
# Extra Install Targets for ascent
########################################
install(FILES ${ascent_headers} DESTINATION include/ascent)

##############################################################
# install files needed for cinema viewer and web client
#############################################################
install(DIRECTORY web_clients/
        DESTINATION share/ascent/web_clients/)
