###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################


###############################################################################
#
# file: src/libs/ascent/CMakeLists.txt
#
###############################################################################


###############################################################################
# Capture config settings into derived CMake vars with ASCENT_ prefix
# these are used to generate headers, our CMake exports, and ascent.mk 
###############################################################################

convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX}
                                    ASCENT_INSTALL_PREFIX)

set(ASCENT_INSTALL_PREFIX ${ASCENT_INSTALL_PREFIX} CACHE STRING "" FORCE )

set(ASCENT_FORTRAN_ENABLED ${FORTRAN_FOUND})
if(FORTRAN_FOUND)
    set(ASCENT_FORTRAN_COMPILER ${CMAKE_Fortran_COMPILER})
endif()

set(ASCENT_PYTHON_ENABLED   ${PYTHON_FOUND})
set(ASCENT_PYTHON_EXECUTABLE ${PYTHON_FOUND})

set(ASCENT_LOGGING_ENABLED ${ENABLE_LOGGING})

set(ASCENT_SOURCE_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_clients)

set(ASCENT_CAMP_ENABLED   ${CAMP_FOUND})
set(ASCENT_UMPIRE_ENABLED ${UMPIRE_FOUND})
set(ASCENT_RAJA_ENABLED   ${RAJA_FOUND})
set(ASCENT_APCOMP_ENABLED ${ENABLE_APCOMP})
set(ASCENT_DRAY_ENABLED   ${ENABLE_DRAY})

set(ASCENT_VTKH_ENABLED   ${ENABLE_VTKH})
set(ASCENT_VTKM_ENABLED   ${VTKM_FOUND})
set(ASCENT_HDF5_ENABLED   ${CONDUIT_RELAY_HDF5_ENABLED})
set(ASCENT_MFEM_ENABLED   ${MFEM_FOUND})

set(ASCENT_ADIOS2_ENABLED ${ADIOS2_FOUND})
set(ASCENT_FIDES_ENABLED  ${FIDES_FOUND})
set(ASCENT_GENTEN_ENABLED ${GENTEN_FOUND})
set(ASCENT_KOKKOS_ENABLED ${KOKKOS_FOUND})
set(ASCENT_BABELFLOW_ENABLED ${BABELFLOW_FOUND})

if(OCCA_FOUND AND UMPIRE_FOUND)
    set(ASCENT_JIT_ENABLED TRUE)
else()
    set(ASCENT_JIT_ENABLED FALSE)
endif()

set(ASCENT_WEBSERVER_ENABLED ${CONDUIT_RELAY_WEBSERVER_ENABLED})

set(ASCENT_OPENMP_ENABLED ${OPENMP_FOUND})
set(ASCENT_CUDA_ENABLED   ${CUDA_FOUND})
set(ASCENT_HIP_ENABLED    ${HIP_FOUND})

if(ASCENT_VTKM_ENABLED)
    set(ASCENT_VTKM_OPENMP_ENABLED ${OPENMP_FOUND})
    set(ASCENT_VTKM_CUDA_ENABLED   ${CUDA_FOUND})
    set(ASCENT_VTKM_KOKKOS_ENABLED ${HIP_FOUND})
endif()

# ASCENT_ZZZ_ENTIRES are derived vars.
# We mark them as advanced b/c otherwise folks may see them
# in ccmake or the cmake gui and think they should be changing them
mark_as_advanced(ASCENT_INSTALL_PREFIX
                 ASCENT_FORTRAN_ENABLED
                 ASCENT_FORTRAN_COMPILER
                 ASCENT_PYTHON_ENABLED
                 ASCENT_PYTHON_EXECUTABLE
                 ASCENT_LOGGING_ENABLED
                 ASCENT_SOURCE_WEB_CLIENT_ROOT
                 ASCENT_CAMP_ENABLED
                 ASCENT_UMPIRE_ENABLED
                 ASCENT_RAJA_ENABLED
                 ASCENT_APCOMP_ENABLED
                 ASCENT_DRAY_ENABLED
                 ASCENT_VTKH_ENABLED
                 ASCENT_VTKM_ENABLED
                 ASCENT_HDF5_ENABLED
                 ASCENT_MFEM_ENABLED
                 ASCENT_ADIOS2_ENABLED
                 ASCENT_FIDES_ENABLED
                 ASCENT_GENTEN_ENABLED
                 ASCENT_KOKKOS_ENABLED
                 ASCENT_JIT_ENABLED
                 ASCENT_WEBSERVER_ENABLED
                 ASCENT_OPENMP_ENABLED
                 ASCENT_CUDA_ENABLED
                 ASCENT_HIP_ENABLED
                 ASCENT_VTKM_OPENMP_ENABLED
                 ASCENT_VTKM_CUDA_ENABLED
                 ASCENT_VTKM_KOKKOS_ENABLED
                 )

# gen config header
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/ascent_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h")

###############################################################################
# setup base source lists
###############################################################################

set(ascent_headers
    # ascent interface
    ${CMAKE_CURRENT_BINARY_DIR}/ascent_config.h
    ascent_exports.h
    ascent.hpp
    c/ascent.h
    # runtimes
    ascent_runtime.hpp
    runtimes/ascent_empty_runtime.hpp
    runtimes/ascent_flow_runtime.hpp
    runtimes/ascent_main_runtime.hpp
    runtimes/ascent_data_object.hpp
    runtimes/ascent_metadata.hpp
    runtimes/ascent_transmogrifier.hpp
    # expressions
    runtimes/ascent_expression_eval.hpp
    runtimes/expressions/ascent_blueprint_architect.hpp
    runtimes/expressions/ascent_blueprint_topologies.hpp
    runtimes/expressions/ascent_conduit_reductions.hpp
    runtimes/expressions/ascent_expression_filters.hpp
    runtimes/expressions/ascent_expressions_ast.hpp
    runtimes/expressions/ascent_expressions_tokens.hpp
    runtimes/expressions/ascent_expressions_parser.hpp
    runtimes/expressions/ascent_math.hpp
    runtimes/expressions/ascent_dispatch.hpp
    runtimes/expressions/ascent_meshes.hpp
    runtimes/expressions/ascent_array.hpp
    runtimes/expressions/ascent_array_utils.hpp
    runtimes/expressions/ascent_array_internals.hpp
    runtimes/expressions/ascent_array_internals_base.hpp
    runtimes/expressions/ascent_array_registry.hpp
    runtimes/expressions/ascent_data_binning.hpp
    runtimes/expressions/ascent_memory_interface.hpp
    runtimes/expressions/ascent_memory_manager.hpp
    runtimes/expressions/ascent_execution_policies.hpp
    runtimes/expressions/ascent_execution_manager.hpp
    runtimes/expressions/ascent_derived_jit.hpp
    runtimes/expressions/ascent_jit_array.hpp
    runtimes/expressions/ascent_jit_field.hpp
    runtimes/expressions/ascent_jit_fusion.hpp
    runtimes/expressions/ascent_jit_kernel.hpp
    runtimes/expressions/ascent_jit_math.hpp
    runtimes/expressions/ascent_jit_topology.hpp
    runtimes/expressions/ascent_insertion_ordered_set.hpp
    runtimes/expressions/ascent_expression_jit_filters.hpp
    # flow
    runtimes/flow_filters/ascent_runtime_filters.hpp
    runtimes/flow_filters/ascent_runtime_param_check.hpp
    runtimes/flow_filters/ascent_runtime_relay_filters.hpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.hpp
    runtimes/flow_filters/ascent_runtime_htg_filters.hpp
    runtimes/flow_filters/ascent_runtime_trigger_filters.hpp
    runtimes/flow_filters/ascent_runtime_query_filters.hpp
    runtimes/flow_filters/ascent_runtime_vtkh_utils.hpp
    runtimes/flow_filters/ascent_runtime_utils.hpp
    # utils
    utils/ascent_actions_utils.hpp
    utils/ascent_data_logger.hpp
    utils/ascent_logging.hpp
    utils/ascent_block_timer.hpp
    utils/ascent_mpi_utils.hpp
    utils/ascent_string_utils.hpp
    utils/ascent_web_interface.hpp
    utils/ascent_resources.hpp
    utils/ascent_resources_cinema_web.hpp
    utils/ascent_resources_ascent_web.hpp
    utils/ascent_annotations.hpp
    # hola
    hola/ascent_hola.hpp)

set(ascent_sources
    # ascent interface
    ascent.cpp
    c/ascent_c.cpp
    # runtimes
    ascent_runtime.cpp
    runtimes/ascent_empty_runtime.cpp
    runtimes/ascent_flow_runtime.cpp
    runtimes/ascent_main_runtime.cpp
    runtimes/ascent_data_object.cpp
    runtimes/ascent_metadata.cpp
    runtimes/ascent_transmogrifier.cpp
    # expressions
    runtimes/ascent_expression_eval.cpp
    runtimes/expressions/ascent_blueprint_architect.cpp
    runtimes/expressions/ascent_blueprint_topologies.cpp
    runtimes/expressions/ascent_conduit_reductions.cpp
    runtimes/expressions/ascent_expression_filters.cpp
    runtimes/expressions/ascent_expressions_ast.cpp
    runtimes/expressions/ascent_expressions_tokens.cpp
    runtimes/expressions/ascent_expressions_parser.cpp
    runtimes/expressions/ascent_memory_manager.cpp
    runtimes/expressions/ascent_array.cpp
    runtimes/expressions/ascent_array_internals.cpp
    runtimes/expressions/ascent_array_internals_base.cpp
    runtimes/expressions/ascent_array_registry.cpp
    runtimes/expressions/ascent_array_utils.cpp
    runtimes/expressions/ascent_data_binning.cpp
    runtimes/expressions/ascent_execution_policies.cpp
    runtimes/expressions/ascent_execution_manager.cpp
    runtimes/expressions/ascent_derived_jit.cpp
    runtimes/expressions/ascent_jit_array.cpp
    runtimes/expressions/ascent_jit_field.cpp
    runtimes/expressions/ascent_jit_fusion.cpp
    runtimes/expressions/ascent_jit_kernel.cpp
    runtimes/expressions/ascent_jit_math.cpp
    runtimes/expressions/ascent_jit_topology.cpp
    runtimes/expressions/ascent_insertion_ordered_set.cpp
    runtimes/expressions/ascent_expression_jit_filters.cpp
    # filters (other filters are added later based on enabled tpls)
    runtimes/flow_filters/ascent_runtime_filters.cpp
    runtimes/flow_filters/ascent_runtime_param_check.cpp
    runtimes/flow_filters/ascent_runtime_relay_filters.cpp
    runtimes/flow_filters/ascent_runtime_blueprint_filters.cpp
    runtimes/flow_filters/ascent_runtime_htg_filters.cpp
    runtimes/flow_filters/ascent_runtime_trigger_filters.cpp
    runtimes/flow_filters/ascent_runtime_query_filters.cpp
    runtimes/flow_filters/ascent_runtime_utils.cpp
    # utils
    utils/ascent_actions_utils.cpp
    utils/ascent_data_logger.cpp
    utils/ascent_block_timer.cpp
    utils/ascent_logging.cpp
    utils/ascent_mpi_utils.cpp
    utils/ascent_string_utils.cpp
    utils/ascent_web_interface.cpp
    utils/ascent_resources.cpp
    utils/ascent_annotations.cpp
    # hola
    hola/ascent_hola.cpp)

# explicit list of any sources that need to be device linked
set(ascent_device_sources runtimes/expressions/ascent_memory_manager.cpp
                          runtimes/expressions/ascent_conduit_reductions.cpp
                          runtimes/expressions/ascent_array.cpp
                          runtimes/expressions/ascent_array_utils.cpp
                          runtimes/expressions/ascent_array_internals.cpp
                          runtimes/expressions/ascent_data_binning.cpp
                          runtimes/expressions/ascent_execution_policies.cpp
                          runtimes/expressions/ascent_blueprint_architect.cpp
                          runtimes/expressions/ascent_derived_jit.cpp 
                          )

##############################
# optional features
#############################

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

    list(APPEND ascent_headers ${ascent_vtkh_dep_headers})
    list(APPEND ascent_sources ${ascent_vtkh_dep_sources})

    # keep a vtkm-specific device source list
    set(ascent_vtkm_device_sources ${ascent_vtkh_dep_sources}
                                   runtimes/flow_filters/ascent_runtime_blueprint_filters.cpp)

    # also make sure that these are in ascent's device source list
    list(APPEND ascent_device_sources ${ascent_vtkm_device_sources})
endif()


if(MFEM_FOUND)
    # mfem data adapter
    list(APPEND ascent_headers runtimes/ascent_mfem_data_adapter.hpp)
    list(APPEND ascent_sources runtimes/ascent_mfem_data_adapter.cpp)
    # no mfem device sources
endif()

if(ENABLE_DRAY)
    # dray filters
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_dray_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_dray_filters.cpp)
    # tag device sources
    list(APPEND ascent_device_sources runtimes/expressions/ascent_expression_filters.cpp)
    list(APPEND ascent_device_sources runtimes/flow_filters/ascent_runtime_dray_filters.cpp)
endif()

if(GENTEN_FOUND)
    # genten filters
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_genten_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_genten_filters.cpp)
    # tag device sources
    list(APPEND ascent_device_sources runtimes/flow_filters/ascent_runtime_genten_filters.cpp)
endif()

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


##################################
# flag all device sources
##################################
if(CUDA_FOUND)
  set_source_files_properties(${ascent_device_sources} PROPERTIES LANGUAGE CUDA)
endif()

if(HIP_FOUND)
  set_source_files_properties(${ascent_device_sources} PROPERTIES LANGUAGE HIP)
endif()

##################
# req'd libs
##################
set(ascent_thirdparty_libs
    conduit::conduit
    ascent_flow
    ascent_lodepng
    ascent_png_utils)

##################
# optional libs
##################
if(OCCA_FOUND)
    list(APPEND ascent_thirdparty_libs occa)
endif()

if(RAJA_FOUND)
    list(APPEND ascent_thirdparty_libs RAJA)
endif()

if(UMPIRE_FOUND)
    list(APPEND ascent_thirdparty_libs umpire)
endif()

if(VTKM_FOUND)
    list(APPEND ascent_thirdparty_libs vtkm vtkh rover)
endif()

if(CALIPER_FOUND)
    list(APPEND ascent_thirdparty_libs caliper)
endif()

if(MFEM_FOUND)
    list(APPEND ascent_thirdparty_libs mfem)
    # mfem may require MPI, even for serial builds
    if(MFEM_MPI_ENABLED)
        list(APPEND ascent_thirdparty_libs ${ascent_blt_mpi_deps})
    endif()
endif()

if(CUDA_FOUND)
    list(APPEND ascent_thirdparty_libs cuda_runtime)
    # record some extra libs for our config.mk
    # dl and rt
    set(ASCENT_CUDA_MAKE_EXTRA_LIBS "-ldl -lrt "  CACHE STRING "" FORCE)
    # pthreads
    if(CMAKE_THREAD_LIBS_INIT)
        set(ASCENT_CUDA_MAKE_EXTRA_LIBS "${ASCENT_CUDA_MAKE_EXTRA_LIBS} ${CMAKE_THREAD_LIBS_INIT}"  CACHE STRING "" FORCE)
    endif()
endif()

if(HIP_FOUND)
    list(APPEND ascent_thirdparty_libs blt_hip_runtime)
endif()

if(OPENMP_FOUND)
    list(APPEND ascent_thirdparty_libs ${ascent_blt_openmp_deps})
endif()

if(ENABLE_DRAY)
    list(APPEND ascent_thirdparty_libs dray)
endif()

if(GENTEN_FOUND)
    list(APPEND ascent_thirdparty_libs genten)
endif()

##########################################
# Build a serial version of ascent
##########################################
if (ENABLE_SERIAL)
    blt_add_library(NAME        ascent
                    SOURCES     ${ascent_sources}
                    HEADERS     ${ascent_headers}
                    DEPENDS_ON  ${ascent_thirdparty_libs})

    #  compile defs
    target_compile_definitions(ascent PRIVATE ASCENT_EXPORTS_FLAG)

    # symbol visibility
    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(ascent PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # special device source logic for vtk-m features
    if(VTKM_FOUND)
        vtkm_add_target_information(ascent DEVICE_SOURCES ${ascent_vtkm_device_sources})
    endif()

    # build includes
    # for generated config header
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
    # this allows us to include as <header.hpp> during builds
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/>)
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/c>)
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/utils>)
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/runtimes>)
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/hola>)
    target_include_directories(ascent PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/runtimes/flow_filters>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(ascent PUBLIC $<INSTALL_INTERFACE:include/ascent/>)

    # install target for serial ascent lib
    install(TARGETS ascent
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
            )

    # manual device link for CUDA
    if(VTKM_FOUND AND CUDA_FOUND AND NOT BUILD_SHARED_LIBS)
        set(internal_targets_to_link $<TARGET_FILE:rover>)

        if(ENABLE_VTKH)
          list(APPEND internal_targets_to_link
                      $<TARGET_FILE:vtkh_rendering>
                      $<TARGET_FILE:vtkh_compositing>
                      $<TARGET_FILE:vtkh_filters>
                      $<TARGET_FILE:vtkh_core>
                      $<TARGET_FILE:vtkh_utils>
                      $<TARGET_FILE:ascent_lodepng>)
        endif()

        if(ENABLE_DRAY)
            list(APPEND internal_targets_to_link
                        $<TARGET_FILE:dray>)
        endif()

        set(imported_targets_to_link "")

        # updated for vtkm 1.9
        list(APPEND imported_targets_to_link
                    vtkmdiympi_nompi
                    vtkm_io
                    vtkm_rendering
                    vtkm_filter_clean_grid
                    vtkm_filter_contour
                    vtkm_filter_core
                    vtkm_filter_connected_components
                    vtkm_filter_density_estimate
                    vtkm_filter_entity_extraction
                    vtkm_filter_field_conversion
                    vtkm_filter_field_transform
                    vtkm_filter_flow
                    vtkm_filter_geometry_refinement
                    vtkm_filter_image_processing
                    vtkm_filter_mesh_info
                    vtkm_filter_multi_block
                    vtkm_filter_resampling
                    vtkm_filter_scalar_topology
                    vtkm_filter_vector_analysis
                    vtkm_filter_zfp
                    vtkm_worklet
                    vtkm_source
                    vtkm_cont
                    )

        if(TARGET vtkmdiympi_nompi)
            list(APPEND imported_targets_to_link vtkmdiympi_nompi)
        endif()

        if(TARGET vtkmdiympi)
            list(APPEND imported_targets_to_link vtkmdiympi)
        endif()

        if(RAJA_FOUND)
            list(APPEND imported_targets_to_link RAJA)
        endif()

        if(UMPIRE_FOUND)
            list(APPEND imported_targets_to_link umpire)
        endif()
        
        if(CAMP_FOUND)
            list(APPEND imported_targets_to_link camp)
        endif()

        set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ascent.dir")
        # clear output result
        set(locations_on_disk "")

        # device link
        source_objects(BINARY_DIR ${binary_dir}
                       SOURCES  ${ascent_device_sources}
                       TARGETS ${internal_targets_to_link}
                       IMPORTED_TARGETS ${imported_targets_to_link}
                       RESULT locations_on_disk)

        separate_arguments(params NATIVE_COMMAND "${CMAKE_CUDA_FLAGS} ${locations_on_disk}")
        message(STATUS "${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm.o")

        add_custom_command(TARGET ascent PRE_LINK
                           DEPENDS ascent rover vtkh
                           COMMAND ${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm.o
                           COMMENT "manual device link step for Ascent"
                           )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bcal_vtkm.o DESTINATION lib)
    endif()

endif()

################################################
# Build Parallel (MPI) version of ascent
################################################
if(MPI_FOUND)

    # req'd libs
    set(ascent_mpi_thirdparty_libs ${ascent_blt_mpi_deps} conduit::conduit_mpi)

    if (BABELFLOW_FOUND)
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
      list(APPEND ascent_mpi_thirdparty_libs StreamStat::StreamingStatistics)
      list(APPEND ascent_mpi_thirdparty_libs TopoFileParser::TopologyFileParser)

      include_directories(${BABELFLOW_INCLUDE_DIRS})
      include_directories(${PMT_INCLUDE_DIRS})
      include_directories(${StreamStat_INCLUDE_DIRS})
      include_directories(${TopoParser_INCLUDE_DIRS})
    endif ()

    # hola mpi support is only in par (mpi ver)
    list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_hola_filters.hpp)
    list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_hola_filters.cpp)

    list(APPEND ascent_headers hola/ascent_hola_mpi.hpp)
    list(APPEND ascent_sources hola/ascent_hola_mpi.cpp)

    # adios support is only in par (mpi ver)
    if (FIDES_FOUND)
        list(APPEND ascent_headers runtimes/flow_filters/ascent_runtime_adios2_filters.hpp)
        list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_adios2_filters.cpp)
        list(APPEND ascent_mpi_thirdparty_libs adios2::adios2 fides)
    endif()

    # this combines our mpi specific deps to the broader list of ascent deps
    list(APPEND ascent_mpi_thirdparty_libs ${ascent_thirdparty_libs})

    if(VTKM_FOUND)
        # swap vtkh_mpi for vtkh
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs vtkh)
        list(APPEND ascent_mpi_thirdparty_libs vtkh_mpi)
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs rover)
        list(APPEND ascent_mpi_thirdparty_libs rover_mpi)

        # babelflow specific vtk-m sources
        if (BABELFLOW_FOUND)
            list(APPEND ascent_sources runtimes/flow_filters/ascent_runtime_babelflow_iso.cpp)
        endif()

        list(APPEND ascent_vtkm_device_sources runtimes/flow_filters/ascent_runtime_babelflow_iso.cpp)
    endif()

    if(ENABLE_DRAY)
        # swap dray_mpi for dray
        list(REMOVE_ITEM ascent_mpi_thirdparty_libs dray)
        list(APPEND ascent_mpi_thirdparty_libs dray_mpi)
    endif()

    blt_add_library(NAME        ascent_mpi
                    SOURCES     ${ascent_sources} ${ascent_mpi_sources}
                    HEADERS     ${ascent_headers} ${ascent_mpi_headers}
                    DEPENDS_ON  ${ascent_mpi_thirdparty_libs}
                    )

    #  compile defs
    target_compile_definitions(ascent_mpi PRIVATE ASCENT_EXPORTS_FLAG)
    target_compile_definitions(ascent_mpi PRIVATE ASCENT_MPI_ENABLED)

    # symbol visibility
    if(ENABLE_HIDDEN_VISIBILITY)
        set_target_properties(ascent_mpi PROPERTIES CXX_VISIBILITY_PRESET hidden)
    endif()

    # special device source logic for vtk-m features
    if(VTKM_FOUND)
        vtkm_add_target_information(ascent_mpi DEVICE_SOURCES ${ascent_vtkm_device_sources})
    endif()

    # build includes
    # for generated config header
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)
    # this allows us to include as <header.hpp> during builds
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/>)
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/c>)
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/utils>)
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/runtimes>)
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/hola>)
    target_include_directories(ascent_mpi PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/libs/ascent/runtimes/flow_filters>)
    # install includes
    # setup exported target to point to proper include dir
    target_include_directories(ascent_mpi PUBLIC $<INSTALL_INTERFACE:include/ascent/>)

    # install target for mpi ver
    install(TARGETS ascent_mpi
            EXPORT  ascent
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            RUNTIME DESTINATION lib
            )

    # manual device link for CUDA
    if(VTKM_FOUND AND CUDA_FOUND AND NOT BUILD_SHARED_LIBS)
        set(internal_targets_to_link $<TARGET_FILE:rover>)

        if(ENABLE_VTKH)
            list(APPEND internal_targets_to_link
                       $<TARGET_FILE:vtkh_rendering_mpi>
                       $<TARGET_FILE:vtkh_compositing_mpi>
                       $<TARGET_FILE:vtkh_filters_mpi>
                       $<TARGET_FILE:vtkh_core_mpi>
                       $<TARGET_FILE:vtkh_utils_mpi>)
        endif()

        if(ENABLE_DRAY)
            list(APPEND internal_targets_to_link
                        $<TARGET_FILE:dray_mpi>)
        endif()

        set(imported_targets_to_link "")

        # updated for vtkm 1.9
        list(APPEND imported_targets_to_link
                    vtkmdiympi_nompi
                    vtkm_io
                    vtkm_rendering
                    vtkm_filter_clean_grid
                    vtkm_filter_contour
                    vtkm_filter_core
                    vtkm_filter_connected_components
                    vtkm_filter_density_estimate
                    vtkm_filter_entity_extraction
                    vtkm_filter_field_conversion
                    vtkm_filter_field_transform
                    vtkm_filter_flow
                    vtkm_filter_geometry_refinement
                    vtkm_filter_image_processing
                    vtkm_filter_mesh_info
                    vtkm_filter_multi_block
                    vtkm_filter_resampling
                    vtkm_filter_scalar_topology
                    vtkm_filter_vector_analysis
                    vtkm_filter_zfp
                    vtkm_worklet
                    vtkm_source
                    vtkm_cont
                    )

        if(TARGET vtkmdiympi_nompi)
            list(APPEND imported_targets_to_link vtkmdiympi_nompi)
        endif()

        if(TARGET vtkmdiympi)
            list(APPEND imported_targets_to_link vtkmdiympi)
        endif()

        if(RAJA_FOUND)
            list(APPEND imported_targets_to_link RAJA)
        endif()

        if(UMPIRE_FOUND)
            list(APPEND imported_targets_to_link umpire)
        endif()

        if(CAMP_FOUND)
            list(APPEND imported_targets_to_link camp)
        endif()

        set(binary_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/ascent_mpi.dir")
        # clear result
        set(locations_on_disk "")
        
        # device link
        source_objects(BINARY_DIR ${binary_dir}
                       SOURCES    ${ascent_device_sources}
                       TARGETS    ${internal_targets_to_link}
                       IMPORTED_TARGETS ${imported_targets_to_link}
                       RESULT locations_on_disk)

        separate_arguments(params NATIVE_COMMAND "${CMAKE_CUDA_FLAGS} ${locations_on_disk}")
        message(STATUS "${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm_mpi.o")
        add_custom_command(TARGET ascent_mpi PRE_LINK
                            DEPENDS ascent_mpi rover_mpi
                            COMMAND ${CMAKE_CUDA_COMPILER} --device-link ${params} -lcudadevrt -lcudart_static --output-file bcal_vtkm_mpi.o
                            COMMENT "manual device link step for Ascent parallel"
                            )
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/bcal_vtkm_mpi.o DESTINATION lib)
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
