###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Cloverleaf3D CMake Build for Ascent
#
###############################################################################


set(clover_sources
    PdV.f90
    PdV_kernel.f90
    accelerate.f90
    accelerate_kernel.f90
    advec_cell_driver.f90
    advec_cell_kernel.f90
    advec_mom_driver.f90
    advec_mom_kernel.f90
    advection.f90
    build_field.f90
    calc_dt.f90
    calc_dt_kernel.f90
    clover.F90
    clover_leaf.f90
    data.f90
    definitions.f90
    field_summary.f90
    field_summary_kernel.f90
    flux_calc.f90
    flux_calc_kernel.f90
    generate_chunk.f90
    generate_chunk_kernel.f90
    hydro.f90
    ideal_gas.f90
    ideal_gas_kernel.f90
    initialise.f90
    initialise_chunk.f90
    initialise_chunk_kernel.f90
    pack_kernel.f90
    parse.f90
    read_input.f90
    report.f90
    reset_field.f90
    reset_field_kernel.f90
    revert.f90
    revert_kernel.f90
    start.f90
    timer.f90
    timestep.f90
    update_halo.f90
    update_halo_kernel.f90
    viscosity.f90
    viscosity_kernel.f90
    visit.F90
    timer_c.c
    clover_main.cpp)

# cloverleaf3d reqs fortran and mpi
if(MPI_FOUND AND FORTRAN_FOUND)
    # copy over the input deck
    configure_file(clover.in ${CMAKE_CURRENT_BINARY_DIR}/clover.in COPYONLY)
    configure_file(ascent_actions.yaml ${CMAKE_CURRENT_BINARY_DIR}/ascent_actions.yaml COPYONLY)
    configure_file(ascent_options.yaml ${CMAKE_CURRENT_BINARY_DIR}/ascent_options.yaml COPYONLY)

    if(MPI_Fortran_USE_MODULE )
        set(clover_compile_flags "-DUSE_MOD")
    else()
        set(clover_compile_flags "-DUSE_MPIF")
    endif()

    set(clover_par_deps ascent_mpi mpi)
    if(OPENMP_FOUND)
      list(APPEND clover_par_deps openmp)
    endif()

    blt_add_executable(
        NAME        cloverleaf3d_par
        SOURCES     ${clover_sources}
        DEPENDS_ON  ${clover_par_deps}
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

    # the intel compiler will auto create a fortran main, which will
    # conflict with the cpp main, use -nofor_main to avoid this
    blt_append_custom_compiler_flag(
            FLAGS_VAR clover_link_flags
            INTEL   "-nofor_main"
        )

    # these flags are needeed for gfortran 10 to avoid
    # compile errors with the mpi interface
    if(CMAKE_Fortran_COMPILER_VERSION GREATER  "9")
        blt_append_custom_compiler_flag(
            FLAGS_VAR clover_compile_flags
            GNU   "-fallow-invalid-boz -fallow-argument-mismatch"
        )
    endif()

    blt_add_target_compile_flags(TO cloverleaf3d_par FLAGS "${clover_compile_flags}")
    blt_add_target_link_flags(TO cloverleaf3d_par FLAGS "${clover_link_flags}")


    # install target for clover mpi
    install(TARGETS cloverleaf3d_par
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/proxies/cloverleaf3d
            ARCHIVE DESTINATION examples/ascent/proxies/cloverleaf3d
            RUNTIME DESTINATION examples/ascent/proxies/cloverleaf3d
    )

    install(FILES ASCENT_README.md
                  clover.in
                  ascent_actions.json
                  ascent_actions_contour_tree_energy.json
                  ascent_actions_contour_tree_velocity.json
                  ascent_options.json
            DESTINATION examples/ascent/proxies/cloverleaf3d)

endif()
