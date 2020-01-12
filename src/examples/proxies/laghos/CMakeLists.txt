project( laghos )

set(laghos_sources serial/laghos_assembly_s.cpp
                   serial/laghos_s.cpp
                   serial/laghos_solver_s.cpp)

set(laghos_headers serial/laghos_assembly_s.hpp
                   serial/laghos_solver_s.hpp)


set(laghos_sources_mpi laghos_assembly.cpp
                       laghos.cpp
                       laghos_solver.cpp)

set(laghos_headers_mpi laghos_assembly.hpp
                       laghos_solver.hpp)

set(laghos_thirdparty_deps mfem)

# copy over the data/files
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# copy example laucnh scripts
file(COPY run_taylor_green_2d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY run_taylor_green_3d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY run_tripple_point_2d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY run_tripple_point_3d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY run_sedov_2d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY run_sedov_3d.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# copy example laucnh scripts
file(COPY ascent_options.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# mfem does not build with both serial and mpi
if(MPI_FOUND AND ENABLE_MPI)
  list(APPEND laghos_thirdparty_deps mpi)
endif()

if(PYTHON_FOUND)
    list(APPEND laghos_thirdparty_deps python)
endif()


if (ENABLE_SERIAL)
    blt_add_executable(
        NAME        laghos_ser
        HEADERS     ${laghos_headers}
        SOURCES     ${laghos_sources}
        DEPENDS_ON  ${laghos_thirdparty_deps} ascent
        OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR})
    
    
    # install target for laghos serial
    install(TARGETS laghos_ser
            EXPORT  ascent
            LIBRARY DESTINATION examples/ascent/proxies/laghos
            ARCHIVE DESTINATION examples/ascent/proxies/laghos
            RUNTIME DESTINATION examples/ascent/proxies/laghos
    )
endif()

if(MPI_FOUND AND ENABLE_MPI)

  blt_add_executable(
      NAME        laghos_mpi
      HEADERS     ${laghos_headers_mpi}
      SOURCES     ${laghos_sources_mpi}
      DEPENDS_ON  ${laghos_thirdparty_deps} ascent_mpi
      OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR})

  # install target for laghos mpi
  install(TARGETS laghos_mpi
          EXPORT  ascent
          LIBRARY DESTINATION examples/ascent/proxies/laghos
          ARCHIVE DESTINATION examples/ascent/proxies/laghos
          RUNTIME DESTINATION examples/ascent/proxies/laghos
  )
endif()

install(DIRECTORY data
        DESTINATION examples/ascent/proxies/laghos)
