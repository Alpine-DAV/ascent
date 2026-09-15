#==============================================================================
# See License.txt
#==============================================================================

set(viskores_filters_headers
    viskoresCellAverage.hpp
    viskoresClip.hpp
    viskoresCleanGrid.hpp
    viskoresClipWithField.hpp
    viskoresCompositeVector.hpp
    viskoresExtractStructured.hpp
    viskoresGhostCellRemove.hpp
    viskoresGradient.hpp
    viskoresHistogram.hpp
    viskoresLagrangian.hpp
    viskoresLinearExtrude.hpp
    viskoresLog.hpp
    viskoresMarchingCubes.hpp
    viskoresMeshQuality.hpp
    viskoresPointAverage.hpp
    viskoresPointTransform.hpp
    viskoresProbe.hpp
    viskoresStatistics.hpp
    viskoresTetrahedralize.hpp
    viskoresTriangulate.hpp
    viskoresThreshold.hpp
    viskoresVectorMagnitude.hpp
    viskoresRevolve.hpp
    GradientParameters.hpp
    )

set(viskores_filters_sources
    viskoresCellAverage.cpp
    viskoresClip.cpp
    viskoresCleanGrid.cpp
    viskoresClipWithField.cpp
    viskoresCompositeVector.cpp
    viskoresExtractStructured.cpp
    viskoresGhostCellRemove.cpp
    viskoresGradient.cpp
    viskoresHistogram.cpp
    viskoresLagrangian.cpp
    viskoresLinearExtrude.cpp
    viskoresLog.cpp
    viskoresMarchingCubes.cpp
    viskoresMeshQuality.cpp
    viskoresPointAverage.cpp
    viskoresPointTransform.cpp
    viskoresProbe.cpp
    viskoresStatistics.cpp
    viskoresTetrahedralize.cpp
    viskoresTriangulate.cpp
    viskoresThreshold.cpp
    viskoresVectorMagnitude.cpp
    viskoresRevolve.cpp
    )

# these are the only headers we need to install
set(viskores_filters_install_headers
    GradientParameters.hpp)

# we use these specific viskores targets b/c our favorite version
# of cmake does not yet support transitive deps for object libs
set(viskores_filters_deps viskores::compiler_flags viskores::diy ${vtkh_base_deps})


blt_add_library(NAME viskores_compiled_filters
                SOURCES ${viskores_filters_sources}
                HEADERS ${viskores_filters_headers}
                DEPENDS_ON ${viskores_filters_deps}
                OBJECT TRUE
                )

viskores_add_target_information(viskores_compiled_filters DEVICE_SOURCES ${viskores_filters_sources})

if(ENABLE_HIDDEN_VISIBILITY)
  set_target_properties(viskores_compiled_filters PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

if(KOKKOS_FOUND)
    # viskores + object libs are fun. Flags are not being propagated, so manually add them
    set_target_properties(viskores_compiled_filters PROPERTIES COMPILE_FLAGS "-I${KOKKOS_DIR}/include" )
endif()

if(CUDA_FOUND)
    # viskores + object libs are fun. Flags are not being propagated, so manually add them
    set_target_properties(viskores_compiled_filters PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr" )
    # defend against viskores code cuda can't reason about
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
        target_compile_definitions(viskores_compiled_filters PRIVATE VTKH_BYPASS_VISKORES_BIH)
    endif()
endif()

install(FILES ${viskores_filters_install_headers}
        DESTINATION include/ascent/vtkh/viskores_filters)
