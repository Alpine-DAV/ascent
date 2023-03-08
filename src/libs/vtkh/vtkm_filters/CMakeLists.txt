#==============================================================================
# See License.txt
#==============================================================================

set(vtkm_filters_headers
    vtkmCellAverage.hpp
    vtkmClip.hpp
    vtkmCleanGrid.hpp
    vtkmClipWithField.hpp
    vtkmCompositeVector.hpp
    vtkmExtractStructured.hpp
    vtkmGradient.hpp
    vtkmHistogram.hpp
    vtkmLagrangian.hpp
    vtkmLog.hpp
    vtkmMarchingCubes.hpp
    vtkmMeshQuality.hpp
    vtkmPointAverage.hpp
    vtkmPointTransform.hpp
    vtkmTetrahedralize.hpp
    vtkmTriangulate.hpp
    vtkmThreshold.hpp
    vtkmVectorMagnitude.hpp
    GradientParameters.hpp
    )

set(vtkm_filters_sources
    vtkmCellAverage.cpp
    vtkmClip.cpp
    vtkmCleanGrid.cpp
    vtkmClipWithField.cpp
    vtkmCompositeVector.cpp
    vtkmExtractStructured.cpp
    vtkmGradient.cpp
    vtkmHistogram.cpp
    vtkmLagrangian.cpp
    vtkmLog.cpp
    vtkmMarchingCubes.cpp
    vtkmMeshQuality.cpp
    vtkmPointAverage.cpp
    vtkmPointTransform.cpp
    vtkmTetrahedralize.cpp
    vtkmTriangulate.cpp
    vtkmThreshold.cpp
    vtkmVectorMagnitude.cpp
    )

# these are the only headers we need to install
set(vtkm_filters_install_headers
    GradientParameters.hpp)

# we use these specific vtk-m targets b/c our favorite version
# of cmake does not yet support transitive deps for object libs
set(vtkm_filters_deps vtkm::compiler_flags vtkm::diy ${vtkh_base_deps})


blt_add_library(NAME vtkm_compiled_filters
                SOURCES ${vtkm_filters_sources}
                HEADERS ${vtkm_filters_headers}
                DEPENDS_ON ${vtkm_filters_deps}
                OBJECT TRUE
                )

vtkm_add_target_information(vtkm_compiled_filters DEVICE_SOURCES ${vtkm_filter_sources})

if(ENABLE_HIDDEN_VISIBILITY)
  set_target_properties(vtkm_compiled_filters PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

if(KOKKOS_FOUND)
    # vtkm + object libs are fun. Flags are not being propagated, so manually add them
    set_target_properties(vtkm_compiled_filters PROPERTIES COMPILE_FLAGS "-I${KOKKOS_DIR}/include" )
endif()

if(CUDA_FOUND)
    # vtkm + object libs are fun. Flags are not being propagated, so manually add them
    set_target_properties(vtkm_compiled_filters PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr" )
    # defend against vtkm code cuda can't reason about
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
        target_compile_definitions(vtkm_compiled_filters PRIVATE VTKH_BYPASS_VTKM_BIH)
    endif()
endif()

install(FILES ${vtkm_filters_install_headers}
        DESTINATION include/ascent/vtkh/vtkm_filters)

