#==============================================================================
# See License.txt
#==============================================================================

set(vtkm_filters_headers
  vtkmCellAverage.hpp
  vtkmClip.hpp
  vtkmCleanGrid.hpp
  vtkmClipWithField.hpp
  vtkmExtractStructured.hpp
  vtkmGradient.hpp
  vtkmLagrangian.hpp
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
  vtkmExtractStructured.cpp
  vtkmGradient.cpp
  vtkmLagrangian.cpp
  vtkmMarchingCubes.cpp
  vtkmMeshQuality.cpp
  vtkmPointAverage.cpp
  vtkmPointTransform.cpp
  vtkmTetrahedralize.cpp
  vtkmTriangulate.cpp
  vtkmThreshold.cpp
  vtkmVectorMagnitude.cpp
  )

set(vtkm_filters_install_headers
    GradientParameters.hpp)

# we use these specific vtk-m targets b/c our favorite version
# of cmake does not yet support transitive deps for object libs
set(vtkm_filters_deps vtkm_compiler_flags vtkm_diy )

if(ENABLE_OPENMP)
  list(APPEND vtkm_filters_deps openmp)
endif()

if(ENABLE_KOKKOS)
   list(APPEND vtkh_filters_deps Kokkos::kokkos)
endif()

if(CUDA_FOUND)
    # triggers cuda compile
    list(APPEND vtkm_filters_deps cuda)
endif()

blt_add_library(
  NAME vtkm_compiled_filters
  SOURCES ${vtkm_filters_sources}
  HEADERS ${vtkm_filters_headers}
  DEPENDS_ON ${vtkm_filters_deps}
  OBJECT TRUE
  )

vtkm_add_target_information(vtkm_compiled_filters DEVICE_SOURCES ${vtkm_filter_sources})

if(ENABLE_KOKKOS)
  # vtkm + object libs are fun. Flags are not being propogated, so manually add them
  set_target_properties(vtkm_compiled_filters PROPERTIES COMPILE_FLAGS "-I${KOKKOS_DIR}/include" )
endif()

if(CUDA_FOUND)
  # vtkm + object libs are fun. Flags are not being propogated, so manually add them
  set_target_properties(vtkm_compiled_filters PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr" )
  # defend against vtkm code cuda can't reason about
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0.0)
    target_compile_definitions(vtkm_compiled_filters PRIVATE VTKH_BYPASS_VTKM_BIH)
  endif()
endif()

install(FILES ${vtkm_filters_install_headers}
  DESTINATION ${VTKh_INSTALL_INCLUDE_DIR}/vtkh/vtkm_filters)

