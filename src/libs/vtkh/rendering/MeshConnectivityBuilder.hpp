//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

//
// This code was originally pulled in from Viskores.
//

#ifndef vtkh_rendering_raytracing_MeshConnectivityBuilder_h
#define vtkh_rendering_raytracing_MeshConnectivityBuilder_h

#include "MeshConnectivityContainers.hpp"

#include <vtkh/vtkh_exports.h>

#include <viskores/cont/Algorithm.h>
#include <viskores/cont/Timer.h>
#include <viskores/rendering/raytracing/MortonCodes.h>

namespace vtkh
{
namespace rendering
{
namespace raytracing
{

class VTKH_API MeshConnectivityBuilder
{
public:
  MeshConnectivityBuilder();
  ~MeshConnectivityBuilder();

  VISKORES_CONT
  MeshConnectivityContainer* BuildConnectivity(const viskores::cont::UnknownCellSet& cellset,
                                               const viskores::cont::CoordinateSystem& coordinates);

  VISKORES_CONT
  viskores::cont::ArrayHandle<viskores::Id4> ExternalTrianglesStructured(
    viskores::cont::CellSetStructured<3>& cellSetStructured);

  viskores::cont::ArrayHandle<viskores::Id> GetFaceConnectivity();

  viskores::cont::ArrayHandle<viskores::Id> GetFaceOffsets();

  viskores::cont::ArrayHandle<viskores::Id4> GetTriangles();

protected:
  VISKORES_CONT
  void BuildConnectivity(viskores::cont::CellSetSingleType<>& cellSetUnstructured,
                         const viskores::cont::CoordinateSystem::MultiplexerArrayType& coordinates,
                         viskores::Bounds coordsBounds);

  VISKORES_CONT
  void BuildConnectivity(viskores::cont::CellSetExplicit<>& cellSetUnstructured,
                         const viskores::cont::CoordinateSystem::MultiplexerArrayType& coordinates,
                         viskores::Bounds coordsBounds);

  viskores::cont::ArrayHandle<viskores::Id> FaceConnectivity;
  viskores::cont::ArrayHandle<viskores::Id> FaceOffsets;
  viskores::cont::ArrayHandle<viskores::Id4> Triangles;
};
}
}
} //namespace viskores::rendering::raytracing
#endif
