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

#include "MeshConnectivityContainers.hpp"

namespace vtkh
{
namespace rendering
{
namespace raytracing
{

MeshConnectivityContainer::MeshConnectivityContainer(){};
MeshConnectivityContainer::~MeshConnectivityContainer(){};

template <typename T>
VTKM_CONT void MeshConnectivityContainer::FindEntryImpl(vtkm::rendering::raytracing::Ray<T>& rays)
{
  bool getCellIndex = true;

  Intersector.SetUseWaterTight(true);

  Intersector.IntersectRays(rays, getCellIndex);
}

void MeshConnectivityContainer::FindEntry(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays)
{
  this->FindEntryImpl(rays);
}

void MeshConnectivityContainer::FindEntry(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays)
{
  this->FindEntryImpl(rays);
}

VTKM_CONT
MeshConnectivityContainerUnstructured::MeshConnectivityContainerUnstructured(
  const vtkm::cont::CellSetExplicit<>& cellset,
  const vtkm::cont::CoordinateSystem& coords,
  const IdHandle& faceConn,
  const IdHandle& faceOffsets,
  const Id4Handle& triangles)
  : FaceConnectivity(faceConn)
  , FaceOffsets(faceOffsets)
  , Cellset(cellset)
  , Coords(coords)
{
  this->Triangles = triangles;
  //
  // Grab the cell arrays
  //
  CellConn =
    Cellset.GetConnectivityArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());
  CellOffsets =
    Cellset.GetOffsetsArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());
  Shapes = Cellset.GetShapesArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());

  Intersector.SetData(Coords, Triangles);
}

MeshConnectivityContainerUnstructured::~MeshConnectivityContainerUnstructured(){};

vtkm::rendering::raytracing::MeshConnectivity MeshConnectivityContainerUnstructured::PrepareForExecution(
  vtkm::cont::DeviceAdapterId deviceId,
  vtkm::cont::Token& token) const
{
  return vtkm::rendering::raytracing::MeshConnectivity(this->FaceConnectivity,
                          this->FaceOffsets,
                          this->CellConn,
                          this->CellOffsets,
                          this->Shapes,
                          deviceId,
                          token);
}

VTKM_CONT
MeshConnectivityContainerSingleType::MeshConnectivityContainerSingleType(
  const vtkm::cont::CellSetSingleType<>& cellset,
  const vtkm::cont::CoordinateSystem& coords,
  const IdHandle& faceConn,
  const Id4Handle& triangles)
  : FaceConnectivity(faceConn)
  , Coords(coords)
  , Cellset(cellset)
{

  this->Triangles = triangles;

  this->Intersector.SetUseWaterTight(true);

  this->CellConnectivity =
    Cellset.GetConnectivityArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());
  vtkm::cont::ArrayHandleConstant<vtkm::UInt8> shapes =
    Cellset.GetShapesArray(vtkm::TopologyElementTagCell(), vtkm::TopologyElementTagPoint());

  this->ShapeId = shapes.ReadPortal().Get(0);
  vtkm::rendering::raytracing::CellTables tables;
  this->NumIndices = tables.FaceLookUp(tables.CellTypeLookUp(ShapeId), 2);

  if (this->NumIndices == 0)
  {
    std::stringstream message;
    message << "Unstructured Mesh Connecitity Single type Error: unsupported cell type: ";
    message << ShapeId;
    throw vtkm::cont::ErrorBadValue(message.str());
  }
  vtkm::Id start = 0;
  this->NumFaces = tables.FaceLookUp(tables.CellTypeLookUp(this->ShapeId), 1);
  vtkm::Id numCells = this->CellConnectivity.ReadPortal().GetNumberOfValues();
  this->CellOffsets =
    vtkm::cont::make_ArrayHandleCounting<vtkm::Id>(start, this->NumIndices, numCells);

  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("mesh_conn_construction");

  this->Intersector.SetData(Coords, Triangles);
}

vtkm::rendering::raytracing::MeshConnectivity MeshConnectivityContainerSingleType::PrepareForExecution(
  vtkm::cont::DeviceAdapterId deviceId,
  vtkm::cont::Token& token) const
{
  return vtkm::rendering::raytracing::MeshConnectivity(this->FaceConnectivity,
                          this->CellConnectivity,
                          this->CellOffsets,
                          this->ShapeId,
                          this->NumIndices,
                          this->NumFaces,
                          deviceId,
                          token);
}

MeshConnectivityContainerStructured::MeshConnectivityContainerStructured(
  const vtkm::cont::CellSetStructured<3>& cellset,
  const vtkm::cont::CoordinateSystem& coords,
  const Id4Handle& triangles)
  : Coords(coords)
  , Cellset(cellset)
{

  this->Triangles = triangles;
  this->Intersector.SetUseWaterTight(true);

  this->PointDims = this->Cellset.GetPointDimensions();
  this->CellDims = this->Cellset.GetCellDimensions();

  this->Intersector.SetData(Coords, Triangles);
}

vtkm::rendering::raytracing::MeshConnectivity MeshConnectivityContainerStructured::PrepareForExecution(
  vtkm::cont::DeviceAdapterId,
  vtkm::cont::Token&) const
{
  return vtkm::rendering::raytracing::MeshConnectivity(CellDims, PointDims);
}
}
}
} //namespace vtkm::rendering::raytracing
