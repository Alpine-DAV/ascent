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

#include "ConnectivityTracer.hpp"
#include "MeshConnectivityBuilder.hpp"

#include <iomanip>

#ifndef CELL_SHAPE_ZOO
#define CELL_SHAPE_ZOO 255
#endif

#ifndef CELL_SHAPE_STRUCTURED
#define CELL_SHAPE_STRUCTURED 254
#endif

namespace vtkh
{
namespace rendering
{

// A more useful  bounds check that tells you where it happened
#ifndef NDEBUG
#define BOUNDS_CHECK(HANDLE, INDEX) BoundsCheck((HANDLE), (INDEX), __FILE__, __LINE__)
#else
#define BOUNDS_CHECK(HANDLE, INDEX)
#endif

template <typename ArrayHandleType>
VTKM_EXEC inline void BoundsCheck(const ArrayHandleType& handle,
                                  const vtkm::Id& index,
                                  const char* file,
                                  int line)
{
  if (index < 0 || index >= handle.GetNumberOfValues())
    printf("Bad Index %d  at file %s line %d\n", (int)index, file, line);
}

namespace raytracing
{
namespace detail
{

class AdjustSample : public vtkm::worklet::WorkletMapField
{
  vtkm::Float64 SampleDistance;

public:
  VTKM_CONT
  AdjustSample(const vtkm::Float64 sampleDistance)
    : SampleDistance(sampleDistance)
  {
  }
  using ControlSignature = void(FieldIn, FieldInOut);
  using ExecutionSignature = void(_1, _2);
  template <typename FloatType>
  VTKM_EXEC inline void operator()(const vtkm::UInt8& status, FloatType& currentDistance) const
  {
    if (status != RAY_ACTIVE)
      return;

    currentDistance += vtkm::FMod(currentDistance, (FloatType)SampleDistance);
  }
}; //class AdvanceRay

template <typename FloatType>
void RayTracking<FloatType>::Compact(vtkm::cont::ArrayHandle<FloatType>& compactedDistances,
                                     vtkm::cont::ArrayHandle<vtkm::UInt8>& masks)
{
  //
  // These distances are stored in the rays, and it has
  // already been compacted.
  //
  CurrentDistance = compactedDistances;

  vtkm::cont::ArrayHandleCast<vtkm::Id, vtkm::cont::ArrayHandle<vtkm::UInt8>> castedMasks(masks);

  bool distance1IsEnter = EnterDist == &Distance1;

  vtkm::cont::ArrayHandle<FloatType> compactedDistance1;
  vtkm::cont::Algorithm::CopyIf(Distance1, masks, compactedDistance1);
  Distance1 = compactedDistance1;

  vtkm::cont::ArrayHandle<FloatType> compactedDistance2;
  vtkm::cont::Algorithm::CopyIf(Distance2, masks, compactedDistance2);
  Distance2 = compactedDistance2;

  vtkm::cont::ArrayHandle<vtkm::Int32> compactedExitFace;
  vtkm::cont::Algorithm::CopyIf(ExitFace, masks, compactedExitFace);
  ExitFace = compactedExitFace;

  if (distance1IsEnter)
  {
    EnterDist = &Distance1;
    ExitDist = &Distance2;
  }
  else
  {
    EnterDist = &Distance2;
    ExitDist = &Distance1;
  }
}

template <typename FloatType>
void RayTracking<FloatType>::Init(const vtkm::Id size,
                                  vtkm::cont::ArrayHandle<FloatType>& distances)
{

  ExitFace.Allocate(size);
  Distance1.Allocate(size);
  Distance2.Allocate(size);

  CurrentDistance = distances;
  //
  // Set the initial Distances
  //
  vtkm::worklet::DispatcherMapField<vtkm::rendering::raytracing::CopyAndOffset<FloatType>> resetDistancesDispatcher(
    vtkm::rendering::raytracing::CopyAndOffset<FloatType>(0.0f));
  resetDistancesDispatcher.Invoke(distances, *EnterDist);

  //
  // Init the exit faces. This value is used to load the next cell
  // base on the cell and face it left
  //
  vtkm::cont::ArrayHandleConstant<vtkm::Int32> negOne(-1, size);
  vtkm::cont::Algorithm::Copy(negOne, ExitFace);

  vtkm::cont::ArrayHandleConstant<FloatType> negOnef(-1.f, size);
  vtkm::cont::Algorithm::Copy(negOnef, *ExitDist);
}

template <typename FloatType>
void RayTracking<FloatType>::Swap()
{
  vtkm::cont::ArrayHandle<FloatType>* tmpPtr;
  tmpPtr = EnterDist;
  EnterDist = ExitDist;
  ExitDist = tmpPtr;
}

} //namespace detail

void ConnectivityTracer::Init()
{
  //
  // Check to see if a sample distance was set
  //
  vtkm::Bounds coordsBounds = Coords.GetBounds();
  vtkm::Float64 maxLength = 0.;
  maxLength = vtkm::Max(maxLength, coordsBounds.X.Length());
  maxLength = vtkm::Max(maxLength, coordsBounds.Y.Length());
  maxLength = vtkm::Max(maxLength, coordsBounds.Z.Length());
  BumpDistance = maxLength * BumpEpsilon;

  if (SampleDistance <= 0)
  {
    BoundingBox[0] = vtkm::Float32(coordsBounds.X.Min);
    BoundingBox[1] = vtkm::Float32(coordsBounds.X.Max);
    BoundingBox[2] = vtkm::Float32(coordsBounds.Y.Min);
    BoundingBox[3] = vtkm::Float32(coordsBounds.Y.Max);
    BoundingBox[4] = vtkm::Float32(coordsBounds.Z.Min);
    BoundingBox[5] = vtkm::Float32(coordsBounds.Z.Max);

    BackgroundColor[0] = 1.f;
    BackgroundColor[1] = 1.f;
    BackgroundColor[2] = 1.f;
    BackgroundColor[3] = 1.f;
    const vtkm::Float32 defaultSampleRate = 200.f;
    // We need to set some default sample distance
    vtkm::Vec3f_32 extent;
    extent[0] = BoundingBox[1] - BoundingBox[0];
    extent[1] = BoundingBox[3] - BoundingBox[2];
    extent[2] = BoundingBox[5] - BoundingBox[4];
    SampleDistance = vtkm::Magnitude(extent) / defaultSampleRate;
  }
}

vtkm::Id ConnectivityTracer::GetNumberOfMeshCells() const
{
  return CellSet.GetNumberOfCells();
}

void ConnectivityTracer::SetColorMap(const vtkm::cont::ArrayHandle<vtkm::Vec4f_32>& colorMap)
{
  ColorMap = colorMap;
}

// TODO: Comment out volume functionality
void ConnectivityTracer::SetVolumeData(const vtkm::cont::Field& scalarField,
                                       const vtkm::Range& scalarBounds,
                                       const vtkm::cont::UnknownCellSet& cellSet,
                                       const vtkm::cont::CoordinateSystem& coords,
                                       const vtkm::cont::Field& ghostField)
{
  //TODO: Need a way to tell if we have been updated
  ScalarField = scalarField;
  GhostField = ghostField;
  ScalarBounds = scalarBounds;
  CellSet = cellSet;
  Coords = coords;
  MeshConnIsConstructed = false;

  const bool isSupportedField = ScalarField.IsCellField() || ScalarField.IsPointField();
  if (!isSupportedField)
  {
    throw vtkm::cont::ErrorBadValue("Field not accociated with cell set or points");
  }
  FieldAssocPoints = ScalarField.IsPointField();

  this->Integrator = Volume;

  if (MeshContainer == nullptr)
  {
    delete MeshContainer;
  }
  MeshConnectivityBuilder builder;
  MeshContainer = builder.BuildConnectivity(cellSet, coords);

  Locator.SetCellSet(this->CellSet);
  Locator.SetCoordinates(this->Coords);
  Locator.Update();
}

// Absorption-only case
void ConnectivityTracer::SetEnergyData(const vtkm::cont::Field& absorption,
                                       const vtkm::Int32 numEnergyGroups,
                                       const vtkm::cont::UnknownCellSet& cellSet,
                                       const vtkm::cont::CoordinateSystem& coords)
{
  bool isSupportedField = absorption.GetAssociation() == vtkm::cont::Field::Association::Cells;
  if (!isSupportedField)
  {
    throw vtkm::cont::ErrorBadValue("Absorption Field '" + absorption.GetName() +
                                    "' not associated with cells");
  }

  ScalarField = absorption;
  CellSet = cellSet;
  Coords = coords;
  MeshConnIsConstructed = false;
  HasEmission = false;
  NumEnergyGroups = numEnergyGroups;

  // Do some basic range checking
  if (NumEnergyGroups < 1)
    throw vtkm::cont::ErrorBadValue("Number of energy bins is less than 1");
  vtkm::Id binCount = ScalarField.GetNumberOfValues();
  vtkm::Id cellCount = this->GetNumberOfMeshCells();

  //TODO: Need a way to tell if we have been updated
  this->Integrator = Energy;

  if (MeshContainer == nullptr)
  {
    delete MeshContainer;
  }

  MeshConnectivityBuilder builder;
  MeshContainer = builder.BuildConnectivity(cellSet, coords);
  Locator.SetCellSet(this->CellSet);
  Locator.SetCoordinates(this->Coords);
  Locator.Update();
}

// Absorption + Emission case
void ConnectivityTracer::SetEnergyData(const vtkm::cont::Field& absorption,
                                       const vtkm::Int32 numEnergyGroups,
                                       const vtkm::cont::UnknownCellSet& cellSet,
                                       const vtkm::cont::CoordinateSystem& coords,
                                       const vtkm::cont::Field& emission)
{
  bool isSupportedField = absorption.GetAssociation() == vtkm::cont::Field::Association::Cells;
  if (!isSupportedField)
    throw vtkm::cont::ErrorBadValue("Absorption Field '" + absorption.GetName() +
                                    "' not associated with cells");
  ScalarField = absorption;
  CellSet = cellSet;
  Coords = coords;
  MeshConnIsConstructed = false;
  NumEnergyGroups = numEnergyGroups;
  // Check for emission
  HasEmission = false;

  if (emission.GetAssociation() != vtkm::cont::Field::Association::Any)
  {
    if (emission.GetAssociation() != vtkm::cont::Field::Association::Cells)
      throw vtkm::cont::ErrorBadValue("Emission Field '" + emission.GetName() +
                                      "' not associated with cells");
    HasEmission = true;
    EmissionField = emission;
  }
  // Do some basic range checking
  if (NumEnergyGroups < 1)
    throw vtkm::cont::ErrorBadValue("Number of energy bins is less than 1");
  vtkm::Id binCount = ScalarField.GetNumberOfValues();
  vtkm::Id cellCount = this->GetNumberOfMeshCells();
  if (HasEmission)
  {
    binCount = EmissionField.GetNumberOfValues();
  }
  //TODO: Need a way to tell if we have been updated
  this->Integrator = Energy;

  if (MeshContainer == nullptr)
  {
    delete MeshContainer;
  }

  MeshConnectivityBuilder builder;
  MeshContainer = builder.BuildConnectivity(cellSet, coords);
  Locator.SetCellSet(this->CellSet);
  Locator.SetCoordinates(this->Coords);
  Locator.Update();
}

void ConnectivityTracer::SetBackgroundColor(const vtkm::Vec4f_32& backgroundColor)
{
  BackgroundColor = backgroundColor;
}

void ConnectivityTracer::SetSampleDistance(const vtkm::Float32& distance)
{
  if (distance <= 0.f)
    throw vtkm::cont::ErrorBadValue("Sample distance must be positive.");
  SampleDistance = distance;
}

void ConnectivityTracer::ResetTimers()
{
  IntersectTime = 0.;
  IntegrateTime = 0.;
  SampleTime = 0.;
  LostRayTime = 0.;
  MeshEntryTime = 0.;
}

void ConnectivityTracer::LogTimers()
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->AddLogData("intersect ", IntersectTime);
  logger->AddLogData("integrate ", IntegrateTime);
  logger->AddLogData("sample_cells ", SampleTime);
  logger->AddLogData("lost_rays ", LostRayTime);
  logger->AddLogData("mesh_entry", LostRayTime);
}

template <typename FloatType>
void ConnectivityTracer::PrintRayStatus(vtkm::rendering::raytracing::Ray<FloatType>& rays)
{
  vtkm::Id raysExited = vtkm::rendering::raytracing::RayOperations::GetStatusCount(rays, RAY_EXITED_MESH);
  vtkm::Id raysActive = vtkm::rendering::raytracing::RayOperations::GetStatusCount(rays, RAY_ACTIVE);
  vtkm::Id raysAbandoned = vtkm::rendering::raytracing::RayOperations::GetStatusCount(rays, RAY_ABANDONED);
  vtkm::Id raysExitedDom = vtkm::rendering::raytracing::RayOperations::GetStatusCount(rays, RAY_EXITED_DOMAIN);
  std::cout << "\r Ray Status " << std::setw(10) << std::left << " Lost " << std::setw(10)
            << std::left << RaysLost << std::setw(10) << std::left << " Exited " << std::setw(10)
            << std::left << raysExited << std::setw(10) << std::left << " Active " << std::setw(10)
            << raysActive << std::setw(10) << std::left << " Abandoned " << std::setw(10)
            << raysAbandoned << " Exited Domain " << std::setw(10) << std::left << raysExitedDom
            << "\n";
}

//
//  Advance Ray
//      After a ray leaves the mesh, we need to check to see
//      of the ray re-enters the mesh within this domain. This
//      function moves the ray forward some offset to prevent
//      "shadowing" and hitting the same exit point.
//
template <typename FloatType>
class AdvanceRay : public vtkm::worklet::WorkletMapField
{
  FloatType Offset;

public:
  VTKM_CONT
  AdvanceRay(const FloatType offset = 0.00001)
    : Offset(offset)
  {
  }
  using ControlSignature = void(FieldIn, FieldInOut);
  using ExecutionSignature = void(_1, _2);

  VTKM_EXEC inline void operator()(const vtkm::UInt8& status, FloatType& distance) const
  {
    if (status == RAY_EXITED_MESH)
      distance += Offset;
  }
}; //class AdvanceRay

class LocateCell : public vtkm::worklet::WorkletMapField
{
private:
vtkm::rendering::raytracing::CellIntersector<255> Intersector;

public:
  LocateCell() {}

  using ControlSignature = void(FieldInOut,
                                WholeArrayIn,
                                FieldIn,
                                FieldInOut,
                                FieldInOut,
                                FieldInOut,
                                FieldInOut,
                                FieldIn,
                                ExecObject meshConnectivity);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9);

  template <typename FloatType, typename PointPortalType>
  VTKM_EXEC inline void operator()(vtkm::Id& currentCell,
                                   PointPortalType& vertices,
                                   const vtkm::Vec<FloatType, 3>& dir,
                                   FloatType& enterDistance,
                                   FloatType& exitDistance,
                                   vtkm::Int32& enterFace,
                                   vtkm::UInt8& rayStatus,
                                   const vtkm::Vec<FloatType, 3>& origin,
                                   const vtkm::rendering::raytracing::MeshConnectivity& meshConn) const
  {
    if (enterFace != -1 && rayStatus == RAY_ACTIVE)
    {
      currentCell = meshConn.GetConnectingCell(currentCell, enterFace);
      if (currentCell == -1)
        rayStatus = RAY_EXITED_MESH;
      enterFace = -1;
    }
    //This ray is dead or exited the mesh and needs re-entry
    if (rayStatus != RAY_ACTIVE)
    {
      return;
    }
    FloatType xpoints[8];
    FloatType ypoints[8];
    FloatType zpoints[8];
    vtkm::Id cellConn[8];
    FloatType distances[6];

    const vtkm::Int32 numIndices = meshConn.GetCellIndices(cellConn, currentCell);
    //load local cell data
    for (int i = 0; i < numIndices; ++i)
    {
      BOUNDS_CHECK(vertices, cellConn[i]);
      vtkm::Vec<FloatType, 3> point = vtkm::Vec<FloatType, 3>(vertices.Get(cellConn[i]));
      xpoints[i] = point[0];
      ypoints[i] = point[1];
      zpoints[i] = point[2];
    }
    const vtkm::UInt8 cellShape = meshConn.GetCellShape(currentCell);
    Intersector.IntersectCell(xpoints, ypoints, zpoints, dir, origin, distances, cellShape);

    vtkm::rendering::raytracing::CellTables tables;
    const vtkm::Int32 numFaces = tables.FaceLookUp(tables.CellTypeLookUp(cellShape), 1);
    //vtkm::Int32 minFace = 6;
    vtkm::Int32 maxFace = -1;

    FloatType minDistance = static_cast<FloatType>(1e32);
    FloatType maxDistance = static_cast<FloatType>(-1);
    for (vtkm::Int32 i = 0; i < numFaces; ++i)
    {
      FloatType dist = distances[i];

      if (dist != -1)
      {
        if (dist < minDistance)
        {
          minDistance = dist;
          //minFace = i;
        }
        if (dist > maxDistance)
        {
          maxDistance = dist;
          maxFace = i;
        }
      }
    }

    if (maxDistance <= enterDistance || minDistance == maxDistance)
    {
      rayStatus = RAY_LOST;
    }
    else
    {
      enterDistance = minDistance;
      exitDistance = maxDistance;
      enterFace = maxFace;
    }

  } //operator
};  //class LocateCell

class RayBumper : public vtkm::worklet::WorkletMapField
{
private:
  vtkm::rendering::raytracing::CellIntersector<255> Intersector;
  vtkm::Float64 BumpDistance;

public:
  RayBumper(vtkm::Float64 bumpDistance)
    : BumpDistance(bumpDistance)
  {
  }


  using ControlSignature = void(FieldInOut,
                                WholeArrayIn,
                                FieldInOut,
                                FieldInOut,
                                FieldInOut,
                                FieldInOut,
                                FieldIn,
                                FieldInOut,
                                ExecObject meshConnectivity,
                                ExecObject locator);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10);

  template <typename FloatType, typename PointPortalType, typename LocatorType>
  VTKM_EXEC inline void operator()(vtkm::Id& currentCell,
                                   PointPortalType& vertices,
                                   FloatType& enterDistance,
                                   FloatType& exitDistance,
                                   vtkm::Int32& enterFace,
                                   vtkm::UInt8& rayStatus,
                                   const vtkm::Vec<FloatType, 3>& origin,
                                   vtkm::Vec<FloatType, 3>& rdir,
                                   const vtkm::rendering::raytracing::MeshConnectivity& meshConn,
                                   const LocatorType& locator) const
  {
    // We only process lost rays
    if (rayStatus != RAY_LOST)
    {
      return;
    }
    const FloatType bumpDistance = static_cast<FloatType>(BumpDistance);
    FloatType query_distance = enterDistance + bumpDistance;

    bool valid_cell = false;

    vtkm::Id cellId = currentCell;

    while (!valid_cell)
    {
      // push forward and look for a new cell
      while (cellId == currentCell)
      {
        query_distance += bumpDistance;
        vtkm::Vec<FloatType, 3> location = origin + rdir * (query_distance);
        vtkm::Vec<vtkm::FloatDefault, 3> pcoords;
        locator.FindCell(location, cellId, pcoords);
      }

      currentCell = cellId;
      if (currentCell == -1)
      {
        rayStatus = RAY_EXITED_MESH;
        return;
      }

      FloatType xpoints[8];
      FloatType ypoints[8];
      FloatType zpoints[8];
      vtkm::Id cellConn[8];
      FloatType distances[6];

      const vtkm::Int32 numIndices = meshConn.GetCellIndices(cellConn, currentCell);
      //load local cell data
      for (int i = 0; i < numIndices; ++i)
      {
        BOUNDS_CHECK(vertices, cellConn[i]);
        vtkm::Vec<FloatType, 3> point = vtkm::Vec<FloatType, 3>(vertices.Get(cellConn[i]));
        xpoints[i] = point[0];
        ypoints[i] = point[1];
        zpoints[i] = point[2];
      }

      const vtkm::UInt8 cellShape = meshConn.GetCellShape(currentCell);
      Intersector.IntersectCell(xpoints, ypoints, zpoints, rdir, origin, distances, cellShape);

      vtkm::rendering::raytracing::CellTables tables;
      const vtkm::Int32 numFaces = tables.FaceLookUp(tables.CellTypeLookUp(cellShape), 1);

      //vtkm::Int32 minFace = 6;
      vtkm::Int32 maxFace = -1;
      FloatType minDistance = static_cast<FloatType>(1e32);
      FloatType maxDistance = static_cast<FloatType>(-1);
      for (int i = 0; i < numFaces; ++i)
      {
        FloatType dist = distances[i];

        if (dist != -1)
        {
          if (dist < minDistance)
          {
            minDistance = dist;
            //minFace = i;
          }
          if (dist >= maxDistance)
          {
            maxDistance = dist;
            maxFace = i;
          }
        }
      }

      if (minDistance < maxDistance && minDistance > exitDistance)
      {
        enterDistance = minDistance;
        exitDistance = maxDistance;
        enterFace = maxFace;
        rayStatus = RAY_ACTIVE; //re-activate ray
        valid_cell = true;
      }
    }

  } //operator
};  //class RayBumper

class AddPathLengths : public vtkm::worklet::WorkletMapField
{
public:
  VTKM_CONT
  AddPathLengths() {}

  using ControlSignature = void(FieldIn,     // ray status
                                FieldIn,     // cell enter distance
                                FieldIn,     // cell exit distance
                                FieldInOut); // ray absorption data

  using ExecutionSignature = void(_1, _2, _3, _4);

  template <typename FloatType>
  VTKM_EXEC inline void operator()(const vtkm::UInt8& rayStatus,
                                   const FloatType& enterDistance,
                                   const FloatType& exitDistance,
                                   FloatType& distance) const
  {
    if (rayStatus != RAY_ACTIVE)
    {
      return;
    }

    if (exitDistance <= enterDistance)
    {
      return;
    }

    FloatType segmentLength = exitDistance - enterDistance;
    distance += segmentLength;
  }
};

class Integrate : public vtkm::worklet::WorkletMapField
{
private:
  const vtkm::Int32 NumEnergyGroups;
  const vtkm::Float32 UnitScalar;

public:
  VTKM_CONT
  Integrate(const vtkm::Int32 numEnergyGroups, const vtkm::Float32 unitScalar)
    : NumEnergyGroups(numEnergyGroups)
    , UnitScalar(unitScalar)
  {
  }

  using ControlSignature = void(FieldIn,         // ray status
                                FieldIn,         // cell enter distance
                                FieldIn,         // cell exit distance
                                FieldInOut,      // current distance
                                WholeArrayIn,    // cell absorption data array
                                WholeArrayInOut, // optical depth data
                                FieldIn);        // current cell

  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, WorkIndex);

  template <typename FloatType, typename CellDataPortalType, typename RayDataPortalType>
  VTKM_EXEC inline void operator()(const vtkm::UInt8& rayStatus,
                                   const FloatType& enterDistance,
                                   const FloatType& exitDistance,
                                   FloatType& currentDistance,
                                   const CellDataPortalType& absorbtionData,
                                   RayDataPortalType& opticalDepthBins,
                                   const vtkm::Id& currentCell,
                                   const vtkm::Id& rayIndex) const
  {
    if (rayStatus != RAY_ACTIVE || exitDistance <= enterDistance)
    {
      return;
    }

    FloatType segmentLength = exitDistance - enterDistance;
    vtkm::Id rayOffset = NumEnergyGroups * rayIndex;

    // Get the cell value and use VecTraits to handle both scalar and vector fields
    using CellValueType = typename CellDataPortalType::ValueType;
    using VecTraits = vtkm::VecTraits<CellValueType>;
    
    BOUNDS_CHECK(absorbtionData, currentCell);
    CellValueType cellValue = absorbtionData.Get(currentCell);
    
    // Use VecTraits for uniform handling - dispatcher ensures we get the right array type
    for (vtkm::Int32 i = 0; i < NumEnergyGroups; i++)
    {
      FloatType absorb = static_cast<FloatType>(VecTraits::GetComponent(cellValue, i));
      absorb *= UnitScalar;

      const int rayOffsetI = rayOffset + i;
      BOUNDS_CHECK(opticalDepthBins, rayOffsetI);
      FloatType opticalDepth = static_cast<FloatType>(opticalDepthBins.Get(rayOffsetI));      
      opticalDepthBins.Set(rayOffsetI, opticalDepth + absorb * segmentLength);
    }
    
    currentDistance = exitDistance;
  }
};

class IntegrateEmission : public vtkm::worklet::WorkletMapField
{
private:
  const vtkm::Int32 NumEnergyGroups;
  const vtkm::Float32 UnitScalar;
  bool DivideEmisByAbsorb;

public:
  VTKM_CONT
  IntegrateEmission(const vtkm::Int32 numEnergyGroups,
                    const vtkm::Float32 unitScalar,
                    const bool divideEmisByAbsorb)
    : NumEnergyGroups(numEnergyGroups)
    , UnitScalar(unitScalar)
    , DivideEmisByAbsorb(divideEmisByAbsorb)
  {
  }

  using ControlSignature = void(FieldIn,         // ray status
                                FieldIn,         // cell enter distance
                                FieldIn,         // cell exit distance
                                FieldInOut,      // current distance
                                WholeArrayIn,    // cell absorption data array
                                WholeArrayIn,    // cell emission data array
                                WholeArrayInOut, // ray absorption data
                                WholeArrayInOut, // ray emission data
                                WholeArrayInOut, // optical depth data
                                FieldIn);        // current cell

  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, WorkIndex);

  template <typename FloatType,
            typename CellAbsPortalType,
            typename CellEmisPortalType,
            typename RayDataPortalType>
  VTKM_EXEC inline void operator()(const vtkm::UInt8& rayStatus,
                                   const FloatType& enterDistance,
                                   const FloatType& exitDistance,
                                   FloatType& currentDistance,
                                   const CellAbsPortalType& absorptionData,
                                   const CellEmisPortalType& emissionData,
                                   RayDataPortalType& absorptionBins,
                                   RayDataPortalType& emissionBins,
                                   RayDataPortalType& opticalDepthBins,
                                   const vtkm::Id& currentCell,
                                   const vtkm::Id& rayIndex) const
  {
    if (rayStatus != RAY_ACTIVE || exitDistance <= enterDistance)
    {
      return;
    }

    FloatType segmentLength = exitDistance - enterDistance;
    vtkm::Id rayOffset = NumEnergyGroups * rayIndex;

    // Get the cell values to determine if we have vector or scalar fields
    using AbsValueType = typename CellAbsPortalType::ValueType;
    using EmisValueType = typename CellEmisPortalType::ValueType;
    using AbsVecTraits = vtkm::VecTraits<AbsValueType>;
    using EmisVecTraits = vtkm::VecTraits<EmisValueType>;
    
    BOUNDS_CHECK(absorptionData, currentCell);
    BOUNDS_CHECK(emissionData, currentCell);
    AbsValueType absCellValue = absorptionData.Get(currentCell);
    EmisValueType emisCellValue = emissionData.Get(currentCell);
    
    // Check if these are vector fields or scalar fields
    vtkm::IdComponent absNumComponents = AbsVecTraits::GetNumberOfComponents(absCellValue);
    vtkm::IdComponent emisNumComponents = EmisVecTraits::GetNumberOfComponents(emisCellValue);
    
    //
    // Traditionally, we would only keep track of a single intensity value per ray
    // per bin and we would integrate from the beginning to end of the ray. In a
    // distributed memory setting, we would move cell data around so that the
    // entire ray could be traced, but in situ, moving that much cell data around
    // could blow memory. Here we are keeping track of two values. Total absorption
    // through this contiguous segment of the mesh, and the amount of emitted energy
    // that makes it out of this mesh segment. If this is really run on a single node,
    // we can get the final energy value by multiplying the background intensity by
    // the total absorption of the mesh segment and add in the amount of emitted
    // energy that escapes.
    //

    // NumEnergyGroups can potentially be a very large number, so the loops are duplicated
    // like this to avoid checking if DivideEmisByAbsorb == true each iteration.

    if (DivideEmisByAbsorb)
    {
      for (vtkm::Int32 i = 0; i < NumEnergyGroups; i++)
      {
        FloatType absorb = static_cast<FloatType>(AbsVecTraits::GetComponent(absCellValue, i));
        FloatType emission = static_cast<FloatType>(EmisVecTraits::GetComponent(emisCellValue, i));

        absorb *= UnitScalar;
        emission *= UnitScalar;
        FloatType tmp = vtkm::Exp(-absorb * segmentLength);

        const int rayOffsetI = rayOffset + i;
        BOUNDS_CHECK(absorptionBins, rayOffsetI);
        FloatType absorbIntensity = static_cast<FloatType>(absorptionBins.Get(rayOffsetI));
        BOUNDS_CHECK(emissionBins, rayOffsetI);
        FloatType emissionIntensity = static_cast<FloatType>(emissionBins.Get(rayOffsetI));
        BOUNDS_CHECK(opticalDepthBins, rayOffsetI);
        FloatType opticalDepth = static_cast<FloatType>(opticalDepthBins.Get(rayOffsetI));

        absorptionBins.Set(rayOffsetI, absorbIntensity * tmp);
        // The only difference with this loop vs the other is that we do (emission / absorb) here.
        emissionBins.Set(rayOffsetI, emissionIntensity * tmp + (emission / absorb) * (1.0f - tmp));
        opticalDepthBins.Set(rayOffsetI, opticalDepth + absorb * segmentLength);
      }
    }
    else // (!DivideEmisByAbsorb)
    {
      for (vtkm::Int32 i = 0; i < NumEnergyGroups; i++)
      {
        FloatType absorb = static_cast<FloatType>(AbsVecTraits::GetComponent(absCellValue, i));
        FloatType emission = static_cast<FloatType>(EmisVecTraits::GetComponent(emisCellValue, i));

        absorb *= UnitScalar;
        emission *= UnitScalar;
        FloatType tmp = vtkm::Exp(-absorb * segmentLength);

        const int rayOffsetI = rayOffset + i;
        BOUNDS_CHECK(absorptionBins, rayOffsetI);
        FloatType absorbIntensity = static_cast<FloatType>(absorptionBins.Get(rayOffsetI));
        BOUNDS_CHECK(emissionBins, rayOffsetI);
        FloatType emissionIntensity = static_cast<FloatType>(emissionBins.Get(rayOffsetI));
        BOUNDS_CHECK(opticalDepthBins, rayOffsetI);
        FloatType opticalDepth = static_cast<FloatType>(opticalDepthBins.Get(rayOffsetI));

        absorptionBins.Set(rayOffsetI, absorbIntensity * tmp);
        // Here we just use emission directly
        emissionBins.Set(rayOffsetI, emissionIntensity * tmp + emission * (1.0f - tmp));
        opticalDepthBins.Set(rayOffsetI, opticalDepth + absorb * segmentLength);
      }
    }
    currentDistance = exitDistance;
  }
};

//
//  IdentifyMissedRay is a debugging routine that detects
//  rays that fail to have any value because of a external
//  intersection and cell intersection mismatch
//
//
class IdentifyMissedRay : public vtkm::worklet::WorkletMapField
{
public:
  vtkm::Id Width;
  vtkm::Id Height;
  vtkm::Vec4f_32 BGColor;
  IdentifyMissedRay(const vtkm::Id width, const vtkm::Id height, vtkm::Vec4f_32 bgcolor)
    : Width(width)
    , Height(height)
    , BGColor(bgcolor)
  {
  }
  using ControlSignature = void(FieldIn, WholeArrayIn);
  using ExecutionSignature = void(_1, _2);


  VTKM_EXEC inline bool IsBGColor(const vtkm::Vec4f_32 color) const
  {
    bool isBG = false;

    if (color[0] == BGColor[0] && color[1] == BGColor[1] && color[2] == BGColor[2] &&
        color[3] == BGColor[3])
      isBG = true;
    return isBG;
  }

  template <typename ColorBufferType>
  VTKM_EXEC inline void operator()(const vtkm::Id& pixelId, ColorBufferType& buffer) const
  {
    vtkm::Id x = pixelId % Width;
    vtkm::Id y = pixelId / Width;

    // Conservative check, we only want to check pixels in the middle
    if (x <= 0 || y <= 0)
      return;
    if (x >= Width - 1 || y >= Height - 1)
      return;
    vtkm::Vec4f_32 pixel;
    pixel[0] = static_cast<vtkm::Float32>(buffer.Get(pixelId * 4 + 0));
    pixel[1] = static_cast<vtkm::Float32>(buffer.Get(pixelId * 4 + 1));
    pixel[2] = static_cast<vtkm::Float32>(buffer.Get(pixelId * 4 + 2));
    pixel[3] = static_cast<vtkm::Float32>(buffer.Get(pixelId * 4 + 3));
    if (!IsBGColor(pixel))
      return;
    vtkm::Id p0 = (y)*Width + (x + 1);
    vtkm::Id p1 = (y)*Width + (x - 1);
    vtkm::Id p2 = (y + 1) * Width + (x);
    vtkm::Id p3 = (y - 1) * Width + (x);
    pixel[0] = static_cast<vtkm::Float32>(buffer.Get(p0 * 4 + 0));
    pixel[1] = static_cast<vtkm::Float32>(buffer.Get(p0 * 4 + 1));
    pixel[2] = static_cast<vtkm::Float32>(buffer.Get(p0 * 4 + 2));
    pixel[3] = static_cast<vtkm::Float32>(buffer.Get(p0 * 4 + 3));
    if (IsBGColor(pixel))
      return;
    pixel[0] = static_cast<vtkm::Float32>(buffer.Get(p1 * 4 + 0));
    pixel[1] = static_cast<vtkm::Float32>(buffer.Get(p1 * 4 + 1));
    pixel[2] = static_cast<vtkm::Float32>(buffer.Get(p1 * 4 + 2));
    pixel[3] = static_cast<vtkm::Float32>(buffer.Get(p1 * 4 + 3));
    if (IsBGColor(pixel))
      return;
    pixel[0] = static_cast<vtkm::Float32>(buffer.Get(p2 * 4 + 0));
    pixel[1] = static_cast<vtkm::Float32>(buffer.Get(p2 * 4 + 1));
    pixel[2] = static_cast<vtkm::Float32>(buffer.Get(p2 * 4 + 2));
    pixel[3] = static_cast<vtkm::Float32>(buffer.Get(p2 * 4 + 3));
    if (IsBGColor(pixel))
      return;
    pixel[0] = static_cast<vtkm::Float32>(buffer.Get(p3 * 4 + 0));
    pixel[1] = static_cast<vtkm::Float32>(buffer.Get(p3 * 4 + 1));
    pixel[2] = static_cast<vtkm::Float32>(buffer.Get(p3 * 4 + 2));
    pixel[3] = static_cast<vtkm::Float32>(buffer.Get(p3 * 4 + 3));
    if (IsBGColor(pixel))
      return;

    printf("Possible error ray missed ray %d\n", (int)pixelId);
  }
};

template <typename FloatType>
class SampleCellAssocCells : public vtkm::worklet::WorkletMapField
{
private:
  vtkm::rendering::raytracing::CellSampler<255> Sampler;
  FloatType SampleDistance;
  FloatType MinScalar;
  FloatType InvDeltaScalar;

public:
  SampleCellAssocCells(const FloatType& sampleDistance,
                       const FloatType& minScalar,
                       const FloatType& maxScalar)
    : SampleDistance(sampleDistance)
    , MinScalar(minScalar)
  {
    InvDeltaScalar = (minScalar == maxScalar) ? 1.f : 1.f / (maxScalar - minScalar);
  }

  using ControlSignature = void(FieldIn,
                                WholeArrayIn,
                                WholeArrayIn,
                                FieldIn,
                                FieldIn,
                                FieldInOut,
                                FieldInOut,
                                WholeArrayIn,
                                WholeArrayInOut,
                                FieldIn);
  using ExecutionSignature = void(_1, _2, _3, _4, _5, _6, _7, _8, _9, WorkIndex, _10);

  template <typename ScalarPortalType,
            typename GhostPortalType,
            typename ColorMapType,
            typename FrameBufferType>
  VTKM_EXEC inline void operator()(const vtkm::Id& currentCell,
                                   ScalarPortalType& scalarPortal,
                                   GhostPortalType& ghostPortal,
                                   const FloatType& enterDistance,
                                   const FloatType& exitDistance,
                                   FloatType& currentDistance,
                                   vtkm::UInt8& rayStatus,
                                   const ColorMapType& colorMap,
                                   FrameBufferType& frameBuffer,
                                   const vtkm::Id& pixelIndex,
                                   const FloatType& maxDistance) const
  {

    if (rayStatus != RAY_ACTIVE)
      return;
    if (int(ghostPortal.Get(currentCell)) != 0)
      return;

    vtkm::Vec4f_32 color;
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 0);
    color[0] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 0));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 1);
    color[1] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 1));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 2);
    color[2] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 2));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 3);
    color[3] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 3));

    vtkm::Float32 scalar;
    BOUNDS_CHECK(scalarPortal, currentCell);
    scalar = vtkm::Float32(scalarPortal.Get(currentCell));
    //
    // There can be mismatches in the initial enter distance and the current distance
    // due to lost rays at cell borders. For now,
    // we will just advance the current position to the enter distance, since otherwise,
    // the pixel would never be sampled.
    //
    if (currentDistance < enterDistance)
      currentDistance = enterDistance;

    const vtkm::Id colorMapSize = colorMap.GetNumberOfValues();
    vtkm::Float32 lerpedScalar;
    lerpedScalar = static_cast<vtkm::Float32>((scalar - MinScalar) * InvDeltaScalar);
    vtkm::Id colorIndex = vtkm::Id(lerpedScalar * vtkm::Float32(colorMapSize));
    if (colorIndex < 0)
      colorIndex = 0;
    if (colorIndex >= colorMapSize)
      colorIndex = colorMapSize - 1;
    BOUNDS_CHECK(colorMap, colorIndex);
    vtkm::Vec4f_32 sampleColor = colorMap.Get(colorIndex);

    while (enterDistance <= currentDistance && currentDistance <= exitDistance)
    {
      //composite
      vtkm::Float32 alpha = sampleColor[3] * (1.f - color[3]);
      color[0] = color[0] + sampleColor[0] * alpha;
      color[1] = color[1] + sampleColor[1] * alpha;
      color[2] = color[2] + sampleColor[2] * alpha;
      color[3] = alpha + color[3];

      currentDistance += SampleDistance;
      if (color[3] >= 1.f || currentDistance >= maxDistance)
      {
        rayStatus = RAY_TERMINATED;
        break;
      }
    }

    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 0);
    frameBuffer.Set(pixelIndex * 4 + 0, color[0]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 1);
    frameBuffer.Set(pixelIndex * 4 + 1, color[1]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 2);
    frameBuffer.Set(pixelIndex * 4 + 2, color[2]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 3);
    frameBuffer.Set(pixelIndex * 4 + 3, color[3]);
  }
}; //class Sample cell

template <typename FloatType>
class SampleCellAssocPoints : public vtkm::worklet::WorkletMapField
{
private:
  vtkm::rendering::raytracing::CellSampler<255> Sampler;
  FloatType SampleDistance;
  FloatType MinScalar;
  FloatType InvDeltaScalar;

public:
  SampleCellAssocPoints(const FloatType& sampleDistance,
                        const FloatType& minScalar,
                        const FloatType& maxScalar)
    : SampleDistance(sampleDistance)
    , MinScalar(minScalar)
  {
    InvDeltaScalar = (minScalar == maxScalar) ? 1.f : 1.f / (maxScalar - minScalar);
  }


  using ControlSignature = void(FieldIn,
                                WholeArrayIn,
                                WholeArrayIn,
                                FieldIn,
                                FieldIn,
                                FieldInOut,
                                FieldIn,
                                FieldInOut,
                                FieldIn,
                                ExecObject meshConnectivity,
                                WholeArrayIn,
                                WholeArrayInOut,
                                FieldIn);
  using ExecutionSignature =
    void(_1, _2, _3, _4, _5, _6, _7, _8, WorkIndex, _9, _10, _11, _12, _13);

  template <typename PointPortalType,
            typename ScalarPortalType,
            typename ColorMapType,
            typename FrameBufferType>
  VTKM_EXEC inline void operator()(const vtkm::Id& currentCell,
                                   PointPortalType& vertices,
                                   ScalarPortalType& scalarPortal,
                                   const FloatType& enterDistance,
                                   const FloatType& exitDistance,
                                   FloatType& currentDistance,
                                   const vtkm::Vec3f_32& dir,
                                   vtkm::UInt8& rayStatus,
                                   const vtkm::Id& pixelIndex,
                                   const vtkm::Vec<FloatType, 3>& origin,
                                   vtkm::rendering::raytracing::MeshConnectivity& meshConn,
                                   const ColorMapType& colorMap,
                                   FrameBufferType& frameBuffer,
                                   const FloatType& maxDistance) const
  {

    if (rayStatus != RAY_ACTIVE)
      return;

    vtkm::Vec4f_32 color;
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 0);
    color[0] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 0));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 1);
    color[1] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 1));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 2);
    color[2] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 2));
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 3);
    color[3] = static_cast<vtkm::Float32>(frameBuffer.Get(pixelIndex * 4 + 3));

    if (color[3] >= 1.f)
    {
      rayStatus = RAY_TERMINATED;
      return;
    }
    vtkm::Vec<vtkm::Float32, 8> scalars;
    vtkm::Vec<vtkm::Vec<FloatType, 3>, 8> points;
    // silence "may" be uninitialized warning
    for (vtkm::Int32 i = 0; i < 8; ++i)
    {
      scalars[i] = 0.f;
      points[i] = vtkm::Vec<FloatType, 3>(0.f, 0.f, 0.f);
    }
    //load local scalar cell data
    vtkm::Id cellConn[8];
    const vtkm::Int32 numIndices = meshConn.GetCellIndices(cellConn, currentCell);
    for (int i = 0; i < numIndices; ++i)
    {
      BOUNDS_CHECK(scalarPortal, cellConn[i]);
      scalars[i] = static_cast<vtkm::Float32>(scalarPortal.Get(cellConn[i]));
      BOUNDS_CHECK(vertices, cellConn[i]);
      points[i] = vtkm::Vec<FloatType, 3>(vertices.Get(cellConn[i]));
    }
    //
    // There can be mismatches in the initial enter distance and the current distance
    // due to lost rays at cell borders. For now,
    // we will just advance the current position to the enter distance, since otherwise,
    // the pixel would never be sampled.
    //
    if (currentDistance < enterDistance)
    {
      currentDistance = enterDistance;
    }

    const vtkm::Id colorMapSize = colorMap.GetNumberOfValues();
    const vtkm::Int32 cellShape = meshConn.GetCellShape(currentCell);

    while (enterDistance <= currentDistance && currentDistance <= exitDistance)
    {
      vtkm::Vec<FloatType, 3> sampleLoc = origin + currentDistance * dir;
      vtkm::Float32 lerpedScalar;
      bool validSample = Sampler.SampleCell(points, scalars, sampleLoc, lerpedScalar, cellShape);
      if (!validSample)
      {
        //
        // There is a slight mismatch between intersections and parametric coordinates
        // which results in a invalid sample very close to the cell edge. Just throw
        // this sample away, and move to the next sample.
        //

        //There should be a sample here, so offset and try again.

        currentDistance += 0.00001f;
        continue;
      }
      lerpedScalar = static_cast<vtkm::Float32>((lerpedScalar - MinScalar) * InvDeltaScalar);
      vtkm::Id colorIndex = vtkm::Id(lerpedScalar * vtkm::Float32(colorMapSize));

      colorIndex = vtkm::Min(vtkm::Max(colorIndex, vtkm::Id(0)), colorMapSize - 1);
      BOUNDS_CHECK(colorMap, colorIndex);
      vtkm::Vec4f_32 sampleColor = colorMap.Get(colorIndex);
      //composite
      sampleColor[3] *= (1.f - color[3]);
      color[0] = color[0] + sampleColor[0] * sampleColor[3];
      color[1] = color[1] + sampleColor[1] * sampleColor[3];
      color[2] = color[2] + sampleColor[2] * sampleColor[3];
      color[3] = sampleColor[3] + color[3];

      currentDistance += SampleDistance;
      if (color[3] >= 1.0 || currentDistance >= maxDistance)
      {
        rayStatus = RAY_TERMINATED;
        break;
      }
    }

    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 0);
    frameBuffer.Set(pixelIndex * 4 + 0, color[0]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 1);
    frameBuffer.Set(pixelIndex * 4 + 1, color[1]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 2);
    frameBuffer.Set(pixelIndex * 4 + 2, color[2]);
    BOUNDS_CHECK(frameBuffer, pixelIndex * 4 + 3);
    frameBuffer.Set(pixelIndex * 4 + 3, color[3]);
  }
}; //class Sample cell

template <typename FloatType>
void ConnectivityTracer::IntersectCell(vtkm::rendering::raytracing::Ray<FloatType>& rays,
                                       detail::RayTracking<FloatType>& tracker)
{
  vtkm::cont::Timer timer;
  timer.Start();
  vtkm::worklet::DispatcherMapField<LocateCell> locateDispatch;
  locateDispatch.Invoke(rays.HitIdx,
                        this->Coords,
                        rays.Dir,
                        *(tracker.EnterDist),
                        *(tracker.ExitDist),
                        tracker.ExitFace,
                        rays.Status,
                        rays.Origin,
                        MeshContainer);

  if (this->CountRayStatus)
    RaysLost = vtkm::rendering::raytracing::RayOperations::GetStatusCount(rays, RAY_LOST);
  this->IntersectTime += timer.GetElapsedTime();
}

template <typename FloatType>
void ConnectivityTracer::AccumulatePathLengths(vtkm::rendering::raytracing::Ray<FloatType>& rays,
                                               detail::RayTracking<FloatType>& tracker)
{
  vtkm::worklet::DispatcherMapField<AddPathLengths> dispatcher;
  dispatcher.Invoke(
    rays.Status, *(tracker.EnterDist), *(tracker.ExitDist), rays.GetBuffer("path_lengths").Buffer);
}

template <typename FloatType>
void ConnectivityTracer::FindLostRays(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker)
{
  vtkm::cont::Timer timer;
  timer.Start();

  vtkm::worklet::DispatcherMapField<RayBumper> bumpDispatch(RayBumper(this->BumpDistance));
  bumpDispatch.Invoke(rays.HitIdx,
                      this->Coords,
                      *(tracker.EnterDist),
                      *(tracker.ExitDist),
                      tracker.ExitFace,
                      rays.Status,
                      rays.Origin,
                      rays.Dir,
                      MeshContainer,
                      &this->Locator);

  this->LostRayTime += timer.GetElapsedTime();
}

template <typename FloatType>
void ConnectivityTracer::SampleCells(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker)
{
  using SampleP = SampleCellAssocPoints<FloatType>;
  using SampleC = SampleCellAssocCells<FloatType>;
  vtkm::cont::Timer timer;
  timer.Start();

  VTKM_ASSERT(rays.Buffers.at(0).GetNumChannels() == 4);

  if (FieldAssocPoints)
  {
    vtkm::worklet::DispatcherMapField<SampleP> dispatcher(
      SampleP(this->SampleDistance,
              vtkm::Float32(this->ScalarBounds.Min),
              vtkm::Float32(this->ScalarBounds.Max)));
    dispatcher.Invoke(rays.HitIdx,
                      this->Coords,
                      vtkm::rendering::raytracing::GetScalarFieldArray(this->ScalarField),
                      *(tracker.EnterDist),
                      *(tracker.ExitDist),
                      tracker.CurrentDistance,
                      rays.Dir,
                      rays.Status,
                      rays.Origin,
                      MeshContainer,
                      this->ColorMap,
                      rays.Buffers.at(0).Buffer,
                      rays.MaxDistance);
  }
  else
  {
    vtkm::worklet::DispatcherMapField<SampleC> dispatcher(
      SampleC(this->SampleDistance,
              vtkm::Float32(this->ScalarBounds.Min),
              vtkm::Float32(this->ScalarBounds.Max)));

    dispatcher.Invoke(rays.HitIdx,
                      vtkm::rendering::raytracing::GetScalarFieldArray(this->ScalarField),
                      GhostField.GetData().ExtractComponent<vtkm::UInt8>(0),
                      *(tracker.EnterDist),
                      *(tracker.ExitDist),
                      tracker.CurrentDistance,
                      rays.Status,
                      this->ColorMap,
                      rays.Buffers.at(0).Buffer,
                      rays.MaxDistance);
  }

  this->SampleTime += timer.GetElapsedTime();
}

template <typename FloatType>
void ConnectivityTracer::IntegrateCells(vtkm::rendering::raytracing::Ray<FloatType>& rays,
                                        detail::RayTracking<FloatType>& tracker)
{
  vtkm::cont::Timer timer;
  timer.Start();

  vtkm::cont::ArrayHandle<FloatType> optical_depth = rays.GetBuffer("optical_depths").Buffer;

  if (HasEmission)
  {
    vtkm::cont::ArrayHandle<FloatType> absorption = rays.Buffers.at(0).Buffer;
    vtkm::cont::ArrayHandle<FloatType> emission = rays.GetBuffer("emission").Buffer;
    vtkm::worklet::DispatcherMapField<IntegrateEmission> dispatcher(IntegrateEmission(NumEnergyGroups,
                                                                                      UnitScalar,
                                                                                      DivideEmisByAbsorb));
    dispatcher.Invoke(rays.Status,
                      *(tracker.EnterDist),
                      *(tracker.ExitDist),
                      rays.Distance,
                      ScalarField.GetData(),
                      EmissionField.GetData(),
                      absorption,
                      emission,
                      optical_depth,
                      rays.HitIdx);
  }
  else
  {
    vtkm::worklet::DispatcherMapField<Integrate> dispatcher(Integrate(NumEnergyGroups, UnitScalar));
    dispatcher.Invoke(rays.Status,
                      *(tracker.EnterDist),
                      *(tracker.ExitDist),
                      rays.Distance,
                      ScalarField.GetData(),
                      optical_depth,
                      rays.HitIdx);
  }

  IntegrateTime += timer.GetElapsedTime();
}

// template <typename FloatType>
// void ConnectivityTracer<CellType>::PrintDebugRay(Ray<FloatType>& rays, vtkm::Id rayId)
// {
//   vtkm::Id index = -1;
//   for (vtkm::Id i = 0; i < rays.NumRays; ++i)
//   {
//     if (rays.PixelIdx.WritePortal().Get(i) == rayId)
//     {
//       index = i;
//       break;
//     }
//   }
//   if (index == -1)
//   {
//     return;
//   }

//   std::cout << "++++++++RAY " << rayId << "++++++++\n";
//   std::cout << "Status: " << (int)rays.Status.WritePortal().Get(index) << "\n";
//   std::cout << "HitIndex: " << rays.HitIdx.WritePortal().Get(index) << "\n";
//   std::cout << "Dist " << rays.Distance.WritePortal().Get(index) << "\n";
//   std::cout << "MinDist " << rays.MinDistance.WritePortal().Get(index) << "\n";
//   std::cout << "Origin " << rays.Origin.ReadPortal().Get(index) << "\n";
//   std::cout << "Dir " << rays.Dir.ReadPortal().Get(index) << "\n";
//   std::cout << "+++++++++++++++++++++++++\n";
// }

template <typename FloatType>
void ConnectivityTracer::OffsetMinDistances(vtkm::rendering::raytracing::Ray<FloatType>& rays)
{
  vtkm::worklet::DispatcherMapField<AdvanceRay<FloatType>> dispatcher(
    AdvanceRay<FloatType>(FloatType(this->BumpDistance)));
  dispatcher.Invoke(rays.Status, rays.MinDistance);
}

template <typename FloatType>
void ConnectivityTracer::FindMeshEntry(vtkm::rendering::raytracing::Ray<FloatType>& rays)
{
  vtkm::cont::Timer entryTimer;
  entryTimer.Start();
  //
  // if ray misses the external face it will be marked RAY_EXITED_MESH
  //
  MeshContainer->FindEntry(rays);
  MeshEntryTime += entryTimer.GetElapsedTime();
}

template <typename FloatType>
void ConnectivityTracer::IntegrateMeshSegment(vtkm::rendering::raytracing::Ray<FloatType>& rays)
{
  this->Init(); // sets sample distance
  detail::RayTracking<FloatType> rayTracker;
  rayTracker.Init(rays.NumRays, rays.Distance);

  bool hasPathLengths = rays.HasBuffer("path_lengths");

  if (this->Integrator == Volume)
  {
    vtkm::worklet::DispatcherMapField<detail::AdjustSample> adispatcher(SampleDistance);
    adispatcher.Invoke(rays.Status, rayTracker.CurrentDistance);
  }

  while (vtkm::rendering::raytracing::RayOperations::RaysInMesh(rays))
  {
    //
    // Rays the leave the mesh will be marked as RAYEXITED_MESH
    this->IntersectCell(rays, rayTracker);
    //
    // If the ray was lost due to precision issues, we find it.
    // If it is marked RAY_ABANDONED, then something went wrong.
    //
    this->FindLostRays(rays, rayTracker);
    //
    // integrate along the ray
    //
    if (this->Integrator == Volume)
      this->SampleCells(rays, rayTracker);
    else
      this->IntegrateCells(rays, rayTracker);

    if (hasPathLengths)
    {
      this->AccumulatePathLengths(rays, rayTracker);
    }
    //swap enter and exit distances
    rayTracker.Swap();
    if (this->CountRayStatus)
      this->PrintRayStatus(rays);
  } //for
}

template <typename FloatType>
void ConnectivityTracer::FullTrace(vtkm::rendering::raytracing::Ray<FloatType>& rays)
{

  this->RaysLost = 0;
  vtkm::rendering::raytracing::RayOperations::ResetStatus(rays, RAY_EXITED_MESH);

  if (this->CountRayStatus)
  {
    this->PrintRayStatus(rays);
  }

  bool cullMissedRays = true;
  bool workRemaining = true;

  do
  {
    FindMeshEntry(rays);

    if (cullMissedRays)
    {
      vtkm::cont::ArrayHandle<vtkm::UInt8> activeRays;
      activeRays = vtkm::rendering::raytracing::RayOperations::CompactActiveRays(rays);
      cullMissedRays = false;
    }

    IntegrateMeshSegment(rays);

    workRemaining = vtkm::rendering::raytracing::RayOperations::RaysProcessed(rays) != rays.NumRays;
    //
    // Ensure that we move the current distance forward some
    // epsilon so we don't re-enter the cell we just left.
    //
    if (workRemaining)
    {
      vtkm::rendering::raytracing::RayOperations::CopyDistancesToMin(rays);
      this->OffsetMinDistances(rays);
    }
  } while (workRemaining);
}

template <typename FloatType>
void ConnectivityTracer::PartialTrace(vtkm::rendering::raytracing::Ray<FloatType> &rays,
                                      std::vector<PartialComposite<FloatType>> &partials)
{

  //this->CountRayStatus = true;
  bool hasPathLengths = rays.HasBuffer("path_lengths");
  RaysLost = 0;
  vtkm::rendering::raytracing::RayOperations::ResetStatus(rays, RAY_EXITED_MESH);

  if (CountRayStatus)
  {
    PrintRayStatus(rays);
  }

  bool workRemaining = true;

  do
  {
    FindMeshEntry(rays);

    vtkm::cont::ArrayHandle<vtkm::UInt8> activeRays;
    activeRays = vtkm::rendering::raytracing::RayOperations::CompactActiveRays(rays);

    if (0 == rays.NumRays)
    {
      break;
    }

    IntegrateMeshSegment(rays);

    PartialComposite<FloatType> partial;
    partial.OpticalDepth = rays.GetBuffer("optical_depths").Copy();
    vtkm::cont::Algorithm::Copy(rays.Distance, partial.Distances);
    vtkm::cont::Algorithm::Copy(rays.PixelIdx, partial.PixelIds);

    if (HasEmission)
    {
      partial.Transmission = rays.Buffers.at(0).Copy();
      partial.Intensity = rays.GetBuffer("emission").Copy();
    }

    if (hasPathLengths)
    {
      partial.PathLengths = rays.GetBuffer("path_lengths").Copy().Buffer;
    }

    partials.push_back(partial);

    // reset buffers
    rays.GetBuffer("optical_depths").InitConst(0.0f);

    if (HasEmission)
    {
      rays.Buffers.at(0).InitConst(1.0f);
      rays.GetBuffer("emission").InitConst(0.0f);
    }

    if (hasPathLengths)
    {
      rays.GetBuffer("path_lengths").InitConst(0.0f);
    }

    workRemaining = vtkm::rendering::raytracing::RayOperations::RaysProcessed(rays) != rays.NumRays;
    //
    // Ensure that we move the current distance forward some
    // epsilon so we don't re-enter the cell we just left.
    //
    if (workRemaining)
    {
      vtkm::rendering::raytracing::RayOperations::CopyDistancesToMin(rays);
      OffsetMinDistances(rays);
    }
  } while (workRemaining);
}

template class detail::RayTracking<vtkm::Float32>;
template class detail::RayTracking<vtkm::Float64>;

// template struct vtkm::rendering::raytracing::PartialComposite<vtkm::Float32>;
// template struct vtkm::rendering::raytracing::PartialComposite<vtkm::Float64>;

template void ConnectivityTracer::FullTrace<vtkm::Float32>(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

template void
ConnectivityTracer::PartialTrace<vtkm::Float32>(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                                                std::vector<PartialComposite<vtkm::Float32>> &partials);

template void ConnectivityTracer::IntegrateMeshSegment<vtkm::Float32>(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

template void ConnectivityTracer::FindMeshEntry<vtkm::Float32>(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays);

template void ConnectivityTracer::FullTrace<vtkm::Float64>(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);

template void
ConnectivityTracer::PartialTrace<vtkm::Float64>(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                                                std::vector<PartialComposite<vtkm::Float64>> &partials);

template void ConnectivityTracer::IntegrateMeshSegment<vtkm::Float64>(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);

template void ConnectivityTracer::FindMeshEntry<vtkm::Float64>(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays);
}
}
} // namespace vtkm::rendering::raytracing
