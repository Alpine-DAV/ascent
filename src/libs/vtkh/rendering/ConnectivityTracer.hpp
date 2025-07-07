//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================
#ifndef vtkh_rendering_raytracing_ConnectivityTracer_h
#define vtkh_rendering_raytracing_ConnectivityTracer_h

#include "MeshConnectivityContainers.hpp"
#include "PartialComposite.hpp"

#include <vtkh/vtkh_exports.h>

#include <vtkm/cont/CellLocatorGeneral.h>
#include <vtkm/cont/Timer.h>
#include <vtkm/rendering/raytracing/CellIntersector.h>
#include <vtkm/rendering/raytracing/CellSampler.h>
#include <vtkm/rendering/raytracing/RayOperations.h>
#include <vtkm/worklet/WorkletMapField.h>

namespace vtkh
{
namespace rendering
{
namespace raytracing
{
namespace detail
{

//forward declare so we can be friends
struct RenderFunctor;

//
//  Ray tracker manages memory and pointer
//  swapping for current cell intersection data
//
template <typename FloatType>
class RayTracking
{
public:
  vtkm::cont::ArrayHandle<vtkm::Int32> ExitFace;
  vtkm::cont::ArrayHandle<FloatType> CurrentDistance;
  vtkm::cont::ArrayHandle<FloatType> Distance1;
  vtkm::cont::ArrayHandle<FloatType> Distance2;
  vtkm::cont::ArrayHandle<FloatType>* EnterDist;
  vtkm::cont::ArrayHandle<FloatType>* ExitDist;

  RayTracking()
  {
    EnterDist = &Distance1;
    ExitDist = &Distance2;
  }

  void Compact(vtkm::cont::ArrayHandle<FloatType>& compactedDistances,
               vtkm::cont::ArrayHandle<vtkm::UInt8>& masks);

  void Init(const vtkm::Id size, vtkm::cont::ArrayHandle<FloatType>& distances);

  void Swap();
};

} //namespace detail

/**
 * \brief ConnectivityTracer is volumetric ray tracer for unstructured
 *        grids. Capabilities include volume rendering and integrating
 *        absorption and emission of N energy groups for simulated
 *        radiograhy.
 */
class VTKH_API ConnectivityTracer
{
public:
  ConnectivityTracer()
    : MeshContainer(nullptr)
    , BumpEpsilon(1e-3)
    , CountRayStatus(false)
    , UnitScalar(1.f)
  {
  }

  ~ConnectivityTracer()
  {
    if (MeshContainer != nullptr)
    {
      delete MeshContainer;
    }
  }

  enum IntegrationMode
  {
    Volume,
    Energy
  };

  void SetVolumeData(const vtkm::cont::Field& scalarField,
                     const vtkm::Range& scalarBounds,
                     const vtkm::cont::UnknownCellSet& cellSet,
                     const vtkm::cont::CoordinateSystem& coords,
                     const vtkm::cont::Field& ghostField);

  void SetEnergyData(const vtkm::cont::Field& absorption,
                     const vtkm::Int32 numBins,
                     const vtkm::cont::UnknownCellSet& cellSet,
                     const vtkm::cont::CoordinateSystem& coords,
                     const vtkm::cont::Field& emission);

  void SetBackgroundColor(const vtkm::Vec4f_32& backgroundColor);
  void SetSampleDistance(const vtkm::Float32& distance);
  void SetColorMap(const vtkm::cont::ArrayHandle<vtkm::Vec4f_32>& colorMap);

  MeshConnectivityContainer* GetMeshContainer() { return MeshContainer; }

  void Init();

  void SetDebugOn(bool on) { CountRayStatus = on; }

  void SetUnitScalar(const vtkm::Float32 unitScalar) { UnitScalar = unitScalar; }
  void SetEpsilon(const vtkm::Float64 epsilon) { BumpEpsilon = epsilon; }


  vtkm::Id GetNumberOfMeshCells() const;

  void ResetTimers();
  void LogTimers();

  ///
  /// Traces rays fully through the mesh. Rays can exit and re-enter
  /// multiple times before leaving the domain. This is fast path for
  /// structured meshs or meshes that are not interlocking.
  /// Note: rays will be compacted
  ///
  template <typename FloatType>
  void FullTrace(vtkm::rendering::raytracing::Ray<FloatType>& rays);

  ///
  /// Integrates rays through the mesh. If rays leave the mesh and
  /// re-enter, then those become two separate partial composites.
  /// This is need to support domain decompositions that are like
  /// puzzle pieces. Note: rays will be compacted
  ///
  template <typename FloatType>
  void PartialTrace(vtkm::rendering::raytracing::Ray<FloatType> &rays,
                    std::vector<PartialComposite<FloatType>> &partials);

  ///
  /// Integrates the active rays though the mesh until all rays
  /// have exited.
  ///  Precondition: rays.HitIdx is set to a valid mesh cell
  ///
  template <typename FloatType>
  void IntegrateMeshSegment(vtkm::rendering::raytracing::Ray<FloatType>& rays);

  ///
  /// Find the entry point in the mesh
  ///
  template <typename FloatType>
  void FindMeshEntry(vtkm::rendering::raytracing::Ray<FloatType>& rays);

private:
  template <typename FloatType>
  void IntersectCell(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void AccumulatePathLengths(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void FindLostRays(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void SampleCells(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void IntegrateCells(vtkm::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void OffsetMinDistances(vtkm::rendering::raytracing::Ray<FloatType>& rays);

  template <typename FloatType>
  void PrintRayStatus(vtkm::rendering::raytracing::Ray<FloatType>& rays);

protected:
  // Data set info
  vtkm::cont::Field ScalarField;
  vtkm::cont::Field EmissionField;
  vtkm::cont::Field GhostField;
  vtkm::cont::UnknownCellSet CellSet;
  vtkm::cont::CoordinateSystem Coords;
  vtkm::Range ScalarBounds;
  vtkm::Float32 BoundingBox[6];

  vtkm::cont::ArrayHandle<vtkm::Vec4f_32> ColorMap;

  vtkm::Vec4f_32 BackgroundColor;
  vtkm::Float32 SampleDistance;
  vtkm::Id RaysLost;
  IntegrationMode Integrator;

  MeshConnectivityContainer* MeshContainer;
  vtkm::cont::CellLocatorGeneral Locator;
  vtkm::Float64 BumpEpsilon;
  vtkm::Float64 BumpDistance;
  //
  // flags
  bool CountRayStatus;
  bool MeshConnIsConstructed;
  bool DebugFiltersOn;
  bool ReEnterMesh; // Do not try to re-enter the mesh
  bool CreatePartialComposites;
  bool FieldAssocPoints;
  bool HasEmission; // Mode for integrating through energy bins

  // timers
  vtkm::Float64 IntersectTime;
  vtkm::Float64 IntegrateTime;
  vtkm::Float64 SampleTime;
  vtkm::Float64 LostRayTime;
  vtkm::Float64 MeshEntryTime;
  vtkm::Float32 UnitScalar;

}; // class ConnectivityTracer<CellType,ConnectivityType>
}
}
} // namespace vtkm::rendering::raytracing
#endif
