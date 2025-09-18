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

#ifndef vtkh_rendering_raytracing_ConnectivityTracer_h
#define vtkh_rendering_raytracing_ConnectivityTracer_h

#include "MeshConnectivityContainers.hpp"
#include "PartialComposite.hpp"

#include <vtkh/vtkh_exports.h>

#include <viskores/cont/CellLocatorGeneral.h>
#include <viskores/cont/Timer.h>
#include <viskores/rendering/raytracing/CellIntersector.h>
#include <viskores/rendering/raytracing/CellSampler.h>
#include <viskores/rendering/raytracing/RayOperations.h>
#include <viskores/worklet/WorkletMapField.h>

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
  viskores::cont::ArrayHandle<viskores::Int32> ExitFace;
  viskores::cont::ArrayHandle<FloatType> CurrentDistance;
  viskores::cont::ArrayHandle<FloatType> Distance1;
  viskores::cont::ArrayHandle<FloatType> Distance2;
  viskores::cont::ArrayHandle<FloatType>* EnterDist;
  viskores::cont::ArrayHandle<FloatType>* ExitDist;

  RayTracking()
  {
    EnterDist = &Distance1;
    ExitDist = &Distance2;
  }

  void Compact(viskores::cont::ArrayHandle<FloatType>& compactedDistances,
               viskores::cont::ArrayHandle<viskores::UInt8>& masks);

  void Init(const viskores::Id size, viskores::cont::ArrayHandle<FloatType>& distances);

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

  void SetVolumeData(const viskores::cont::Field& scalarField,
                     const viskores::Range& scalarBounds,
                     const viskores::cont::UnknownCellSet& cellSet,
                     const viskores::cont::CoordinateSystem& coords,
                     const viskores::cont::Field& ghostField);

  // Absporption-only case
  void SetEnergyData(const viskores::cont::Field& absorption,
                     const viskores::Int32 numBins,
                     const viskores::cont::UnknownCellSet& cellSet,
                     const viskores::cont::CoordinateSystem& coords);

  // Absorption + Emission case
  void SetEnergyData(const viskores::cont::Field& absorption,
                     const viskores::Int32 numBins,
                     const viskores::cont::UnknownCellSet& cellSet,
                     const viskores::cont::CoordinateSystem& coords,
                     const viskores::cont::Field& emission);

  void SetBackgroundColor(const viskores::Vec4f_32& backgroundColor);
  void SetSampleDistance(const viskores::Float32& distance);
  void SetColorMap(const viskores::cont::ArrayHandle<viskores::Vec4f_32>& colorMap);

  MeshConnectivityContainer* GetMeshContainer() { return MeshContainer; }

  void Init();

  void SetDebugOn(bool on) { CountRayStatus = on; }

  void SetUnitScalar(const viskores::Float32 unitScalar) { UnitScalar = unitScalar; }
  void SetDivideEmisByAbsorb(const bool divide_emis_by_absorb) {DivideEmisByAbsorb = divide_emis_by_absorb; }
  void SetEpsilon(const viskores::Float64 epsilon) { BumpEpsilon = epsilon; }


  viskores::Id GetNumberOfMeshCells() const;

  void ResetTimers();
  void LogTimers();

  ///
  /// Traces rays fully through the mesh. Rays can exit and re-enter
  /// multiple times before leaving the domain. This is fast path for
  /// structured meshs or meshes that are not interlocking.
  /// Note: rays will be compacted
  ///
  template <typename FloatType>
  void FullTrace(viskores::rendering::raytracing::Ray<FloatType>& rays);

  ///
  /// Integrates rays through the mesh. If rays leave the mesh and
  /// re-enter, then those become two separate partial composites.
  /// This is need to support domain decompositions that are like
  /// puzzle pieces. Note: rays will be compacted
  ///
  template <typename FloatType>
  void PartialTrace(viskores::rendering::raytracing::Ray<FloatType> &rays,
                    std::vector<PartialComposite<FloatType>> &partials);

  ///
  /// Integrates the active rays though the mesh until all rays
  /// have exited.
  ///  Precondition: rays.HitIdx is set to a valid mesh cell
  ///
  template <typename FloatType>
  void IntegrateMeshSegment(viskores::rendering::raytracing::Ray<FloatType>& rays);

  ///
  /// Find the entry point in the mesh
  ///
  template <typename FloatType>
  void FindMeshEntry(viskores::rendering::raytracing::Ray<FloatType>& rays);

private:
  template <typename FloatType>
  void IntersectCell(viskores::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void AccumulatePathLengths(viskores::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void FindLostRays(viskores::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void SampleCells(viskores::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void IntegrateCells(viskores::rendering::raytracing::Ray<FloatType>& rays, detail::RayTracking<FloatType>& tracker);

  template <typename FloatType>
  void OffsetMinDistances(viskores::rendering::raytracing::Ray<FloatType>& rays);

  template <typename FloatType>
  void PrintRayStatus(viskores::rendering::raytracing::Ray<FloatType>& rays);

protected:
  // Data set info
  viskores::cont::Field ScalarField;
  viskores::cont::Field EmissionField;
  viskores::cont::Field GhostField;
  viskores::cont::UnknownCellSet CellSet;
  viskores::cont::CoordinateSystem Coords;
  viskores::Range ScalarBounds;
  viskores::Float32 BoundingBox[6];

  viskores::cont::ArrayHandle<viskores::Vec4f_32> ColorMap;

  viskores::Vec4f_32 BackgroundColor;
  viskores::Float32 SampleDistance;
  viskores::Id RaysLost;
  IntegrationMode Integrator;

  MeshConnectivityContainer* MeshContainer;
  viskores::cont::CellLocatorGeneral Locator;
  viskores::Float64 BumpEpsilon;
  viskores::Float64 BumpDistance;
  //
  // flags
  bool CountRayStatus;
  bool MeshConnIsConstructed;
  bool DebugFiltersOn;
  bool ReEnterMesh; // Do not try to re-enter the mesh
  bool CreatePartialComposites;
  bool FieldAssocPoints;
  bool HasEmission; // Mode for integrating through energy bins
  bool DivideEmisByAbsorb;

  // timers
  viskores::Float64 IntersectTime;
  viskores::Float64 IntegrateTime;
  viskores::Float64 SampleTime;
  viskores::Float64 LostRayTime;
  viskores::Float64 MeshEntryTime;
  viskores::Float32 UnitScalar;

}; // class ConnectivityTracer<CellType,ConnectivityType>
}
}
} // namespace viskores::rendering::raytracing
#endif
