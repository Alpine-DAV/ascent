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

#ifndef vtkh_rendering_ConnectivityProxy_h
#define vtkh_rendering_ConnectivityProxy_h

#include "PartialComposite.hpp"

#include <vtkh/vtkh_exports.h>

#include <viskores/cont/DataSet.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/raytracing/Camera.h>
#include <viskores/rendering/raytracing/Ray.h>
#include <viskores/rendering/raytracing/RayOperations.h>

namespace vtkh
{
namespace rendering
{

using PartialVector64 = std::vector<vtkh::rendering::raytracing::PartialComposite<viskores::Float64>>;
using PartialVector32 = std::vector<vtkh::rendering::raytracing::PartialComposite<viskores::Float32>>;

class VTKH_API ConnectivityProxy
{
public:
  ConnectivityProxy(const viskores::cont::DataSet& dataset, const std::string& fieldName);

  ConnectivityProxy(const viskores::cont::DataSet& dataSet,
                    const std::string& fieldName,
                    const std::string& coordinateName);

  ConnectivityProxy(const viskores::cont::UnknownCellSet& cellset,
                    const viskores::cont::CoordinateSystem& coords,
                    const viskores::cont::Field& scalarField);

  ConnectivityProxy(const ConnectivityProxy&);
  ConnectivityProxy& operator=(const ConnectivityProxy&);

  ConnectivityProxy(ConnectivityProxy&&) noexcept;
  ConnectivityProxy& operator=(ConnectivityProxy&&) noexcept;

  ~ConnectivityProxy();

  enum struct RenderMode
  {
    Volume,
    Energy,
  };

  void SetRenderMode(RenderMode mode);
  void SetSampleDistance(const viskores::Float32&);
  void SetScalarField(const std::string& fieldName);
  void SetEmissionField(const std::string& fieldName);
  void SetScalarRange(const viskores::Range& range);
  void SetColorMap(viskores::cont::ArrayHandle<viskores::Vec4f_32>& colormap);
  void SetCompositeBackground(bool on);
  void SetDebugPrints(bool on);
  void SetUnitScalar(viskores::Float32 unitScalar);
  void SetDivideEmisByAbsorb(const bool divide_emis_by_absorb);
  void SetEpsilon(viskores::Float64 epsilon); // epsilon for bumping lost rays

  viskores::Bounds GetSpatialBounds();
  viskores::Range GetScalarFieldRange();
  viskores::Range GetScalarRange();

  void Trace(const viskores::rendering::Camera& camera, viskores::rendering::CanvasRayTracer* canvas);
  void Trace(viskores::rendering::raytracing::Ray<viskores::Float64>& rays);
  void Trace(viskores::rendering::raytracing::Ray<viskores::Float32>& rays);

  void PartialTrace(viskores::rendering::raytracing::Ray<viskores::Float64>& rays, PartialVector64& partials);
  void PartialTrace(viskores::rendering::raytracing::Ray<viskores::Float32>& rays, PartialVector32& partials);

protected:
  struct InternalsType;
  std::unique_ptr<InternalsType> Internals;
};
}
} //namespace viskores::rendering
#endif //vtk_m_rendering_ConnectivityProxy_h
