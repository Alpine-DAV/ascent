//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//============================================================================

#include "ConnectivityProxy.hpp"
#include "ConnectivityTracer.hpp"

namespace vtkh
{
namespace rendering
{
struct ConnectivityProxy::InternalsType
{
protected:
  using ColorMapType = vtkm::cont::ArrayHandle<vtkm::Vec4f_32>;
  using TracerType = vtkh::rendering::raytracing::ConnectivityTracer;

  TracerType Tracer;
  std::string CoordinateName;
  std::string FieldName;
  std::string EmissionFieldName;
  RenderMode Mode;
  vtkm::Bounds SpatialBounds;
  ColorMapType ColorMap;
  vtkm::cont::DataSet Dataset;
  vtkm::Range ScalarRange;
  bool CompositeBackground;

public:
  InternalsType(const vtkm::cont::DataSet& dataSet,
                const std::string& coordinateName,
                const std::string& fieldName)
  {
    Dataset = dataSet;
    CoordinateName = coordinateName;
    Mode = RenderMode::Energy;
    CompositeBackground = true;
    if (!fieldName.empty())
    {
      this->SetScalarField(fieldName);
    }
  }

  VTKM_CONT
  void SetUnitScalar(vtkm::Float32 unitScalar) { Tracer.SetUnitScalar(unitScalar); }

  VTKM_CONT
  void SetDivideEmisByAbsorb(const bool divide_emis_by_absorb)
  {
    Tracer.SetDivideEmisByAbsorb(divide_emis_by_absorb);
  }

  void SetSampleDistance(const vtkm::Float32& distance)
  {
    if (this->Mode != RenderMode::Volume)
    {
      throw vtkm::cont::ErrorBadValue(
        "Conn Proxy: volume mode must be set before sample distance set");
    }
    Tracer.SetSampleDistance(distance);
  }

  VTKM_CONT
  void SetRenderMode(RenderMode mode) { Mode = mode; }

  VTKM_CONT
  RenderMode GetRenderMode() { return Mode; }

  VTKM_CONT
  void SetScalarField(const std::string& fieldName)
  {
    this->FieldName = fieldName;
    const vtkm::cont::ArrayHandle<vtkm::Range> range =
      this->Dataset.GetField(this->FieldName).GetRange();
    ScalarRange = range.ReadPortal().Get(0);
  }

  VTKM_CONT
  void SetColorMap(vtkm::cont::ArrayHandle<vtkm::Vec4f_32>& colormap)
  {
    Tracer.SetColorMap(colormap);
  }

  VTKM_CONT
  void SetCompositeBackground(bool on) { CompositeBackground = on; }

  VTKM_CONT
  void SetDebugPrints(bool on) { Tracer.SetDebugOn(on); }

  VTKM_CONT
  void SetEpsilon(vtkm::Float64 epsilon) { Tracer.SetEpsilon(epsilon); }

  VTKM_CONT
  void SetEmissionField(const std::string& fieldName)
  {
    if (this->Mode != RenderMode::Energy)
    {
      throw vtkm::cont::ErrorBadValue(
        "Conn Proxy: energy mode must be set before setting emission field");
    }
    this->EmissionFieldName = fieldName;
  }

  VTKM_CONT
  vtkm::Bounds GetSpatialBounds() const { return SpatialBounds; }

  VTKM_CONT
  vtkm::Range GetScalarFieldRange()
  {
    const vtkm::cont::ArrayHandle<vtkm::Range> range =
      this->Dataset.GetField(this->FieldName).GetRange();
    ScalarRange = range.ReadPortal().Get(0);
    return ScalarRange;
  }

  VTKM_CONT
  void SetScalarRange(const vtkm::Range& range) { this->ScalarRange = range; }

  VTKM_CONT
  vtkm::Range GetScalarRange() { return this->ScalarRange; }

  VTKM_CONT
  void Trace(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays)
  {

    if (this->Mode == RenderMode::Volume)
    {
      Tracer.SetVolumeData(this->Dataset.GetField(this->FieldName),
                           this->ScalarRange,
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetGhostCellField());
    }
    else
    {
      Tracer.SetEnergyData(this->Dataset.GetField(this->FieldName),
                           rays.Buffers.at(0).GetNumChannels(),
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetField(this->EmissionFieldName));
    }

    Tracer.FullTrace(rays);
  }

  VTKM_CONT
  void Trace(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays)
  {
    if (this->Mode == RenderMode::Volume)
    {
      Tracer.SetVolumeData(this->Dataset.GetField(this->FieldName),
                           this->ScalarRange,
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetGhostCellField());
    }
    else
    {
      Tracer.SetEnergyData(this->Dataset.GetField(this->FieldName),
                           rays.Buffers.at(0).GetNumChannels(),
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetField(this->EmissionFieldName));
    }

    Tracer.FullTrace(rays);
  }

  VTKM_CONT
  void PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                    PartialVector64 &partials)
  {
    if (this->Mode == RenderMode::Volume)
    {
      Tracer.SetVolumeData(this->Dataset.GetField(this->FieldName),
                           this->ScalarRange,
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetGhostCellField());
    }
    else
    {
      Tracer.SetEnergyData(this->Dataset.GetField(this->FieldName),
                           rays.Buffers.at(0).GetNumChannels(),
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetField(this->EmissionFieldName));
    }

    Tracer.PartialTrace(rays, partials);
  }

  VTKM_CONT
  void PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays,
                    PartialVector32 &partials)
  {
    if (this->Mode == RenderMode::Volume)
    {
      Tracer.SetVolumeData(this->Dataset.GetField(this->FieldName),
                           this->ScalarRange,
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetGhostCellField());
    }
    else
    {
      Tracer.SetEnergyData(this->Dataset.GetField(this->FieldName),
                           rays.Buffers.at(0).GetNumChannels(),
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetField(this->EmissionFieldName));
    }

    Tracer.PartialTrace(rays, partials);
  }

  VTKM_CONT
  void Trace(const vtkm::rendering::Camera& camera, vtkm::rendering::CanvasRayTracer* canvas)
  {

    if (canvas == nullptr)
    {
      throw vtkm::cont::ErrorBadValue("Conn Proxy: null canvas");
    }
    vtkm::rendering::raytracing::Camera rayCamera;
    rayCamera.SetParameters(
      camera, (vtkm::Int32)canvas->GetWidth(), (vtkm::Int32)canvas->GetHeight());
    vtkm::rendering::raytracing::Ray<vtkm::Float32> rays;
    rayCamera.CreateRays(rays, this->Dataset.GetCoordinateSystem(this->CoordinateName).GetBounds());
    rays.Buffers.at(0).InitConst(0.f);
    vtkm::rendering::raytracing::RayOperations::MapCanvasToRays(rays, camera, *canvas);

    if (this->Mode == RenderMode::Volume)
    {
      Tracer.SetVolumeData(this->Dataset.GetField(this->FieldName),
                           this->ScalarRange,
                           this->Dataset.GetCellSet(),
                           this->Dataset.GetCoordinateSystem(this->CoordinateName),
                           this->Dataset.GetGhostCellField());
    }
    else
    {
      throw vtkm::cont::ErrorBadValue("ENERGY MODE Not implemented for this use case\n");
    }

    Tracer.FullTrace(rays);

    canvas->WriteToCanvas(rays, rays.Buffers.at(0).Buffer, camera);
    if (CompositeBackground)
    {
      canvas->BlendBackground();
    }
  }
};


VTKM_CONT
ConnectivityProxy::ConnectivityProxy(const vtkm::cont::DataSet& dataSet,
                                     const std::string& fieldName)
  : Internals(
      std::make_unique<InternalsType>(dataSet, dataSet.GetCoordinateSystemName(), fieldName))
{
}

VTKM_CONT
ConnectivityProxy::ConnectivityProxy(const vtkm::cont::DataSet& dataSet,
                                     const std::string& fieldName,
                                     const std::string& coordinateName)
  : Internals(std::make_unique<InternalsType>(dataSet, coordinateName, fieldName))
{
}


VTKM_CONT
ConnectivityProxy::ConnectivityProxy(const vtkm::cont::UnknownCellSet& cellset,
                                     const vtkm::cont::CoordinateSystem& coords,
                                     const vtkm::cont::Field& scalarField)
{
  vtkm::cont::DataSet dataset;
  dataset.SetCellSet(cellset);
  dataset.AddCoordinateSystem(coords);
  dataset.AddField(scalarField);

  Internals = std::make_unique<InternalsType>(dataset, coords.GetName(), scalarField.GetName());
}

ConnectivityProxy::ConnectivityProxy(const ConnectivityProxy& rhs)
  : Internals(nullptr)
{
  // rhs might have been moved, its Internal would be nullptr
  if (rhs.Internals)
  {
    Internals = std::make_unique<InternalsType>(*rhs.Internals);
  }
}

ConnectivityProxy& ConnectivityProxy::operator=(const ConnectivityProxy& rhs)
{
  // both *this and rhs might have been moved.
  if (!rhs.Internals)
  {
    Internals.reset();
  }
  else if (!Internals)
  {
    Internals = std::make_unique<InternalsType>(*rhs.Internals);
  }
  else
  {
    *Internals = *rhs.Internals;
  }

  return *this;
}

VTKM_CONT
ConnectivityProxy::ConnectivityProxy(ConnectivityProxy&&) noexcept = default;
VTKM_CONT
ConnectivityProxy& ConnectivityProxy::operator=(vtkh::rendering::ConnectivityProxy&&) noexcept =
  default;
VTKM_CONT
ConnectivityProxy::~ConnectivityProxy() = default;

VTKM_CONT
void ConnectivityProxy::SetSampleDistance(const vtkm::Float32& distance)
{
  Internals->SetSampleDistance(distance);
}

VTKM_CONT
void ConnectivityProxy::SetRenderMode(RenderMode mode)
{
  Internals->SetRenderMode(mode);
}

VTKM_CONT
void ConnectivityProxy::SetScalarField(const std::string& fieldName)
{
  Internals->SetScalarField(fieldName);
}

VTKM_CONT
void ConnectivityProxy::SetColorMap(vtkm::cont::ArrayHandle<vtkm::Vec4f_32>& colormap)
{
  Internals->SetColorMap(colormap);
}

VTKM_CONT
void ConnectivityProxy::SetEmissionField(const std::string& fieldName)
{
  Internals->SetEmissionField(fieldName);
}

VTKM_CONT
vtkm::Bounds ConnectivityProxy::GetSpatialBounds()
{
  return Internals->GetSpatialBounds();
}

VTKM_CONT
vtkm::Range ConnectivityProxy::GetScalarFieldRange()
{
  return Internals->GetScalarFieldRange();
}

VTKM_CONT
void ConnectivityProxy::SetCompositeBackground(bool on)
{
  return Internals->SetCompositeBackground(on);
}

VTKM_CONT
void ConnectivityProxy::SetScalarRange(const vtkm::Range& range)
{
  Internals->SetScalarRange(range);
}

VTKM_CONT
vtkm::Range ConnectivityProxy::GetScalarRange()
{
  return Internals->GetScalarRange();
}

VTKM_CONT
void ConnectivityProxy::Trace(vtkm::rendering::raytracing::Ray<vtkm::Float64>& rays)
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("connectivity_trace_64");
  if (this->Internals->GetRenderMode() == RenderMode::Volume)
  {
    logger->AddLogData("volume_mode", "true");
  }
  else
  {
    logger->AddLogData("volume_mode", "false");
  }

  Internals->Trace(rays);
  logger->CloseLogEntry(-1.0);
}

VTKM_CONT
void ConnectivityProxy::PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float32> &rays,
                                     PartialVector32 &partials)
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("connectivity_trace_32");
  if (this->Internals->GetRenderMode() == RenderMode::Volume)
  {
    logger->AddLogData("volume_mode", "true");
  }
  else
  {
    logger->AddLogData("volume_mode", "false");
  }

  Internals->PartialTrace(rays, partials);

  logger->CloseLogEntry(-1.0);
}

VTKM_CONT
void ConnectivityProxy::Trace(vtkm::rendering::raytracing::Ray<vtkm::Float32>& rays)
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("connectivity_trace_32");
  if (this->Internals->GetRenderMode() == RenderMode::Volume)
  {
    logger->AddLogData("volume_mode", "true");
  }
  else
  {
    logger->AddLogData("volume_mode", "false");
  }

  Internals->Trace(rays);

  logger->CloseLogEntry(-1.0);
}

VTKM_CONT
void ConnectivityProxy::PartialTrace(vtkm::rendering::raytracing::Ray<vtkm::Float64> &rays,
                                     PartialVector64 &partials)
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("connectivity_trace_64");
  if (this->Internals->GetRenderMode() == RenderMode::Volume)
  {
    logger->AddLogData("volume_mode", "true");
  }
  else
  {
    logger->AddLogData("volume_mode", "false");
  }

  Internals->PartialTrace(rays, partials);

  logger->CloseLogEntry(-1.0);
}

VTKM_CONT
void ConnectivityProxy::Trace(const vtkm::rendering::Camera& camera,
                              vtkm::rendering::CanvasRayTracer* canvas)
{
  vtkm::rendering::raytracing::Logger* logger = vtkm::rendering::raytracing::Logger::GetInstance();
  logger->OpenLogEntry("connectivity_trace_32");
  logger->AddLogData("volume_mode", "true");

  Internals->Trace(camera, canvas);

  logger->CloseLogEntry(-1.0);
}

VTKM_CONT
void ConnectivityProxy::SetDebugPrints(bool on)
{
  Internals->SetDebugPrints(on);
}

VTKM_CONT
void ConnectivityProxy::SetEpsilon(vtkm::Float64 epsilon)
{
  Internals->SetEpsilon(epsilon);
}

VTKM_CONT
void ConnectivityProxy::SetUnitScalar(vtkm::Float32 unitScalar)
{
  Internals->SetUnitScalar(unitScalar);
}

VTKM_CONT
void ConnectivityProxy::SetDivideEmisByAbsorb(const bool divide_emis_by_absorb)
{
  Internals->SetDivideEmisByAbsorb(divide_emis_by_absorb);
}

}
} // namespace vtkm::rendering
