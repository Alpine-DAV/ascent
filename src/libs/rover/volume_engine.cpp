//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <volume_engine.hpp>
#include <rover_exceptions.hpp>
#include <utils/rover_logging.hpp>
namespace rover {

VolumeEngine::VolumeEngine()
{
  m_tracer = NULL;
  m_num_samples = 400;
}

VolumeEngine::~VolumeEngine()
{
  if(m_tracer) delete m_tracer;
}

void
VolumeEngine::set_data_set(vtkm::cont::DataSet &dataset)
{
  if(m_tracer) delete m_tracer;
  m_tracer = new vtkm::rendering::ConnectivityProxy(dataset);
}

int VolumeEngine::get_num_channels()
{
  return 4;
}

void
VolumeEngine::set_primary_field(const std::string &primary_field)
{
  m_primary_field = primary_field;
  m_tracer->SetScalarField(m_primary_field);
}

void
VolumeEngine::init_rays(Ray32 &rays)
{
  vtkm::cont::ArrayHandle<vtkm::Float32> signature;
  signature.Allocate(4);
  signature.WritePortal().Set(0,1.f);
  signature.WritePortal().Set(1,1.f);
  signature.WritePortal().Set(2,1.f);
  signature.WritePortal().Set(3,0.f);
  rays.Buffers.at(0).InitChannels(signature);
}

void
VolumeEngine::init_rays(Ray64 &rays)
{
  vtkm::cont::ArrayHandle<vtkm::Float64> signature;
  signature.Allocate(4);
  signature.WritePortal().Set(0,1.);
  signature.WritePortal().Set(1,1.);
  signature.WritePortal().Set(2,1.);
  signature.WritePortal().Set(3,0.);
  rays.Buffers.at(0).InitChannels(signature);
}

PartialVector32
VolumeEngine::partial_trace(Ray32 &rays)
{
  if(m_tracer == NULL)
  {
    std::cout<<"Volume Engine Error: must set the data set before tracing\n";
  }

  if(this->m_primary_field == "")
  {
    throw RoverException("Primary field is not set. Unable to render\n");
  }

  ROVER_INFO("tracing  rays");
  rays.Buffers.at(0).InitConst(0.);
  vtkmColorMap corrected = correct_opacity();
  m_tracer->SetColorMap(corrected);
  return m_tracer->PartialTrace(rays);
}

PartialVector64
VolumeEngine::partial_trace(Ray64 &rays)
{
  if(m_tracer == NULL)
  {
    std::cout<<"Volume Engine Error: must set the data set before tracing\n";
  }

  if(this->m_primary_field == "")
  {
    throw RoverException("Primary field is not set. Unable to render\n");
  }

  ROVER_INFO("tracing  rays");
  rays.Buffers.at(0).InitConst(0.);
  vtkmColorMap corrected = correct_opacity();
  m_tracer->SetColorMap(corrected);
  return m_tracer->PartialTrace(rays);
}

vtkmRange
VolumeEngine::get_primary_range()
{
  return m_tracer->GetScalarFieldRange();
}

vtkmColorMap
VolumeEngine::correct_opacity()
{
  const float correction_scalar = 10.f;
  float samples = m_num_samples;

  float ratio = correction_scalar / samples;
  vtkmColorMap corrected;
  corrected.Allocate(m_color_map.GetNumberOfValues());

  auto map_portal = m_color_map.ReadPortal();
  auto corr_portal = corrected.WritePortal();

  const int num_points = m_color_map.GetNumberOfValues();
#ifdef ROVER_OPENMP_ENABLED
    #pragma omp parallel for
#endif
  for(int i = 0; i < num_points; i++)
  {
    vtkm::Vec<vtkm::Float32,4> color = map_portal.Get(i);
    color[3] = 1.f - vtkm::Pow((1.f - color[3]), ratio);
    corr_portal.Set(i, color);
  }

  return corrected;
}

void
VolumeEngine::set_composite_background(bool on)
{
  m_tracer->SetCompositeBackground(on);
};

void
VolumeEngine::set_primary_range(const vtkmRange &range)
{
  return m_tracer->SetScalarRange(range);
}

void
VolumeEngine::set_samples(const vtkm::Bounds &global_bounds, const int &samples)
{
  const vtkm::Float32 num_samples = static_cast<float>(samples);
  vtkm::Vec<vtkm::Float32,3> totalExtent;
  totalExtent[0] = vtkm::Float32(global_bounds.X.Max - global_bounds.X.Min);
  totalExtent[1] = vtkm::Float32(global_bounds.Y.Max - global_bounds.Y.Min);
  totalExtent[2] = vtkm::Float32(global_bounds.Z.Max - global_bounds.Z.Min);
  vtkm::Float32 sample_distance = vtkm::Magnitude(totalExtent) / num_samples;
  m_tracer->SetSampleDistance(sample_distance);
  m_num_samples = samples;
}

}; //namespace rover
