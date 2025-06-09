//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Ascent.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef rover_engine_h
#define rover_engine_h

#include <vtkm/rendering/ConnectivityProxy.h>
#include <vtkm/cont/ColorTable.h>

#include <rover_config.h>
#include <vtkm_typedefs.hpp>
#include <settings.hpp>

namespace rover
{

struct ArraySizeFunctor
{
  vtkm::Id  *m_size;
  ArraySizeFunctor(vtkm::Id *size)
    : m_size(size)
  {}
  
  template<typename T, typename Storage>
  void operator()(const vtkm::cont::ArrayHandle<T, Storage> &array) const
  {
    *m_size = array.GetNumberOfValues();
  } //operator
};

class Engine
{
public:
  Engine();
  ~Engine();

  void validate_tracer();
  void set_data_set(vtkmDataSet &);
  PartialVector32 partial_trace(Ray32 &rays);
  PartialVector64 partial_trace(Ray64 &rays);
  void init_rays(Ray32 &rays);
  void init_rays(Ray64 &rays);
  void set_primary_range(const vtkmRange &range);
  void set_composite_background(bool on);
  vtkmRange get_primary_range();
  int get_num_channels();

  void set_primary_field();
  void set_secondary_field();

  void set_samples(const vtkm::Bounds &global_bounds, const int &samples)
  {
    (void)samples;
    (void)global_bounds;
  }

  void set_color_table(const vtkmColorTable &color_map, int samples = 1024)
  {
    constexpr vtkm::Float32 conversionToFloatSpace = (1.0f / 255.0f);
    vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::UInt8, 4>> temp;

    color_map.Sample(samples, temp);
    m_color_map.Allocate(samples);
    auto portal = m_color_map.WritePortal();
    auto colorPortal = temp.ReadPortal();

    for (vtkm::Id i = 0; i < samples; ++i)
    {
      auto color = colorPortal.Get(i);
      vtkm::Vec<vtkm::Float32, 4> t(color[0] * conversionToFloatSpace,
                                    color[1] * conversionToFloatSpace,
                                    color[2] * conversionToFloatSpace,
                                    color[3] * conversionToFloatSpace);
      portal.Set(i, t);
    }

  }

  void set_color_map(const vtkmColorMap &color_map)
  {
    m_color_map = color_map;
  }

  vtkmColorMap get_color_map() const
  {
    return m_color_map;
  }
protected:
  vtkmColorMap m_color_map;
  vtkmDataSet m_data_set;
  vtkm::rendering::ConnectivityProxy *m_tracer;

  int detect_num_bins();
  template<typename Precision>
  void init_emission(vtkm::rendering::raytracing::Ray<Precision> &rays,
                     const int num_bins);
};

}; // namespace rover
#endif
